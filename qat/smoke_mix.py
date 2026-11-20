"""Two-rank pretraining/resume smoke using explicitly synthetic four-domain data.

This fixture checks plumbing, not mathematical/code model quality. It never
enters the production corpus or the development-based mixture selection.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ['web_zh', 'web_en', 'math', 'code']


def setup(destination):
    if destination.exists():
        raise FileExistsError(f'Smoke outputs must be new: {destination}')
    data = destination / 'fixture'
    data.mkdir(parents=True)
    streams = {}
    for i, name in enumerate(DOMAINS):
        entries = {}
        lang = 'zh' if i % 2 == 0 else 'en'
        for split, windows in [('train', 160), ('val', 8)]:
            source = ROOT / f'data/ufw_v1/{lang}_{split}.bin'
            nbytes = windows * 2049 * 4
            with source.open('rb') as handle:
                handle.seek((i // 2) * nbytes)
                content = handle.read(nbytes)
            if len(content) != nbytes:
                raise ValueError(f'Fixture source too short: {source}')
            output = data / f'{name}_{split}.bin'
            output.write_bytes(content)
            entries[split] = {'path': output.name, 'sha256': sha256_file(output),
                              'tokens': windows * 2049}
        streams[name] = {'train': [entries['train']], 'val': entries['val']}
    atomic_json(data / 'manifest.json', {'version': 1, 'streams': streams,
                'purpose': 'Synthetic web-derived four-domain plumbing smoke ONLY'})
    atomic_json(data / 'READY', {'manifest_sha256': sha256_file(data / 'manifest.json')})
    cfg = dict(model=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'), data=str(data),
               data_manifest=str(data / 'manifest.json'), val_data=str(data),
               langs=DOMAINS, eval_langs=DOMAINS, weights=[0.35, 0.35, 0.15, 0.15],
               mode='qat', seq_len=2048, micro_batch=8, global_batch_tokens=65536,
               total_tokens=4 * 65536, lr=3e-3, alpha_ramp_frac=0.5,
               eval_every_tokens=4 * 65536, save_every_tokens=2 * 65536,
               val_windows=4, strict_resume=True, max_data_passes=1.0,
               seed=20261110, tensorboard=True, chunked_ce=False)
    for name in ['uninterrupted', 'resumed', 'bf16']:
        atomic_json(destination / name / 'config.json', {**cfg, 'mode': 'bf16' if name == 'bf16' else 'qat'})
    return destination


def run_one(destination, name, gpus, extra=()):
    run = destination / name
    command = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=2', str(ROOT / 'qat/train.py'), '--config', str(run / 'config.json'), *extra]
    with (run / 'train.log').open('a') as handle:
        subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus,
                       OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'), stdin=subprocess.DEVNULL,
                       stdout=handle, stderr=subprocess.STDOUT, check=True)


def compare(destination):
    import torch
    a = torch.load(destination / 'uninterrupted/checkpoints/final/state.pt', map_location='cpu',
                   weights_only=False, mmap=True)
    b = torch.load(destination / 'resumed/checkpoints/final/state.pt', map_location='cpu',
                   weights_only=False, mmap=True)
    if a['step'] != b['step'] or a['tokens'] != b['tokens'] or a['step'] != 4:
        raise ValueError('Resume step/token mismatch')
    if set(a['model']) != set(b['model']):
        raise ValueError('Parameter sets differ')
    delta_sq = total_sq = 0.0
    max_abs = 0.0
    for key in a['model']:
        x, y = a['model'][key].float(), b['model'][key].float()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError(f'Nonfinite weights: {key}')
        delta_sq += (x - y).square().sum().item()
        total_sq += x.square().sum().item()
        max_abs = max(max_abs, (x - y).abs().max().item())
    relative = (delta_sq / max(total_sq, 1e-30)) ** 0.5
    # BF16/flash-attention backward is nondeterministic. This is a smoke tolerance,
    # not a claim of bitwise numerical equivalence; CPU tests check exact RNG/state.
    if relative > 0.01:
        raise ValueError(f'Resume numerical divergence too large: {relative}')
    for state in (a, b):
        if not state.get('optimizer', {}).get('state'):
            raise ValueError('Missing optimizer state')
        if len(state.get('rng_by_rank', [])) != 2:
            raise ValueError('Missing rank RNG states')
    result = {'status': 'passed', 'step': a['step'], 'tokens': a['tokens'],
              'weight_relative_l2': relative, 'weight_max_abs': max_abs,
              'scope': '2-rank synthetic plumbing/resume smoke; not production model-quality evidence'}
    atomic_json(destination / 'comparison.json', result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['setup', 'run', 'compare'])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    destination = args.output.resolve()
    if args.action == 'setup':
        setup(destination)
    elif args.action == 'compare':
        compare(destination)
    else:
        run_one(destination, 'uninterrupted', '0,1')
        run_one(destination, 'resumed', '0,1', ['--max-steps', '2'])
        run_one(destination, 'resumed', '0,1', ['--resume', str(destination / 'resumed/checkpoints/latest/state.pt')])
        run_one(destination, 'bf16', '0,1')
        compare(destination)


if __name__ == '__main__':
    main()
