"""Two four-GPU runs over mix_v2, with a hash-bound preflight before launch."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat import run_mix as mix
from data_pipeline.sampling import MixedSampler, TokenStream, load_manifest
from data_pipeline.sft.gsm8k_data import atomic_json

mix.SPECS = [
    ('mix_v2_qat_70_15_15', 'qat', [0.35, 0.35, 0.15, 0.15]),
    ('mix_v2_bf16_70_15_15', 'bf16', [0.35, 0.35, 0.15, 0.15]),
]
# Whole optimizer updates; 2,000,683,008 tokens per run.
mix.TOTAL = 1908 * 1_048_576
mix.REPORT = mix.ROOT / 'reports/pretrain_v2/mix_v2'
mix.GPUS_PER_RUN = 4
mix.ALPHA_RAMP_FRAC = 0.3
DATA = mix.ROOT / 'data/mix_v2'


def smoke(name, cfg, gpus):
    run = mix.REPORT / 'smoke' / name
    if run.exists():
        raise FileExistsError(f'Smoke output must be new: {run}')
    # Same full training schedule, batch, precision and four-rank topology;
    # only validation is shortened. Never resume smoke into production.
    atomic_json(run / 'config.json', {**cfg, 'val_windows': 4})
    command = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', str(mix.ROOT / 'qat/train.py'),
               '--config', str(run / 'config.json'), '--max-steps', '2']
    with (run / 'train.log').open('w') as log:
        subprocess.run(command, cwd=mix.ROOT, check=True, stdin=subprocess.DEVNULL,
                       stdout=log, stderr=subprocess.STDOUT,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus, OMP_NUM_THREADS='4',
                                TOKENIZERS_PARALLELISM='false'))
    rows = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines()]
    start = next(r for r in rows if r.get('event') == 'start')
    assert start['world'] == 4 and start['accum'] == 16
    assert rows[-1]['event'] == 'paused' and rows[-1]['step'] == 2
    losses = [r['loss_ce'] for r in rows if 'loss_ce' in r]
    assert losses and all(math.isfinite(v) for v in losses)
    key = 'val_ternary' if cfg['mode'] == 'qat' else 'val_float'
    validation = [r[key] for r in rows if key in r][-1]
    assert set(validation) == set(mix.DOMAINS)
    assert all(math.isfinite(v) for v in validation.values())
    return {'status': 'passed', 'world': 4, 'steps': 2, 'losses': losses,
            'validation': validation, 'log': str(run / 'train.log')}


def preflight():
    mix.REPORT.mkdir(parents=True, exist_ok=True)
    configs = mix.prepare_configs(DATA)
    identity = mix.experiment_identity(DATA)
    with (mix.REPORT / 'tests.log').open('w') as log:
        subprocess.run([sys.executable, '-I', '-m', 'pytest', str(mix.ROOT / 'tests'), '-q'],
                       cwd=mix.ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PASS regression tests', flush=True)
    manifest = load_manifest(DATA / 'manifest.json')
    schedules = {}
    for name, cfg in configs.items():
        sampler = MixedSampler(cfg['data'], cfg['langs'], cfg['weights'], cfg['seq_len'],
                               cfg['seed'], manifest=manifest, canonical_order=True)
        n = cfg['total_tokens'] // cfg['seq_len']
        sampler.plan(n, max_data_passes=1.0)
        schedules[name] = sampler.usage(n)
        for domain in mix.DOMAINS:
            stream = TokenStream([manifest['streams'][domain]['val']], cfg['seq_len'])
            assert stream.num_windows >= cfg['val_windows']
    atomic_json(mix.REPORT / 'schedule.json', schedules)
    print('PASS all data SHA256 and full schedules', flush=True)
    mix.check_idle_gpus()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {name: pool.submit(smoke, name, cfg, '0,1,2,3' if i == 0 else '4,5,6,7')
                for i, (name, cfg) in enumerate(configs.items())}
        results = {name: job.result() for name, job in jobs.items()}
    if identity != mix.experiment_identity(DATA):
        raise ValueError('Experiment changed while preflight was running')
    atomic_json(mix.REPORT / 'preflight.json', {'status': 'passed', 'identity': identity,
                'schedules': schedules, 'smoke': results,
                'scope': 'Regression tests, full corpus hashes/schedules, two-step four-rank smoke per mode'})
    print('PASS four-GPU QAT and BF16 smoke', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['configure', 'preflight', 'launch', 'all'])
    args = parser.parse_args()
    if args.action == 'configure':
        mix.prepare_configs(DATA)
    if args.action in ('preflight', 'all'):
        preflight()
    if args.action in ('launch', 'all'):
        mix.launch(DATA)


if __name__ == '__main__':
    main()
