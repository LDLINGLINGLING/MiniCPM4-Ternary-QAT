"""500M-token ablation: quantize only MLP or only attention, compare with full-QAT baseline."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat import run_mix as mix
from data_pipeline.sft.gsm8k_data import atomic_json

ROOT = mix.ROOT
DATA = ROOT / 'data/mix_v2'

# 500M token / 1,048,576 per step = 477 steps (round down to nearest int)
TOTAL = 477 * 1_048_576  # 500,801,536

SPECS = [
    ('mix_v3_qat_mlp_only',  'qat', [0.35, 0.35, 0.15, 0.15],
     ['gate_proj', 'up_proj', 'down_proj']),
    ('mix_v3_qat_attn_only', 'qat', [0.35, 0.35, 0.15, 0.15],
     ['q_proj', 'k_proj', 'v_proj', 'o_proj']),
]

REPORT = ROOT / 'reports/pretrain_v2/mix_v3_ablation'
DOMAINS = mix.DOMAINS


def configs():
    manifest = str(DATA / 'manifest.json')
    common = dict(
        model=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'),
        data=str(DATA), data_manifest=manifest, val_data=str(DATA),
        langs=DOMAINS, eval_langs=DOMAINS, seq_len=2048, micro_batch=8,
        global_batch_tokens=1_048_576, total_tokens=TOTAL,
        lr=3e-3, seed=20261110, strict_resume=True, max_data_passes=1.0,
        warmup_frac=0.02, decay_frac=0.2, min_lr_ratio=0.1,
        eval_every_tokens=128 * 1_048_576, save_every_tokens=0,
        val_windows=256, chunked_ce=False, tensorboard=True, kd_weight=0.0,
        alpha_ramp_frac=0.3,
    )
    out = {}
    for name, mode, weights, suffixes in SPECS:
        cfg = {**common, 'mode': mode, 'weights': weights,
               'quant_suffixes': suffixes}
        out[name] = cfg
    return out


def prepare():
    REPORT.mkdir(parents=True, exist_ok=True)
    for name, cfg in configs().items():
        path = ROOT / 'runs' / name / 'config.json'
        if path.exists():
            if json.loads(path.read_text()) != cfg:
                raise ValueError(f'Existing config differs: {path}')
        else:
            if path.parent.exists() and any(path.parent.iterdir()):
                raise ValueError(f'Nonempty run directory has no matching config: {path.parent}')
            atomic_json(path, cfg)
    print('CONFIGS written', flush=True)


def launch():
    prepare()
    cfgs = configs()
    pids = {}
    logs = {}
    for i, (name, _, _, _) in enumerate(SPECS):
        gpus = f'{4*i},{4*i+1},{4*i+2},{4*i+3}'
        run = ROOT / 'runs' / name
        latest = run / 'checkpoints/latest/state.pt'
        final = run / 'checkpoints/final/state.pt'
        if final.exists():
            print(f'SKIP {name} (final checkpoint exists)', flush=True)
            continue
        cmd = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', f'--master_port={29860+i}',
               str(ROOT / 'qat/train.py'), '--config', str(run / 'config.json')]
        if latest.exists():
            cmd += ['--resume', str(latest)]
        log = (run / 'train.log').open('a')
        logs[name] = log
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus,
                   OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        pids[name] = proc
        print(f'START {name} pid={proc.pid} gpus={gpus}', flush=True)

    status = {'started': time.time(), 'runs': {}}
    for name, proc in pids.items():
        code = proc.wait()
        status['runs'][name] = {'exit_code': code, 'pid': proc.pid,
                                'status': 'done' if code == 0 else 'failed'}
        print(f'EXIT {name} code={code}', flush=True)
    for log in logs.values():
        log.close()
    atomic_json(REPORT / 'launch_status.json', status)
    if any(r['exit_code'] != 0 for r in status['runs'].values()):
        raise RuntimeError('One or more runs failed')
    print('ALL DONE', flush=True)


def main():
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['configure', 'launch'])
    args = p.parse_args()
    if args.action == 'configure':
        prepare()
    else:
        launch()


if __name__ == '__main__':
    main()
