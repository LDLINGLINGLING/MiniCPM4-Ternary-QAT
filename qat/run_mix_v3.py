"""BitCPM4-style QAT: per-group-32 ternary weights, two-arm controlled comparison.

Two 4-GPU arms of 2,248,144,896 tokens each (2,144 updates), so every domain
stays at or below one data pass. Total training budget 4.5B tokens.

Arm A changes ONLY the quantization granularity versus mix_v2 (per-group-32
instead of per-row), keeping lr 3e-3 and alpha_ramp_frac 0.3, so A vs mix_v2
isolates the granularity effect.

Arm B additionally adopts the BitCPM4 stage-2 learning-rate recipe (lr 5e-3 with
a longer re-warmup), so B vs A isolates the recipe effect.

Reference for both: mix_v2 QAT reached four-domain ternary development CE 1.897
at 2.001B tokens with the per-row quantizer.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat import run_mix as mix
from data_pipeline.sampling import MixedSampler, TokenStream, load_manifest
from data_pipeline.sft.gsm8k_data import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/mix_v2'
REPORT = ROOT / 'reports/pretrain_v2/mix_v3_bitcpm'
STEPS = 2144
TOTAL = STEPS * 1_048_576
GPUS_PER_RUN = 4
WEIGHTS = [0.35, 0.35, 0.15, 0.15]

SPECS = [
    # name, quant_group, lr, warmup_frac, note
    ('mix_v3_bitcpm_pg32', 32, 3e-3, 0.02, 'per-group-32 only'),
    ('mix_v3_bitcpm_pg32_lr5e3', 32, 5e-3, 0.05, 'per-group-32 plus BitCPM4 stage-2 LR'),
]


def configs():
    common = dict(model=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'), data=str(DATA),
                  data_manifest=str(DATA / 'manifest.json'), val_data=str(DATA),
                  langs=mix.DOMAINS, eval_langs=mix.DOMAINS, seq_len=2048, micro_batch=8,
                  global_batch_tokens=1_048_576, total_tokens=TOTAL, mode='qat',
                  weights=WEIGHTS, seed=20261110, strict_resume=True, max_data_passes=1.0,
                  decay_frac=0.2, min_lr_ratio=0.1, alpha_ramp_frac=0.3,
                  eval_every_tokens=128 * 1_048_576, save_every_tokens=0,
                  val_windows=256, chunked_ce=False, tensorboard=True, kd_weight=0.0)
    return {name: {**common, 'quant_group': group, 'lr': lr, 'warmup_frac': warmup}
            for name, group, lr, warmup, _ in SPECS}


def prepare_configs():
    for name, cfg in configs().items():
        path = ROOT / 'runs' / name / 'config.json'
        if path.exists():
            if json.loads(path.read_text()) != cfg:
                raise ValueError(f'Existing config differs: {path}')
        else:
            if path.parent.exists() and any(path.parent.iterdir()):
                raise ValueError(f'Nonempty run has no matching config: {path.parent}')
            atomic_json(path, cfg)
    return configs()


def identity():
    manifest = DATA / 'manifest.json'
    if not (DATA / 'READY').exists() or not manifest.is_file():
        raise ValueError('Prepared data has not been published READY')
    return {'manifest_sha256': mix.sha256_file(manifest),
            'code': mix.code_identity(),
            'configs': configs()}


def preflight():
    REPORT.mkdir(parents=True, exist_ok=True)
    prepare_configs()
    ident = identity()
    with (REPORT / 'tests.log').open('w') as log:
        subprocess.run([sys.executable, '-I', '-m', 'pytest', str(ROOT / 'tests'), '-q'],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('PASS regression tests', flush=True)
    manifest = load_manifest(DATA / 'manifest.json')
    schedules = {}
    for name, cfg in configs().items():
        sampler = MixedSampler(cfg['data'], cfg['langs'], cfg['weights'], cfg['seq_len'],
                               cfg['seed'], manifest=manifest, canonical_order=True)
        n = cfg['total_tokens'] // cfg['seq_len']
        usage = sampler.plan(n, max_data_passes=cfg['max_data_passes'])
        schedules[name] = usage
        # Every domain must be covered by the fixed validation windows.
        for domain in mix.DOMAINS:
            stream = TokenStream([manifest['streams'][domain]['val']], cfg['seq_len'])
            assert stream.num_windows >= cfg['val_windows'], (name, domain)
    atomic_json(REPORT / 'schedule.json', schedules)
    print('PASS data hashes and full schedules', flush=True)

    def smoke(name, cfg, gpus):
        run = REPORT / 'smoke' / name
        if run.exists():
            raise FileExistsError(f'Smoke output must be new: {run}')
        atomic_json(run / 'config.json', {**cfg, 'val_windows': 4})
        cmd = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
               '--nproc_per_node=4', str(ROOT / 'qat/train.py'),
               '--config', str(run / 'config.json'), '--max-steps', '2']
        with (run / 'train.log').open('w') as log:
            subprocess.run(cmd, cwd=ROOT, check=True, stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT,
                           env=dict(os.environ, CUDA_VISIBLE_DEVICES=gpus,
                                    OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'))
        rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines()]
        start = next(r for r in rows if r.get('event') == 'start')
        assert start['world'] == 4, start
        assert rows[-1]['event'] == 'paused' and rows[-1]['step'] == 2, rows[-1]
        losses = [r['loss_ce'] for r in rows if 'loss_ce' in r]
        assert losses and all(math.isfinite(v) for v in losses), losses
        validation = [r['val_ternary'] for r in rows if 'val_ternary' in r][-1]
        assert set(validation) == set(mix.DOMAINS), validation
        assert all(math.isfinite(v) for v in validation.values()), validation
        return {'status': 'passed', 'steps': 2, 'losses': losses, 'validation': validation}

    mix.check_idle_gpus()
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {name: pool.submit(smoke, name, cfg, '0,1,2,3' if i == 0 else '4,5,6,7')
                for i, (name, cfg) in enumerate(configs().items())}
        results = {name: job.result() for name, job in jobs.items()}
    if ident != identity():
        raise ValueError('Experiment changed while preflight was running')
    atomic_json(REPORT / 'preflight.json',
                {'status': 'passed', 'identity': ident, 'schedules': schedules,
                 'smoke': results,
                 'scope': 'Regression tests, data hashes and schedules, two-step four-rank smoke per arm'})
    print('PASS four-GPU smoke on both arms', flush=True)


def launch():
    prepare_configs()
    gate = json.loads((REPORT / 'preflight.json').read_text())
    if gate.get('status') != 'passed' or gate.get('identity') != identity():
        raise ValueError('Preflight absent, failed, or stale; rerun tests and smoke checks')
    mix.check_idle_gpus()
    status_path = REPORT / 'launch_status.json'
    status = {'started_unix': time.time(), 'runs': {}}
    children, handles = [], []

    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for i, (name, _, _, _, _) in enumerate(SPECS):
            run = ROOT / 'runs' / name
            if (run / 'checkpoints/final/state.pt').exists():
                final_record(run)
                status['runs'][name] = {'status': 'already_done', 'exit_code': 0}
                continue
            gpus = list(range(GPUS_PER_RUN * i, GPUS_PER_RUN * (i + 1)))
            cmd = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
                   f'--nproc_per_node={GPUS_PER_RUN}', f'--master_port={29844 + i}',
                   str(ROOT / 'qat/train.py'), '--config', str(run / 'config.json')]
            latest = run / 'checkpoints/latest/state.pt'
            if latest.exists():
                cmd += ['--resume', str(latest)]
            log = (run / 'train.log').open('a')
            handles.append(log)
            child = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                                              OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false'))
            children.append(child)
            status['runs'][name] = {'status': 'running', 'pid': child.pid, 'gpus': gpus,
                                    'command': cmd}
            atomic_json(status_path, status)
            print(f'START {name} pid={child.pid} gpus={gpus}', flush=True)
        by_pid = {row['pid']: name for name, row in status['runs'].items() if 'pid' in row}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(child.wait): child for child in children}
            for future in as_completed(futures):
                child = futures[future]
                code = future.result()
                name = by_pid[child.pid]
                status['runs'][name].update(exit_code=code,
                                            status='done' if code == 0 else 'failed',
                                            finished_unix=time.time())
                atomic_json(status_path, status)
                print(f'EXIT {name} code={code}', flush=True)
        failed = [n for n, r in status['runs'].items() if r['exit_code'] != 0]
        if failed:
            raise RuntimeError(f'Training failed: {failed}')
        status['status'] = 'done'
        status['finished_unix'] = time.time()
        atomic_json(status_path, status)
        print('DONE training', flush=True)
    except BaseException:
        status['status'] = 'failed_or_interrupted'
        atomic_json(status_path, status)
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise
    finally:
        for handle in handles:
            handle.close()


def final_record(run):
    cfg = json.loads((run / 'config.json').read_text())
    rows = [json.loads(l) for l in (run / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    if not any(r.get('event') == 'done' for r in rows):
        raise ValueError(f'No successful completion record: {run}')
    matches = [r for r in rows if r.get('tokens') == TOTAL and 'val_ternary' in r]
    if not matches:
        raise ValueError(f'No final development result: {run}')
    values = matches[-1]['val_ternary']
    if set(values) != set(mix.DOMAINS) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f'Incomplete/nonfinite development metrics: {run}')
    ckpt = run / 'checkpoints/final/state.pt'
    if not ckpt.exists():
        raise ValueError(f'Missing final checkpoint: {run}')
    return {'run': run.name, 'quant_group': cfg['quant_group'], 'lr': cfg['lr'],
            'warmup_frac': cfg['warmup_frac'], 'tokens': TOTAL,
            'development_ce': values,
            'macro_ce': sum(values.values()) / len(mix.DOMAINS),
            'checkpoint': str(ckpt), 'config_sha256': sha256_file(run / 'config.json')}


def summarize():
    expected = configs()
    rows = []
    for name, _, _, _, _ in SPECS:
        run = ROOT / 'runs' / name
        if json.loads((run / 'config.json').read_text()) != expected[name]:
            raise ValueError(f'Configuration mismatch: {name}')
        rows.append(final_record(run))
    # No selection: both arms are reported. Comparing them is the point.
    result = {'rule': 'report both arms; arm A isolates granularity vs mix_v2, arm B adds the LR recipe',
              'reference': {'run': 'mix_v2_qat_70_15_15 (per-row, 2.001B tokens)', 'macro_ce': 1.897},
              'manifest_sha256': sha256_file(DATA / 'manifest.json'), 'arms': rows}
    atomic_json(REPORT / 'summary.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=['configure', 'preflight', 'launch', 'summarize'])
    args = p.parse_args()
    if args.action == 'configure':
        print(json.dumps(prepare_configs(), indent=2))
    elif args.action == 'preflight':
        preflight()
    elif args.action == 'launch':
        launch()
    else:
        print(json.dumps(summarize(), indent=2))


if __name__ == '__main__':
    main()
