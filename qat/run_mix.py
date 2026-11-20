"""Configure, launch and summarize the frozen four-run mixture experiment.

The launch action requires a successful, hash-bound preflight record. Downloaded
files never become training inputs merely because they finish downloading.
"""
import argparse
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[1]
DOMAINS = ['web_zh', 'web_en', 'math', 'code']
SPECS = [
    ('mix_v1_qat_web', 'qat', [0.5, 0.5, 0.0, 0.0]),
    ('mix_v1_qat_70_15_15', 'qat', [0.35, 0.35, 0.15, 0.15]),
    ('mix_v1_qat_50_25_25', 'qat', [0.25, 0.25, 0.25, 0.25]),
    ('mix_v1_bf16_70_15_15', 'bf16', [0.35, 0.35, 0.15, 0.15]),
]
TOTAL = 1024 * 1_048_576
REPORT = ROOT / 'reports/pretrain_v2/mix_v1'
GPUS_PER_RUN = 2
ALPHA_RAMP_FRAC = 0.1


def configs(data_dir):
    common = dict(model=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'), data=str(data_dir),
                  data_manifest=str(data_dir / 'manifest.json'), val_data=str(data_dir),
                  langs=DOMAINS, eval_langs=DOMAINS, seq_len=2048, micro_batch=8,
                  global_batch_tokens=1_048_576, total_tokens=TOTAL, lr=3e-3,
                  seed=20261110, strict_resume=True, max_data_passes=1.0,
                  warmup_frac=0.02, decay_frac=0.2, min_lr_ratio=0.1,
                  eval_every_tokens=128 * 1_048_576, save_every_tokens=0,
                  val_windows=256, chunked_ce=False, tensorboard=True,
                  kd_weight=0.0)
    return {name: {**common, 'mode': mode, 'weights': weights,
                   'alpha_ramp_frac': ALPHA_RAMP_FRAC if mode == 'qat' else 0.0}
            for name, mode, weights in SPECS}


def prepare_configs(data_dir):
    for name, cfg in configs(data_dir).items():
        path = ROOT / 'runs' / name / 'config.json'
        if path.exists():
            if json.loads(path.read_text()) != cfg:
                raise ValueError(f'Existing config differs: {path}')
        else:
            if path.parent.exists() and any(path.parent.iterdir()):
                raise ValueError(f'Nonempty run has no matching config: {path.parent}')
            atomic_json(path, cfg)
    return configs(data_dir)


def code_identity():
    # Moving data code out of qat/ must not silently remove it from preflight.
    paths = sorted(p for directory in ('qat', 'data_pipeline')
                   for p in (ROOT / directory).rglob('*.py')
                   if '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}


def experiment_identity(data_dir):
    manifest = data_dir / 'manifest.json'
    if not (data_dir / 'READY').exists() or not manifest.is_file():
        raise ValueError('Prepared data has not been published READY')
    return {'manifest_sha256': sha256_file(manifest), 'code': code_identity(),
            'configs': configs(data_dir)}


def check_idle_gpus():
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                            capture_output=True, text=True, check=True)
    if result.stdout.strip():
        raise RuntimeError('GPUs have active compute processes; refusing to displace them')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=index', '--format=csv,noheader'], text=True)
    if len(rows.splitlines()) != 8:
        raise RuntimeError('This experiment requires exactly eight visible GPUs')


def launch(data_dir):
    prepare_configs(data_dir)
    gate = json.loads((REPORT / 'preflight.json').read_text())
    if gate.get('status') != 'passed' or gate.get('identity') != experiment_identity(data_dir):
        raise ValueError('Preflight absent, failed, or stale; rerun tests and smoke checks')
    check_idle_gpus()
    status_path = REPORT / 'launch_status.json'
    status = {'started_unix': time.time(), 'runs': {}}
    children = []
    handles = []

    def stop(signum, frame):
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        for i, (name, _, _) in enumerate(SPECS):
            run = ROOT / 'runs' / name
            if (run / 'checkpoints/final/state.pt').exists():
                # A final file alone is not success: validate logs/config below.
                final_record(run)
                status['runs'][name] = {'status': 'already_done', 'exit_code': 0}
                continue
            gpus = list(range(GPUS_PER_RUN * i, GPUS_PER_RUN * (i + 1)))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(map(str, gpus)),
                       OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
            command = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
                       f'--nproc_per_node={GPUS_PER_RUN}', f'--master_port={29840+i}',
                       str(ROOT / 'qat/train.py'), '--config', str(run / 'config.json')]
            latest = run / 'checkpoints/latest/state.pt'
            if latest.exists():
                command += ['--resume', str(latest)]
            log = (run / 'train.log').open('a')
            handles.append(log)
            child = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            children.append(child)
            status['runs'][name] = {'status': 'running', 'pid': child.pid,
                                    'gpus': gpus, 'command': command}
            atomic_json(status_path, status)
            print(f'START {name} pid={child.pid} gpus={gpus}', flush=True)
        by_pid = {row['pid']: name for name, row in status['runs'].items() if 'pid' in row}
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(child.wait): child for child in children}
            for future in as_completed(futures):
                child = futures[future]
                code = future.result()
                name = by_pid[child.pid]
                status['runs'][name].update(exit_code=code, status='done' if code == 0 else 'failed',
                                           finished_unix=time.time())
                atomic_json(status_path, status)
                print(f'EXIT {name} code={code}', flush=True)
        failed = [name for name, row in status['runs'].items() if row['exit_code'] != 0]
        if failed:
            raise RuntimeError(f'Training failed: {failed}')
        result = summarize(data_dir)
        atomic_json(REPORT / 'selection.json', result)
        print(f'SELECTED {result["selected"]["checkpoint"]}', flush=True)
        evaluate_selected(result)
        status['status'] = 'done'
        status['finished_unix'] = time.time()
        atomic_json(status_path, status)
        print('DONE training, selection and downstream evaluation', flush=True)
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
    records = [json.loads(line) for line in (run / 'metrics.jsonl').read_text().splitlines() if line.strip()]
    if not any(r.get('event') == 'done' for r in records):
        raise ValueError(f'No successful completion record: {run}')
    key = 'val_ternary' if cfg['mode'] == 'qat' else 'val_float'
    matches = [r for r in records if r.get('tokens') == TOTAL and key in r]
    if not matches:
        raise ValueError(f'No final development result: {run}')
    rec = matches[-1]
    values = rec[key]
    if set(values) != set(DOMAINS) or not all(math.isfinite(v) for v in values.values()):
        raise ValueError(f'Incomplete/nonfinite development metrics: {run}')
    checkpoint = run / 'checkpoints/final/state.pt'
    if not checkpoint.exists():
        raise ValueError(f'Missing final checkpoint: {run}')
    return {'run': run.name, 'mode': cfg['mode'], 'tokens': TOTAL,
            'development_ce': values, 'macro_ce': sum(values.values()) / len(DOMAINS),
            'checkpoint': str(checkpoint), 'config_sha256': sha256_file(run / 'config.json')}


def summarize(data_dir):
    expected = configs(data_dir)
    rows = []
    for name, _, _ in SPECS:
        run = ROOT / 'runs' / name
        if json.loads((run / 'config.json').read_text()) != expected[name]:
            raise ValueError(f'Configuration mismatch: {name}')
        rows.append(final_record(run))
    qat = [r for r in rows if r['mode'] == 'qat']
    selected = min(qat, key=lambda r: r['macro_ce'])  # stable, predeclared order breaks ties
    return {'rule': 'QAT final only; min equal-weight four-domain development CE; fixed run order on tie',
            'manifest_sha256': sha256_file(data_dir / 'manifest.json'), 'selected': selected,
            'candidates': qat, 'bf16_control': rows[-1]}


def evaluate_selected(result):
    for gpu, label, row in [(0, 'selected_qat', result['selected']),
                            (1, 'bf16_control', result['bf16_control'])]:
        # Only these two final models are evaluated; no downstream-guided re-selection.
        command = [sys.executable, '-I', str(ROOT / 'qat/eval_downstream.py'),
                   '--ckpt', row['checkpoint'], '--precision', 'ternary' if gpu == 0 else 'float',
                   '--tasks', 'gsm8k', 'mmlu', 'cmmlu', 'ceval',
                   '--gsm-test-file', str(ROOT / 'eval_data/gsm8k/test.jsonl'),
                   '--gsm-fewshot-file', str(ROOT / 'data/gsm8k_sft_v1/fewshot.jsonl'),
                   '--gsm-shots', '8', '--gsm-batch', '64', '--gsm-max-new', '1024',
                   '--out', str(REPORT / f'{label}_downstream.json')]
        with (REPORT / f'{label}_downstream.log').open('a') as log:
            subprocess.run(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu),
                           TOKENIZERS_PARALLELISM='false'), stdin=subprocess.DEVNULL,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'EVAL_DONE {label}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['configure', 'launch', 'identity', 'summarize'])
    parser.add_argument('--data', type=Path, default=ROOT / 'data/mix_v1')
    args = parser.parse_args()
    data = args.data.resolve()
    if args.action == 'configure':
        prepare_configs(data)
    elif args.action == 'launch':
        launch(data)
    elif args.action == 'identity':
        print(json.dumps(experiment_identity(data), indent=2))
    else:
        print(json.dumps(summarize(data), indent=2))


if __name__ == '__main__':
    main()
