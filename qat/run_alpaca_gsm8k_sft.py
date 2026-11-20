"""Launcher: QAT SFT on the GSM8K-mixed general-instruction set, then evaluate.

Continues from the mix_v2 QAT GSM8K-SFT epoch_4 checkpoint, trains on
data/alpaca_gsm8k_sft_v1 (general data + GSM8K repeated 3x), selects the best
epoch by dev_ce, and evaluates on GSM8K + MMLU + CMMLU + C-Eval.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/pretrain_v2/mix_v2_alpaca_gsm8k_sft'
DATA_OUT = ROOT / 'data/alpaca_gsm8k_sft_v1'
INIT_CKPT = ROOT / 'runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt'
RUN_DIR = ROOT / 'runs/mix_v2_qat_alpaca_gsm8k_sft'


def run(cmd, log_path=None, **kw):
    print(f'RUN {" ".join(str(c) for c in cmd)}', flush=True)
    kw.setdefault('cwd', ROOT)
    kw.setdefault('stdin', subprocess.DEVNULL)
    if log_path:
        with Path(log_path).open('a') as fh:
            subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, check=True, **kw)
    else:
        subprocess.run(cmd, check=True, **kw)


def regression_tests():
    run([sys.executable, '-I', '-m', 'pytest', 'tests', '-q'],
        log_path=REPORT / 'tests.log')
    print('TESTS PASSED', flush=True)


def train():
    if RUN_DIR.exists() and (RUN_DIR / 'checkpoints/final/state.pt').exists():
        print('TRAIN already done, skipping', flush=True)
        return
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3,4,5,6,7',
               OMP_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false')
    cmd = [sys.executable, '-I', '-m', 'torch.distributed.run', '--standalone',
           '--nproc_per_node=8', '--master_port=29871',
           'qat/train_alpaca_sft.py',
           '--data-dir', str(DATA_OUT),
           '--run-dir', str(RUN_DIR),
           '--init-ckpt', str(INIT_CKPT),
           '--init-mode', 'qat',
           '--lr', '5e-5',
           '--epochs', '3',
           '--global-batch', '128',
           '--micro-batch', '2',
           '--no-gradient-checkpointing']
    latest = RUN_DIR / 'checkpoints/latest/state.pt'
    if latest.exists():
        cmd += ['--resume', str(latest)]
    run(cmd, log_path=REPORT / 'train.log', env=env)
    print('TRAIN DONE', flush=True)


def evaluate():
    rows = [json.loads(l) for l in (RUN_DIR / 'metrics.jsonl').read_text().splitlines() if l.strip()]
    epochs = [r for r in rows if r.get('event') == 'epoch_complete']
    if not epochs:
        raise ValueError('No epoch_complete records found')
    best = min(epochs, key=lambda r: r['dev_ce'])
    epoch_num = best['epoch']
    ckpt = RUN_DIR / f'snapshots/epoch_{epoch_num}/model.pt'
    print(f'BEST EPOCH {epoch_num} dev_ce={best["dev_ce"]:.4f} ckpt={ckpt}', flush=True)
    atomic_json(REPORT / 'best_epoch.json', {'epoch': epoch_num, **best})

    out_json = REPORT / f'epoch_{epoch_num}_downstream.json'
    if out_json.exists():
        print('EVAL already done', flush=True)
    else:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TOKENIZERS_PARALLELISM='false')
        run([sys.executable, '-I', 'qat/eval_downstream.py',
             '--ckpt', str(ckpt), '--precision', 'ternary',
             '--tasks', 'gsm8k', 'mmlu', 'cmmlu', 'ceval',
             '--gsm-test-file', str(ROOT / 'eval_data/gsm8k/test.jsonl'),
             '--gsm-fewshot-file', str(ROOT / 'data/gsm8k_sft_v1/fewshot.jsonl'),
             '--gsm-max-new', '512',
             '--out', str(out_json)],
            log_path=REPORT / f'eval_epoch_{epoch_num}.log',
            env=env)
        print('EVAL DONE', flush=True)

    d = json.loads(out_json.read_text())['tasks']
    print(json.dumps({k: {'strict_acc': round(d[k].get('strict_acc', d[k]['acc']), 4),
                           'n': d[k]['n']} for k in d}, indent=2))


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    if not (DATA_OUT / 'manifest.json').exists():
        raise SystemExit(f'Missing mixed dataset: {DATA_OUT}')
    regression_tests()
    train()
    evaluate()
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
