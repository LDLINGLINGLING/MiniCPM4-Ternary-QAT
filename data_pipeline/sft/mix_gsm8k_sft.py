"""Mix GSM8K SFT data into the general-instruction SFT set.

Reads data/alpaca_sft_v1 (unified format) and data/gsm8k_sft_v1 ({question, answer}),
repeats GSM8K train REPEAT times, and writes data/alpaca_gsm8k_sft_v1.

GSM8K is repeated so that after EPOCHS passes each GSM8K record is seen
REPEAT*EPOCHS times, compensating for dilution by the general data.
"""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.sft.gsm8k_data import atomic_json

ROOT = Path(__file__).resolve().parents[2]
ALPACA = ROOT / 'data/alpaca_sft_v1'
GSM = ROOT / 'data/gsm8k_sft_v1'
OUT = ROOT / 'data/alpaca_gsm8k_sft_v1'


def load_jsonl(path):
    # Iterate the file handle: Python splits only on \n/\r/\r\n, unlike
    # str.splitlines() which also breaks on U+2028/U+2029 inside JSON strings.
    out = []
    with Path(path).open(encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def gsm_to_record(row, tag):
    return {'prompt': row['question'].strip() + '\n',
            'response': row['answer'].strip(),
            'source': f'gsm8k/{tag}',
            'id': f'gsm8k_{row.get("source_id", tag)}'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alpaca', type=Path, default=ALPACA, help='Unified general SFT directory')
    p.add_argument('--gsm', type=Path, default=GSM, help='Prepared GSM8K directory')
    p.add_argument('--output', type=Path, default=OUT, help='New output directory')
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261110)
    args = p.parse_args()
    if args.repeat < 1:
        p.error('--repeat must be positive')
    out = args.output

    if out.exists():
        raise SystemExit(f'{out} already exists; choose a new output directory')

    alpaca_train = load_jsonl(args.alpaca / 'train.jsonl')
    alpaca_dev = load_jsonl(args.alpaca / 'dev.jsonl')
    gsm_train = [gsm_to_record(r, 'train') for r in load_jsonl(args.gsm / 'train.jsonl')]
    gsm_dev = [gsm_to_record(r, 'dev') for r in load_jsonl(args.gsm / 'dev.jsonl')]

    # Repeated copies get distinct ids so downstream tooling can tell them apart.
    gsm_repeated = []
    for rep in range(args.repeat):
        for r in gsm_train:
            gsm_repeated.append({**r, 'id': f'{r["id"]}_r{rep}'})

    train = alpaca_train + gsm_repeated
    dev = alpaca_dev + gsm_dev
    rng = random.Random(args.seed)
    rng.shuffle(train)
    rng.shuffle(dev)

    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train', train), ('dev', dev)):
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=True) + '\n')

    manifest = {
        'version': 1,
        'seed': args.seed,
        'repeat': args.repeat,
        'train_size': len(train),
        'dev_size': len(dev),
        'sources': {
            'alpaca_sft_v1': len(alpaca_train),
            'gsm8k_train': len(gsm_train),
            'gsm8k_repeated': len(gsm_repeated),
        },
        'gsm8k_share': len(gsm_repeated) / len(train),
        'files': {'train': str((out / 'train.jsonl').resolve()),
                  'dev': str((out / 'dev.jsonl').resolve())},
    }
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
