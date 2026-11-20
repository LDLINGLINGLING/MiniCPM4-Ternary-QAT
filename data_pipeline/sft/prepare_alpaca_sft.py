"""Prepare unified SFT dataset from Alpaca ZH/EN, HC3-Chinese, and Firefly.

Sources → unified format → decontamination (against MMLU/CMMLU/C-Eval/GSM8K) →
dev/train split → manifest.

Each output record:
  {"prompt": "<instruction>\\n<input>\\n", "response": "<output>", "source": "...", "id": "..."}

Prompt is not tokenized here; the trainer chooses the template at load time.
Loss mask: response + EOS only (prompt label = -100).
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.pretrain.decontam import ExclusionIndex, discover_benchmarks, questions
from data_pipeline.sft.gsm8k_data import atomic_json


FIREFLY_KEEP = {
    'Cot', 'OpenQA', 'MRC', 'Translation',
    'Composition', 'ClassicalChinese', 'Dictionary',
}
MAX_CHARS = 8000  # combined prompt+response chars; long tail wastes context


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def load_alpaca_csv(path, source_name):
    rows = list(csv.DictReader(open(path, encoding='utf-8')))
    out = []
    for i, r in enumerate(rows):
        instr = (r.get('instruction') or '').strip()
        inp = (r.get('input') or '').strip()
        resp = (r.get('output') or '').strip()
        if not instr or not resp:
            continue
        prompt = f'{instr}\n{inp}\n' if inp else f'{instr}\n'
        out.append({'prompt': prompt, 'response': resp,
                    'source': source_name, 'id': f'{source_name}_{i}'})
    return out


def load_hc3(paths, source_name='hc3'):
    out = []
    for path in paths:
        for i, line in enumerate(open(path, encoding='utf-8')):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            q = (r.get('question') or '').strip()
            answers = r.get('chatgpt_answers') or r.get('human_answers') or []
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers)
                except Exception:
                    answers = [answers]
            resp = (answers[0] if answers else '').strip()
            if not q or not resp:
                continue
            tag = Path(path).stem
            out.append({'prompt': f'{q}\n', 'response': resp,
                        'source': f'{source_name}/{tag}', 'id': f'{source_name}_{tag}_{i}'})
    return out


def load_firefly(path, source_name='firefly'):
    out = []
    for i, line in enumerate(open(path, encoding='utf-8', errors='replace')):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = r.get('kind', '')
        if FIREFLY_KEEP and kind not in FIREFLY_KEEP:
            continue
        inp = (r.get('input') or '').strip()
        tgt = (r.get('target') or '').strip()
        if not inp or not tgt:
            continue
        out.append({'prompt': f'{inp}\n', 'response': tgt,
                    'source': f'{source_name}/{kind}', 'id': f'{source_name}_{i}'})
    return out


def decontaminate(records, root):
    bench_files = discover_benchmarks(root)
    index = ExclusionIndex(list(questions(bench_files)))
    kept, removed = [], []
    for r in records:
        text = r['prompt'] + r['response']
        hit = index.match(text)
        if hit:
            removed.append({**r, 'hit': hit})
        else:
            kept.append(r)
    return kept, removed


def length_filter(records, max_chars):
    return [r for r in records if len(r['prompt']) + len(r['response']) <= max_chars]


def dedup(records):
    seen = set()
    out = []
    for r in records:
        key = _sha(r['prompt'] + r['response'])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--alpaca-zh', required=True)
    p.add_argument('--alpaca-en', required=True)
    p.add_argument('--hc3-dir', required=True)
    p.add_argument('--firefly', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--root', default=str(Path(__file__).resolve().parents[2]))
    p.add_argument('--dev-size', type=int, default=1000)
    p.add_argument('--seed', type=int, default=20261110)
    p.add_argument('--max-chars', type=int, default=MAX_CHARS)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)

    print('Loading sources...', flush=True)
    records = []
    records += load_alpaca_csv(args.alpaca_zh, 'alpaca_zh')
    print(f'  alpaca_zh: {len(records)}')
    n = len(records)
    records += load_alpaca_csv(args.alpaca_en, 'alpaca_en')
    print(f'  alpaca_en: {len(records)-n}')
    n = len(records)
    hc3_files = sorted(Path(args.hc3_dir).glob('*.jsonl'))
    records += load_hc3(hc3_files)
    print(f'  hc3: {len(records)-n}')
    n = len(records)
    records += load_firefly(args.firefly)
    print(f'  firefly: {len(records)-n}')
    print(f'  total raw: {len(records)}')

    print('Length filter...', flush=True)
    records = length_filter(records, args.max_chars)
    print(f'  after length filter: {len(records)}')

    print('Dedup...', flush=True)
    records = dedup(records)
    print(f'  after dedup: {len(records)}')

    print('Decontaminating...', flush=True)
    kept, removed = decontaminate(records, args.root)
    print(f'  kept: {len(kept)}  removed: {len(removed)}')
    atomic_json(out / 'removed.json', removed[:200])

    print('Splitting dev/train...', flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(kept)
    dev = kept[:args.dev_size]
    train = kept[args.dev_size:]

    # Use ensure_ascii=True to escape  /  Unicode line separators,
    # which splitlines() would otherwise treat as newlines and break JSONL readers.
    with (out / 'train.jsonl').open('w', encoding='utf-8') as fh:
        for r in train:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')
    with (out / 'dev.jsonl').open('w', encoding='utf-8') as fh:
        for r in dev:
            fh.write(json.dumps(r, ensure_ascii=True) + '\n')

    from collections import Counter
    srcs = Counter(r['source'].split('/')[0] for r in train)
    manifest = {
        'version': 1,
        'seed': args.seed,
        'dev_size': len(dev),
        'train_size': len(train),
        'sources': dict(srcs),
        'max_chars': args.max_chars,
        'decontam_removed': len(removed),
        'files': {
            'train': str((out / 'train.jsonl').resolve()),
            'dev': str((out / 'dev.jsonl').resolve()),
        }
    }
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
