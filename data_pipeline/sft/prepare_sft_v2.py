"""Build the v2 SFT mix: general 60% / code 25% / math 15% (by record count).

Sources:
  general  data/alpaca_sft_v1/train.jsonl          (already decontaminated)
  code     data/code_raw_v1/opc_annealing/algorithmic_corpus/*.jsonl
           + firefly 'Program' records
  math     data/gsm8k_sft_v1/train.jsonl           (repeated to hit the target share)

Output: data/sft_v2/{train,dev}.jsonl + manifest.json, unified format
  {"prompt": ..., "response": ..., "source": ..., "id": ...}
"""
import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.pretrain.decontam import ExclusionIndex, discover_benchmarks, questions
from data_pipeline.sft.gsm8k_data import atomic_json

ROOT = Path(__file__).resolve().parents[2]
GENERAL = ROOT / 'data/alpaca_sft_v1'
GSM = ROOT / 'data/gsm8k_sft_v1'
CODE_DIR = ROOT / 'data/code_raw_v1/opc_annealing/algorithmic_corpus'
FIREFLY = ROOT / 'data/sft_raw/firefly/firefly-train-1.1M.jsonl'
OUT = ROOT / 'data/sft_v2'

MAX_CHARS = 8000


def iter_jsonl(path):
    # File iteration splits only on \n/\r/\r\n, unlike str.splitlines() which
    # also breaks on U+2028/U+2029 inside JSON strings.
    with Path(path).open(encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_code(text):
    """Return (instruction, code) or None.

    Python-style records fence the code with ```; bash/cpp/etc. put the
    instruction on the first line followed directly by code.
    """
    t = text.strip()
    if not t:
        return None
    if '```' in t:
        i = t.index('```')
        prompt, response = t[:i].strip(), t[i:].strip()
    else:
        parts = t.split('\n', 1)
        if len(parts) < 2:
            return None
        prompt, response = parts[0].strip(), parts[1].strip()
    if not prompt or not response:
        return None
    if len(prompt) + len(response) > MAX_CHARS:
        return None
    return prompt, response


def load_general(limit, rng, general_dir=GENERAL):
    if limit < 0:
        raise ValueError(f'General quota is negative ({limit}); increase --total or reduce math/code')
    rows = list(iter_jsonl(general_dir / 'train.jsonl'))
    rng.shuffle(rows)
    return rows[:limit]


def load_code(quota, rng, code_dir=CODE_DIR):
    if quota < 0:
        raise ValueError(f'OPC quota is negative ({quota}); Firefly Program already exceeds the code target')
    if quota == 0:
        return []
    files = sorted(code_dir.glob('*.jsonl'))
    by_lang = collections.defaultdict(list)
    for f in files:
        with f.open(encoding='utf-8', errors='replace') as fh:
            first = fh.readline().strip()
        if first:
            by_lang[json.loads(first).get('lang', 'unknown')].append(f)

    langs = sorted(by_lang)
    if not langs:
        raise ValueError(f'No nonempty OPC JSONL shards in {code_dir}')
    per_lang = quota // len(langs)
    out = []
    for lang in langs:
        paths = by_lang[lang][:]
        rng.shuffle(paths)
        got = 0
        for path in paths:
            if got >= per_lang:
                break
            for r in iter_jsonl(path):
                if r.get('lang') != lang:
                    continue
                split = split_code(r.get('text', ''))
                if split is None:
                    continue
                prompt, response = split
                out.append({'prompt': prompt + '\n', 'response': response,
                            'source': f'code/{lang}', 'id': f'code_{lang}_{got}'})
                got += 1
                if got >= per_lang:
                    break
        print(f'  code/{lang}: {got}', flush=True)
    rng.shuffle(out)
    return out


def load_firefly_program(firefly=FIREFLY):
    out = []
    for i, r in enumerate(iter_jsonl(firefly)):
        if r.get('kind') != 'Program':
            continue
        inp = (r.get('input') or '').strip()
        tgt = (r.get('target') or '').strip()
        if not inp or not tgt or len(inp) + len(tgt) > MAX_CHARS:
            continue
        out.append({'prompt': inp + '\n', 'response': tgt,
                    'source': 'code/firefly_Program', 'id': f'firefly_program_{i}'})
    return out


def load_math(repeat, gsm_dir=GSM):
    base = [{'prompt': r['question'].strip() + '\n', 'response': r['answer'].strip(),
             'source': 'gsm8k/train', 'id': r.get('source_id', '')}
            for r in iter_jsonl(gsm_dir / 'train.jsonl')]
    out = []
    for rep in range(repeat):
        for r in base:
            out.append({**r, 'id': f'{r["id"]}_r{rep}'})
    return out


def decontaminate(records, root=ROOT):
    index = ExclusionIndex(list(questions(discover_benchmarks(root))))
    kept, removed = [], 0
    for r in records:
        if index.match(r['prompt'] + r['response']):
            removed += 1
        else:
            kept.append(r)
    return kept, removed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--general', type=Path, default=GENERAL, help='Unified general SFT directory')
    p.add_argument('--gsm', type=Path, default=GSM, help='Prepared GSM8K directory')
    p.add_argument('--code-dir', type=Path, default=CODE_DIR, help='Directory of OPC JSONL shards')
    p.add_argument('--firefly', type=Path, default=FIREFLY, help='Firefly JSONL; Program rows supplement code')
    p.add_argument('--benchmark-root', type=Path, default=ROOT, help='Root containing eval_data/')
    p.add_argument('--output', type=Path, default=OUT, help='New output directory')
    p.add_argument('--total', type=int, default=500_000)
    p.add_argument('--code-frac', type=float, default=0.25)
    p.add_argument('--math-frac', type=float, default=0.15)
    p.add_argument('--math-repeat', type=int, default=11)
    p.add_argument('--seed', type=int, default=20261110)
    args = p.parse_args()
    if args.total < 1 or args.math_repeat < 1:
        p.error('--total and --math-repeat must be positive')
    if args.code_frac < 0 or args.math_frac < 0 or args.code_frac + args.math_frac >= 1:
        p.error('code/math fractions must be nonnegative and sum to less than one')
    out = args.output

    if out.exists():
        raise SystemExit(f'{out} already exists; choose a new output directory')
    rng = random.Random(args.seed)

    print('Loading math...', flush=True)
    math = load_math(args.math_repeat, args.gsm)
    print(f'  math: {len(math)}', flush=True)

    print('Loading firefly Program...', flush=True)
    program = load_firefly_program(args.firefly)
    print(f'  firefly Program: {len(program)}', flush=True)

    code_quota = int(args.total * args.code_frac) - len(program)
    print(f'Loading code (quota {code_quota})...', flush=True)
    code = program + load_code(code_quota, rng, args.code_dir)
    print(f'  code total: {len(code)}', flush=True)

    general_quota = args.total - len(math) - len(code)
    print(f'Loading general (quota {general_quota})...', flush=True)
    general = load_general(general_quota, rng, args.general)
    print(f'  general: {len(general)}', flush=True)

    print('Decontaminating code...', flush=True)
    code, removed = decontaminate(code, args.benchmark_root)
    print(f'  code kept {len(code)}  removed {removed}', flush=True)

    train = general + code + math
    if not train:
        raise ValueError('No SFT records remained; check input directories and quotas')
    rng.shuffle(train)

    dev = list(iter_jsonl(args.general / 'dev.jsonl'))
    dev += [{'prompt': r['question'].strip() + '\n', 'response': r['answer'].strip(),
             'source': 'gsm8k/dev', 'id': r.get('source_id', '')}
            for r in iter_jsonl(args.gsm / 'dev.jsonl')]
    rng.shuffle(dev)

    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train', train), ('dev', dev)):
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=True) + '\n')

    cat = collections.Counter(r['source'].split('/')[0] for r in train)
    src = collections.Counter(r['source'] for r in train)
    manifest = {
        'version': 2,
        'seed': args.seed,
        'target': {'general': 1 - args.code_frac - args.math_frac,
                   'code': args.code_frac, 'math': args.math_frac},
        'train_size': len(train),
        'dev_size': len(dev),
        'categories': dict(cat),
        'category_share': {k: v / len(train) for k, v in cat.items()},
        'sources': dict(src),
        'math_repeat': args.math_repeat,
        'code_decontam_removed': removed,
        'files': {'train': str((out / 'train.jsonl').resolve()),
                  'dev': str((out / 'dev.jsonl').resolve())},
    }
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
