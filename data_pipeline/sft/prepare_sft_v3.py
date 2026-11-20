"""Build the v3 SFT mix: general 60% / code 25% / math 15%, with a broader math set.

General and code are copied verbatim from data/sft_v2 so the only variable
versus v2 is the composition of the math category.

Math = GSM8K (repeated 2x) + UltraData-Math-L3 QA-Synthetic.

Output: data/sft_v3/{train,dev}.jsonl + manifest.json
"""
import argparse
import collections
import glob
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from data_pipeline.pretrain.decontam import ExclusionIndex, discover_benchmarks, questions
from data_pipeline.sft.gsm8k_data import atomic_json

ROOT = Path(__file__).resolve().parents[2]
V2 = ROOT / 'data/sft_v2'
GSM = ROOT / 'data/gsm8k_sft_v1'
ULTRA = ROOT / 'data/math_raw_v1/ultradata_math_l3/data/UltraData-Math-L3/QA-Synthetic'
OUT = ROOT / 'data/sft_v3'

MAX_CHARS = 8000
VERB = re.compile(r'(?i)\b(find|calculate|compute|determine|how many|how much|'
                  r'what is|what are|solve|prove|show that|evaluate|simplify|'
                  r'derive|estimate)\b')


def iter_jsonl(path):
    with Path(path).open(encoding='utf-8', errors='replace') as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def split_math(text):
    """Problem statement up to the last '?' in the opening, remainder = solution.

    The corpus puts the solution immediately after the problem, often separated
    by a single newline rather than a blank line, so a blank-line split leaks
    the start of the solution into the prompt.
    """
    t = text.strip()
    qpos = t[:600].rfind('?')
    if qpos >= 0:
        nl = t.find('\n', qpos)
        if nl > 0:
            q, ans = t[:nl].strip(), t[nl:].strip()
        else:
            q, ans = t[:qpos + 1].strip(), t[qpos + 1:].strip()
    else:
        # Imperative problems ("Calculate ...", "Prove ..."): cut at the first
        # newline that follows a sentence-ending mark.
        nl = -1
        for i, ch in enumerate(t):
            if ch == '\n' and i >= 40 and t[i - 1] in '.?:)':
                nl = i
                break
        if nl < 0:
            parts = t.split('\n\n', 1)
            if len(parts) < 2:
                return None
            q, ans = parts[0].strip(), parts[1].strip()
        else:
            q, ans = t[:nl].strip(), t[nl:].strip()
    if not 20 <= len(q) <= 1500:
        return None
    if not (q.endswith(('.', '?', ':')) or VERB.search(q)):
        return None
    if not 100 <= len(ans) <= 6000:
        return None
    if len(q) + len(ans) > MAX_CHARS:
        return None
    return q, ans


def load_ultradata(quota, rng, ultra_dir=ULTRA):
    if quota < 0:
        raise ValueError('UltraData quota must be nonnegative')
    if quota == 0:
        return []
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(ultra_dir / '*.parquet')))
    rng.shuffle(files)
    out = []
    for path in files:
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                split = split_math(row.get('content', ''))
                if split is None:
                    continue
                q, ans = split
                out.append({'prompt': q + '\n', 'response': ans,
                            'source': 'math/ultradata_l3_qa',
                            'id': f'ultra_{Path(path).stem}_{len(out)}'})
                if len(out) >= quota:
                    return out
    return out


def load_gsm8k(repeat, gsm_dir=GSM):
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
    p.add_argument('--v2', type=Path, default=V2, help='Prepared sft_v2 directory')
    p.add_argument('--gsm', type=Path, default=GSM, help='Prepared GSM8K directory')
    p.add_argument('--ultra', type=Path, default=ULTRA, help='UltraData-Math QA-Synthetic parquet directory')
    p.add_argument('--benchmark-root', type=Path, default=ROOT, help='Root containing eval_data/')
    p.add_argument('--output', type=Path, default=OUT, help='New output directory')
    p.add_argument('--math-size', type=int, default=75_000)
    p.add_argument('--gsm-repeat', type=int, default=2)
    p.add_argument('--seed', type=int, default=20261110)
    args = p.parse_args()
    if args.math_size < 1 or args.gsm_repeat < 1:
        p.error('--math-size and --gsm-repeat must be positive')
    out = args.output

    if out.exists():
        raise SystemExit(f'{out} already exists; choose a new output directory')
    rng = random.Random(args.seed)

    print('Reusing general + code from sft_v2...', flush=True)
    general, code = [], []
    for r in iter_jsonl(args.v2 / 'train.jsonl'):
        src = r['source']
        if src.startswith('code/'):
            code.append(r)
        elif not src.startswith('gsm8k/'):
            general.append(r)
    print(f'  general: {len(general)}  code: {len(code)}', flush=True)

    print('Loading GSM8K...', flush=True)
    gsm = load_gsm8k(args.gsm_repeat, args.gsm)
    print(f'  gsm8k: {len(gsm)}', flush=True)

    ultra_quota = max(0, args.math_size - len(gsm))
    print(f'Loading UltraData-Math L3 QA (quota {ultra_quota})...', flush=True)
    ultra = load_ultradata(ultra_quota, rng, args.ultra)
    print(f'  ultradata: {len(ultra)}', flush=True)

    print('Decontaminating UltraData math...', flush=True)
    ultra, removed = decontaminate(ultra, args.benchmark_root)
    print(f'  kept {len(ultra)}  removed {removed}', flush=True)

    train = general + code + gsm + ultra
    if not train:
        raise ValueError('No SFT records remained; check input directories and quotas')
    rng.shuffle(train)

    dev = list(iter_jsonl(args.v2 / 'dev.jsonl'))
    rng.shuffle(dev)

    out.mkdir(parents=True, exist_ok=False)
    for name, rows in (('train', train), ('dev', dev)):
        with (out / f'{name}.jsonl').open('w', encoding='utf-8') as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=True) + '\n')

    cat = collections.Counter()
    for r in train:
        s = r['source']
        if s.startswith('code/'):
            cat['code'] += 1
        elif s.startswith('gsm8k/') or s.startswith('math/'):
            cat['math'] += 1
        else:
            cat['general'] += 1
    src = collections.Counter(r['source'] for r in train)
    manifest = {
        'version': 3,
        'seed': args.seed,
        'train_size': len(train),
        'dev_size': len(dev),
        'categories': dict(cat),
        'category_share': {k: v / len(train) for k, v in cat.items()},
        'sources': dict(src),
        'gsm_repeat': args.gsm_repeat,
        'math_decontam_removed': removed,
        'files': {'train': str((out / 'train.jsonl').resolve()),
                  'dev': str((out / 'dev.jsonl').resolve())},
    }
    atomic_json(out / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
