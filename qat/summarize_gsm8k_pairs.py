"""Paired original/variant GSM8K diagnostics; no contamination verdict implied."""
import argparse
import json
from pathlib import Path
import numpy as np


def load(path):
    d = json.loads(Path(path).read_text())
    samples = d['tasks']['gsm8k']['samples']
    pairs = {str(s.get('pair_id', s['i'])): s for s in samples}
    if len(pairs) != len(samples):
        raise ValueError('Duplicate pair IDs')
    return pairs


def compare(a, b):
    if not b.keys() <= a.keys():
        raise ValueError('Variant IDs missing from originals')
    keys = sorted(b)
    if not keys:
        raise ValueError('No valid pairs')
    result = {'n': len(keys), 'excluded_from_originals': len(a) - len(keys)}
    rng = np.random.default_rng(1234)
    for metric in ['ok', 'strict_ok']:
        def value(s):
            return bool(s['ok']) if metric == 'ok' else bool(s.get('strict_ok', s['ok'] and '####' in s['text']))
        x = np.array([value(a[k]) for k in keys], dtype=float)
        y = np.array([value(b[k]) for k in keys], dtype=float)
        diffs = y - x
        boot = np.mean(diffs[rng.integers(len(keys), size=(10000, len(keys)))], axis=1)
        result[metric] = {'original_acc': float(x.mean()), 'variant_acc': float(y.mean()),
                          'delta': float(diffs.mean()), 'delta_ci95': np.quantile(boot, [.025, .975]).tolist(),
                          'right_to_wrong': int(((x == 1) & (y == 0)).sum()),
                          'wrong_to_right': int(((x == 0) & (y == 1)).sum()),
                          'original_correct_n': int(x.sum())}
    changed = [k for k in keys if a[k]['gold'] != b[k]['gold']]
    old_answer = [k for k in changed if b[k]['pred'] is not None and abs(b[k]['pred'] - a[k]['gold']) < 1e-6]
    result['changed_answer_n'] = len(changed)
    result['old_answer_predictions'] = len(old_answer)
    result['old_answer_rate'] = len(old_answer) / len(changed) if changed else None
    result['pairs'] = [{'pair_id': k, 'original_ok': a[k]['ok'], 'variant_ok': b[k]['ok'],
                        'original_gold': a[k]['gold'], 'variant_gold': b[k]['gold'], 'variant_pred': b[k]['pred']} for k in keys]
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--original', required=True)
    p.add_argument('--variant', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    r = compare(load(args.original), load(args.variant))
    Path(args.out).write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in r.items() if k != 'pairs'}))


if __name__ == '__main__':
    main()
