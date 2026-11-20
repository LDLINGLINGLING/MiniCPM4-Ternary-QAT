"""Summarize QAT runs: training-loss tail, validation CE (ternary and float), throughput.

Usage: python qat/summarize.py runs/s2_*  [--markdown reports/s2.md]
"""
import argparse
import json
import math
from pathlib import Path


def load(run):
    path = Path(run) / 'metrics.jsonl'
    if not path.exists():
        return None
    recs = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    cfg = json.loads((Path(run) / 'config.json').read_text())
    evals = [r for r in recs if 'val_ternary' in r]
    train = [r for r in recs if 'loss_ce' in r]
    done = any(r.get('event') == 'done' for r in recs)
    return dict(name=Path(run).name, cfg=cfg, evals=evals, train=train, done=done)


def row(r):
    last = r['evals'][-1] if r['evals'] else None
    first = r['evals'][0] if r['evals'] else None
    tern = last['val_ternary'] if last else {}
    flt = last['val_float'] if last else {}
    mean_t = sum(tern.values()) / len(tern) if tern else float('nan')
    mean_t0 = sum(first['val_ternary'].values()) / len(first['val_ternary']) if first else float('nan')
    mean_f0 = sum(first['val_float'].values()) / len(first['val_float']) if first else float('nan')
    rec = (mean_t0 - mean_t) / (mean_t0 - mean_f0) if first and mean_t0 > mean_f0 else float('nan')
    tps = r['train'][-1]['tokens_per_s'] if r['train'] else float('nan')
    return dict(run=r['name'], lr=r['cfg'].get('lr'), ramp=r['cfg'].get('alpha_ramp_frac', 0),
                kd=r['cfg'].get('kd_weight', 0), mode=r['cfg'].get('mode', 'qat'),
                tokens_M=(last['tokens'] / 1e6) if last else 0,
                zh_tern=tern.get('zh', float('nan')), en_tern=tern.get('en', float('nan')),
                mean_tern=mean_t, ppl_tern=math.exp(mean_t) if mean_t == mean_t else float('nan'),
                zh_float=flt.get('zh', float('nan')), en_float=flt.get('en', float('nan')),
                recovery=rec, tok_s=tps, done=r['done'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--markdown')
    args = p.parse_args()
    rows = [row(r) for r in (load(x) for x in sorted(args.runs)) if r]
    rows.sort(key=lambda x: (x['mean_tern'] != x['mean_tern'], x['mean_tern']))
    cols = ['run', 'mode', 'lr', 'ramp', 'kd', 'tokens_M', 'zh_tern', 'en_tern', 'mean_tern', 'ppl_tern',
            'zh_float', 'en_float', 'recovery', 'tok_s', 'done']
    fmt = {'tokens_M': '{:.0f}', 'zh_tern': '{:.3f}', 'en_tern': '{:.3f}', 'mean_tern': '{:.3f}',
           'ppl_tern': '{:.1f}', 'zh_float': '{:.3f}', 'en_float': '{:.3f}', 'recovery': '{:.3f}',
           'tok_s': '{:.0f}'}
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        lines.append('| ' + ' | '.join(fmt.get(c, '{}').format(r[c]) for c in cols) + ' |')
    table = '\n'.join(lines)
    print(table)
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(table + '\n')


if __name__ == '__main__':
    main()
