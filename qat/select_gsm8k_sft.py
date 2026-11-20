"""Select checkpoint using ONLY development scores, emit auditable selection."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    candidates = []
    for path in map(Path, args.runs):
        records = [json.loads(l) for l in (path / 'metrics.jsonl').read_text().splitlines()]
        for epoch in range(1, 6):
            result = json.loads((path / f'dev_epoch_{epoch}.json').read_text())
            score = result['tasks']['gsm8k']
            if score['n'] != 500:
                raise ValueError('Expected exactly 500 development questions')
            cfg = json.loads((path / 'resolved_config.json').read_text())
            # CE is read from epoch record, not from test results.
            matching = [r for r in records if r.get('event') == 'epoch_complete' and r.get('epoch') == epoch]
            if not matching:
                raise ValueError(f'Missing epoch CE: {path}, epoch {epoch}')
            rec = matching[-1]
            ce = (rec.get('val_ternary') or rec.get('val_float') or {}).get('gsm8k_answer_ce', float('inf'))
            candidates.append({'run': str(path), 'epoch': epoch, 'lr': cfg['lr'],
                               'strict_acc': score['strict_acc'], 'acc': score['acc'], 'dev_ce': ce,
                               'checkpoint': str((path / 'snapshots' / f'epoch_{epoch}' / 'model.pt').resolve()),
                               'dev_sha256': result['gsm_test_sha256']})
    if len({r['dev_sha256'] for r in candidates}) != 1:
        raise ValueError('Different development datasets')
    candidates.sort(key=lambda r: (-r['strict_acc'], r['dev_ce'], r['epoch'], r['lr']))
    out = {'selection_rule': 'max dev strict_acc, min dev CE, earliest epoch, smaller LR',
           'selected': candidates[0], 'candidates': candidates}
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(candidates[0]['checkpoint'])


if __name__ == '__main__':
    main()
