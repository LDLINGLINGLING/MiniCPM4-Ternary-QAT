"""Rebuild TensorBoard event files for runs from their metrics.jsonl.

Usage: python qat/backfill_tb.py runs/smoke_qat runs/s2_*
Skips runs that already have a tb/ directory unless --force is given.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat.tb import TBLogger  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('runs', nargs='+')
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    for run in map(Path, args.runs):
        metrics = run / 'metrics.jsonl'
        if not metrics.exists():
            continue
        if (run / 'tb').exists():
            if not args.force:
                print(f'skip {run} (tb/ exists)')
                continue
            shutil.rmtree(run / 'tb')
        cfg_path = run / 'resolved_config.json'
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
        logger = TBLogger(run, cfg)
        n = 0
        for line in metrics.read_text().splitlines():
            if line.strip():
                logger.log(json.loads(line))
                n += 1
        logger.close()
        print(f'{run}: {n} records')


if __name__ == '__main__':
    main()
