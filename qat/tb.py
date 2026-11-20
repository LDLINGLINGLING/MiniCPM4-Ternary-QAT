"""TensorBoard logging shared by train.py and the backfill tool.

Every metrics.jsonl record is flattened into scalar tags, with the token count
as the x-axis so runs with different batch sizes line up:
  train/loss_ce, train/lr, train/grad_norm, ...
  val_ternary/zh, val_ternary/en, val_ternary/mean   (deployed precision)
  val_float/zh, ...                                   (float reference)
  quant/zero_frac, quant/rel_err
"""
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

SKIP = {'step', 'tokens', 'elapsed_s', 'event', 'eval_only'}
GROUPED = {'val_ternary', 'val_float', 'quant'}


def flatten(rec):
    out = {}
    for key, value in rec.items():
        if key in SKIP or value is None:
            continue
        if key in GROUPED and isinstance(value, dict):
            for sub, v in value.items():
                out[f'{key}/{sub}'] = v
            if key.startswith('val_') and value:
                out[f'{key}/mean'] = sum(value.values()) / len(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[f'train/{key}'] = value
    return out


class TBLogger:
    def __init__(self, run_dir, cfg=None):
        self.writer = SummaryWriter(str(Path(run_dir) / 'tb'))
        if cfg is not None:
            self.writer.add_text('config', '```\n' + json.dumps(cfg, indent=2) + '\n```', 0)

    def log(self, rec):
        if 'tokens' not in rec:
            return
        for tag, value in flatten(rec).items():
            self.writer.add_scalar(tag, value, global_step=int(rec['tokens']))
        if 'val_ternary' in rec:
            self.writer.flush()

    def close(self):
        self.writer.close()
