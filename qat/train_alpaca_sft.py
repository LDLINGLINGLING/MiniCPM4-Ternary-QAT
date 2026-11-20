"""General-instruction SFT continuing from an existing QAT or BF16 SFT checkpoint.

Supports the unified SFT format produced by prepare_alpaca_sft.py:
  {"prompt": "...", "response": "...", "source": "...", "id": "..."}

Prompt is prefixed with BOS and NOT supervised (label -100).
Response + EOS is supervised.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
      qat/train_alpaca_sft.py \
      --data-dir data/alpaca_sft_v1 \
      --run-dir runs/mix_v2_qat_alpaca_sft \
      --init-ckpt runs/mix_v2_qat_sft/snapshots/epoch_4/model.pt \
      --init-mode qat \
      --lr 5e-5 --epochs 3
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json, sha256_file
from qat.quant import convert_to_ternary, set_alpha, set_quant_group, ternary_modules

ROOT = Path(__file__).resolve().parents[1]


def quant_group_of(ckpt):
    """Granularity the run that produced ckpt trained with; None means legacy per-row."""
    if not ckpt:
        return None
    path = Path(ckpt).resolve()
    for parent in path.parents:
        for name in ('resolved_config.json', 'config.json'):
            candidate = parent / name
            if candidate.is_file():
                return json.loads(candidate.read_text()).get('quant_group')
        if parent.name == 'runs':
            break
    return None


def atomic_torch_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as fh:
            torch.save(value, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def encode_record(tok, record, max_length):
    """BOS + prompt tokens (no supervision) + response tokens + EOS (supervised)."""
    prompt_ids = tok.encode(record['prompt'], add_special_tokens=True, truncation=False)
    # Remove accidental EOS appended by tokenizer to prompt
    if prompt_ids and prompt_ids[-1] == tok.eos_token_id:
        prompt_ids = prompt_ids[:-1]
    resp_ids = tok.encode(' ' + record['response'], add_special_tokens=False, truncation=False)
    if not resp_ids or resp_ids[-1] != tok.eos_token_id:
        resp_ids.append(tok.eos_token_id)
    ids = prompt_ids + resp_ids
    if len(ids) > max_length:
        return None  # skip over-length instead of truncating
    labels = [-100] * len(prompt_ids) + resp_ids
    return {'input_ids': ids, 'labels': labels,
            'response_tokens': len(resp_ids), 'id': record.get('id', '')}


def collate(examples, pad_id):
    if not examples:
        raise ValueError('Empty microbatch')
    rows = [e for e in examples if e is not None]
    if not rows:
        # all-dummy microbatch
        rows = [{'input_ids': [pad_id, pad_id], 'labels': [-100, -100]}]
    length = max(len(r['input_ids']) for r in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    lbls = torch.full((len(rows), length), -100, dtype=torch.long)
    for i, r in enumerate(rows):
        n = len(r['input_ids'])
        ids[i, :n] = torch.tensor(r['input_ids'])
        lbls[i, :n] = torch.tensor(r['labels'])
    return ids, lbls


def warmup_cosine(step, total, warmup_frac=0.05, min_ratio=0.1):
    warm = max(1, math.ceil(total * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    t = (step - warm) / max(1, total - warm)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def assert_fp32_master(model):
    for name, t in model.named_parameters():
        if t.is_floating_point() and t.dtype != torch.float32:
            raise ValueError(f'Expected FP32 master weight: {name} {t.dtype}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--run-dir', required=True)
    p.add_argument('--model', default=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'))
    p.add_argument('--init-ckpt', required=True)
    p.add_argument('--init-mode', choices=['qat', 'bf16'], default='qat')
    p.add_argument('--resume')
    p.add_argument('--lr', type=float, default=5e-5)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--global-batch', type=int, default=64)
    p.add_argument('--micro-batch', type=int, default=2)
    p.add_argument('--max-length', type=int, default=2048)
    p.add_argument('--warmup-frac', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=0.1)
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--save-every-steps', type=int, default=200)
    p.add_argument('--log-every-steps', type=int, default=10)
    p.add_argument('--no-gradient-checkpointing', action='store_true')
    p.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.epochs < 1 or args.global_batch < 1 or args.micro_batch < 1:
        p.error('epochs/global_batch/micro_batch must be positive')
    return vars(args)


def main():
    cfg = parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local = int(os.environ.get('LOCAL_RANK', '0'))
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    random.seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    torch.manual_seed(cfg['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True

    run_dir = Path(cfg['run_dir'])
    if rank == 0:
        if cfg['resume']:
            if not run_dir.is_dir():
                raise ValueError('--resume requires existing run directory')
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()

    # Load data
    data_dir = Path(cfg['data_dir'])
    manifest = json.loads((data_dir / 'manifest.json').read_text())
    train_records = [json.loads(l) for l in (data_dir / 'train.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
    dev_records = [json.loads(l) for l in (data_dir / 'dev.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg['model'], trust_remote_code=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if rank == 0:
        print(f'Encoding {len(train_records)} train / {len(dev_records)} dev records...', flush=True)

    train_examples = [e for r in train_records for e in [encode_record(tok, r, cfg['max_length'])] if e]
    dev_examples = [e for r in dev_records for e in [encode_record(tok, r, cfg['max_length'])] if e]

    if rank == 0:
        skip = len(train_records) - len(train_examples)
        print(f'  train: {len(train_examples)} (skipped {skip} over-length)', flush=True)
        print(f'  dev:   {len(dev_examples)}', flush=True)

    if not train_examples:
        raise ValueError('No training examples after encoding')

    data_hash = hashlib.sha256(
        json.dumps([e['input_ids'] for e in train_examples], sort_keys=True).encode()
    ).hexdigest()

    # Model
    from qat.train import load_model, param_groups
    raw = load_model(cfg['model'], device, torch.float32)
    init_mode = cfg['init_mode']
    # Inherit the granularity the initial checkpoint was trained with, so SFT
    # continues the same quantizer instead of silently re-quantizing per group.
    init_group = quant_group_of(cfg['init_ckpt'])
    cfg['quant_group'] = init_group
    if init_mode == 'qat':
        set_quant_group(init_group)
        convert_to_ternary(raw)
    assert_fp32_master(raw)

    cfg_mup = {'mup_lr_scale': True, 'weight_decay': cfg['weight_decay']}
    groups = param_groups(raw, cfg_mup)
    optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=(0.9, 0.95), eps=1e-8)

    start_step = 0
    start_epoch = 0
    if cfg['resume']:
        saved = torch.load(cfg['resume'], map_location='cpu', weights_only=False)
        raw.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        start_step = saved['step']
        start_epoch = saved.get('epoch', 0)
        del saved
    else:
        init = torch.load(cfg['init_ckpt'], map_location='cpu', weights_only=False)
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in init['model'].values()):
            raise ValueError('Init checkpoint is not FP32 master state')
        raw.load_state_dict(init['model'], strict=True)
        del init

    if init_mode == 'qat':
        set_alpha(raw, 1.0)

    if not cfg['no_gradient_checkpointing']:
        raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    raw.config.use_cache = False
    model = DDP(raw, device_ids=[local], broadcast_buffers=False) if world > 1 else raw

    updates_per_epoch = math.ceil(len(train_examples) / cfg['global_batch'])
    total_steps = updates_per_epoch * cfg['epochs']

    if rank == 0:
        cfg_save = {**cfg, 'data_hash': data_hash, 'manifest': manifest,
                    'train_examples': len(train_examples), 'dev_examples': len(dev_examples),
                    'updates_per_epoch': updates_per_epoch, 'total_steps': total_steps}
        if not cfg['resume']:
            atomic_json(run_dir / 'resolved_config.json', cfg_save)
        print(json.dumps({'event': 'start', 'world': world, 'total_steps': total_steps,
                          'train': len(train_examples), 'dev': len(dev_examples),
                          'init_mode': init_mode, 'lr': cfg['lr']}), flush=True)

    log_path = run_dir / 'metrics.jsonl'
    tb = None
    if rank == 0 and cfg['tensorboard']:
        try:
            from qat.tb import TBLogger
            tb = TBLogger(run_dir, cfg)
        except Exception:
            pass

    def log(rec):
        if rank != 0:
            return
        line = json.dumps(rec, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with log_path.open('a', encoding='utf-8') as fh:
            fh.write(line + '\n')
        if tb:
            tb.log(rec)

    def save_state(tag='latest'):
        state = {'model': raw.state_dict(), 'optimizer': optimizer.state_dict(),
                 'step': step, 'epoch': epoch,
                 'config': {k: v for k, v in cfg.items() if k not in ('resume',)},
                 'data_hash': data_hash, 'alpha': 1.0 if init_mode == 'qat' else None}
        path = run_dir / 'checkpoints' / tag / 'state.pt'
        if rank == 0:
            atomic_torch_save(state, path)
        if world > 1:
            dist.barrier()

    def save_snapshot(ep):
        state = {'model': raw.state_dict(), 'config': cfg, 'epoch': ep,
                 'alpha': 1.0 if init_mode == 'qat' else None,
                 'precision': 'fp32_master_ternary_qat' if init_mode == 'qat' else 'fp32_master_bf16_sft'}
        path = run_dir / 'snapshots' / f'epoch_{ep}' / 'model.pt'
        if rank == 0:
            atomic_torch_save(state, path)
        if world > 1:
            dist.barrier()

    @torch.no_grad()
    def evaluate_dev():
        raw.eval()
        total_ce = torch.zeros(1, device=device)
        total_tokens = torch.zeros(1, device=device)
        for i in range(0, len(dev_examples), cfg['micro_batch'] * world):
            batch_examples = dev_examples[i + rank * cfg['micro_batch']:
                                          i + rank * cfg['micro_batch'] + cfg['micro_batch']]
            if not batch_examples:
                break
            ids, lbls = collate(batch_examples, pad_id)
            ids, lbls = ids.to(device), lbls.to(device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = raw(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            active = lbls[:, 1:].ne(-100)
            if active.sum() > 0:
                ce_sum = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                         lbls[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
                total_ce += ce_sum
                total_tokens += active.sum()
        if world > 1:
            dist.all_reduce(total_ce)
            dist.all_reduce(total_tokens)
        raw.train()
        return (total_ce / total_tokens.clamp(min=1)).item()

    step = start_step
    started = time.monotonic()
    for epoch in range(start_epoch, cfg['epochs']):
        indices = list(range(len(train_examples)))
        random.Random(cfg['seed'] + epoch).shuffle(indices)
        rounds = math.ceil(len(indices) / (world * cfg['micro_batch']))
        padded = indices + [None] * (rounds * world * cfg['micro_batch'] - len(indices))

        for update in range(updates_per_epoch):
            lr_scale = warmup_cosine(step, total_steps, cfg['warmup_frac'])
            for g in optimizer.param_groups:
                g['lr'] = cfg['lr'] * lr_scale * g.get('lr_mult', 1.0)

            accum = rounds  # microbatches per update
            # re-derive from global batch
            accum = math.ceil(cfg['global_batch'] / (cfg['micro_batch'] * world))
            ce_sum = 0.0
            resp_tokens = 0
            optimizer.zero_grad(set_to_none=True)
            for a in range(accum):
                slot = (update * accum + a) * world + rank
                if slot >= len(padded):
                    break
                idx = padded[slot]
                ex = [train_examples[idx]] if idx is not None else [None]
                ids, lbls = collate(ex, pad_id)
                ids, lbls = ids.to(device), lbls.to(device)
                sync = (a == accum - 1)
                ctx = model.no_sync() if (not sync and world > 1) else nullcontext()
                with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
                active = lbls[:, 1:].ne(-100)
                n = int(active.sum().item())
                if n > 0:
                    ce = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                                         lbls[:, 1:].reshape(-1), ignore_index=-100, reduction='sum')
                    (ce / max(n, 1)).backward()
                    ce_sum += ce.item()
                    resp_tokens += n

            if resp_tokens > 0:
                stats = torch.tensor([ce_sum, float(resp_tokens)], device=device)
                if world > 1:
                    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
                mean_ce = (stats[0] / stats[1].clamp(min=1)).item()
            else:
                mean_ce = float('nan')

            norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), cfg['grad_clip'])
            if not torch.isfinite(norm):
                raise RuntimeError(f'Non-finite gradient norm at step {step}')
            optimizer.step()
            step += 1

            if step % cfg['log_every_steps'] == 0:
                log({'step': step, 'epoch': epoch + 1, 'event': 'train',
                     'loss_ce': mean_ce, 'lr': cfg['lr'] * lr_scale,
                     'grad_norm': norm.item(), 'elapsed_s': time.monotonic() - started,
                     'alpha': 1.0 if init_mode == 'qat' else None})

            if cfg['save_every_steps'] and step % cfg['save_every_steps'] == 0:
                save_state('latest')

        # end of epoch
        dev_ce = evaluate_dev()
        save_snapshot(epoch + 1)
        log({'event': 'epoch_complete', 'epoch': epoch + 1, 'dev_ce': dev_ce,
             'step': step, 'alpha': 1.0 if init_mode == 'qat' else None})

    save_state('final')
    log({'event': 'done', 'step': step, 'epochs': cfg['epochs']})
    if tb:
        tb.close()
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
