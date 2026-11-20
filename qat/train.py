"""Ternary-weight QAT (and BF16 continued-training control) for MiniCPM4-0.5B.

Launch: torchrun --nproc_per_node=N qat/train.py --config runs/<name>/config.json
Modes:
  qat   linear layers inside decoder blocks use ternary fake quant (STE)
  bf16  same training without quantization (control group A in QAT_PLAN.md)
Validation always reports both the float forward and the alpha=1 ternary
forward, so every run is comparable at the deployed precision.
"""
import argparse
import json
import math
import os
import random
from datetime import timedelta
from pathlib import Path

import numpy as np
import sys
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sampling import MixedSampler, TokenStream, load_manifest, val_batches, validation_path  # noqa: E402
from qat.checkpoint_utils import (atomic_torch_save, capture_rng, collective_check, data_identity,
                                  finite_guard, model_identity, rank_zero_call, restore_rng,
                                  save_training_state, validate_resume)  # noqa: E402
from qat import quant
from qat.quant import convert_to_ternary, quant_stats, set_alpha, set_quant_group, set_quantize  # noqa: E402
from qat.tb import TBLogger  # noqa: E402

DEFAULTS = dict(
    model='MiniCPM4-0.5B-fjzcuwlf', data='data/ufw_v1', langs=['zh', 'en'], weights=[0.5, 0.5],
    mode='qat', seq_len=2048, micro_batch=8, global_batch_tokens=1_048_576, total_tokens=500_000_000,
    lr=1e-3, mup_lr_scale=True, min_lr_ratio=0.1, warmup_frac=0.02, decay_frac=0.2,
    weight_decay=0.1, beta1=0.9, beta2=0.95, grad_clip=1.0,
    alpha_ramp_frac=0.0, alpha_ramp_tokens=0, kd_weight=0.0, kd_temp=1.0, teacher_ckpt=None,
    quant_group=32,
    eval_every_tokens=50_000_000, val_windows=256, val_data=None, seed=1234, tensorboard=True,
    save_every_tokens=0, snapshot_tokens=[], chunked_ce=False, ce_chunk_tokens=8192,
    strict_resume=False, data_manifest=None, eval_langs=None, max_data_passes=None,
)


def load_config():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--run-dir', help='Output run directory; defaults to the config file directory')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--resume', help='Explicit state.pt path; mandatory for resuming strict runs')
    p.add_argument('--max-steps', type=int, help='Pause at this absolute update without changing schedules')
    args = p.parse_args()
    cfg = dict(DEFAULTS)
    raw = json.loads(Path(args.config).read_text())
    # Configs written before the BitCPM4 port quantized per row; an absent key must
    # stay absent-semantics rather than silently inheriting the new per-group default.
    if 'quant_group' not in raw:
        raw['quant_group'] = None
    cfg.update(raw)
    cfg['run_dir'] = str(Path(args.run_dir).resolve()) if args.run_dir else str(Path(args.config).resolve().parent)
    cfg['eval_only'] = args.eval_only
    cfg['resume'] = str(Path(args.resume).resolve()) if args.resume else None
    cfg['max_steps'] = args.max_steps
    if args.max_steps is not None and args.max_steps < 0:
        p.error('--max-steps must be nonnegative')
    if cfg['eval_langs'] is None:
        cfg['eval_langs'] = list(cfg['langs'])
    if not cfg['eval_langs'] or len(set(cfg['eval_langs'])) != len(cfg['eval_langs']):
        p.error('eval_langs must contain unique, nonempty validation streams')
    if cfg['val_data'] is None:
        cfg['val_data'] = cfg['data'] if isinstance(cfg['data'], str) else cfg['data'][0]
    if cfg['data_manifest'] is not None:
        cfg['data_manifest'] = str(Path(cfg['data_manifest']).expanduser().resolve())
    return cfg


def load_model(path, device, dtype):
    # configuration_minicpm.py forces flash_attention_2 whenever flash_attn is importable.
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype, trust_remote_code=True)
    model.config.use_cache = False
    return model.to(device)


def lm_logits(model, input_ids):
    """Full-vocab logits in fp32 (the remote forward applies scale_emb/scale_depth/dim_model_base).

    Windows are packed with no padding; the remote flash-attention path insists on a mask,
    so an all-ones mask is passed."""
    mask = torch.ones_like(input_ids)
    return model(input_ids=input_ids, attention_mask=mask, use_cache=False).logits


class FinalHidden(torch.nn.Module):
    """DDP-wrappable forward that stops before lm_head and applies the remote logit scaling."""

    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, input_ids):
        out = self.lm.model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
        return out[0] / (self.lm.config.hidden_size / self.lm.config.dim_model_base)


def _chunk_ce_sum(hidden, weight, labels):
    logits = F.linear(hidden, weight).float()
    return F.cross_entropy(logits, labels, reduction='sum')


def chunked_ce(hidden, weight, labels, chunk_tokens):
    """Mean CE equal to the full-logit path, but each chunk's fp32 logits are recomputed
    in backward instead of all being kept alive (micro_batch x seq x 73k vocab)."""
    h = hidden.reshape(-1, hidden.size(-1))
    y = labels.reshape(-1)
    total = 0.0
    for i in range(0, y.numel(), chunk_tokens):
        total = total + checkpoint(_chunk_ce_sum, h[i:i + chunk_tokens], weight, y[i:i + chunk_tokens],
                                   use_reentrant=False)
    return total / y.numel()


def param_groups(model, cfg):
    """muP: hidden matrices get lr * dim_model_base / hidden_size; embeddings/norms get base lr."""
    conf = model.config
    scale = conf.dim_model_base / conf.hidden_size if cfg['mup_lr_scale'] else 1.0
    matrices, embed, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'embed_tokens' in name or 'lm_head' in name:
            embed.append(p)
        elif p.ndim == 2:
            matrices.append(p)
        else:
            other.append(p)
    return [
        dict(params=matrices, lr_mult=scale, weight_decay=cfg['weight_decay'], name='matrices'),
        dict(params=embed, lr_mult=1.0, weight_decay=0.0, name='embed'),
        dict(params=other, lr_mult=1.0, weight_decay=0.0, name='norms'),
    ]


def lr_at(step, total, cfg):
    """WSD: linear warmup, constant, linear decay to min_lr_ratio over the final decay_frac."""
    warm = max(1, int(total * cfg['warmup_frac']))
    decay_start = int(total * (1 - cfg['decay_frac']))
    if step < warm:
        return (step + 1) / warm
    if step < decay_start:
        return 1.0
    frac = (step - decay_start) / max(1, total - decay_start)
    return 1.0 - frac * (1.0 - cfg['min_lr_ratio'])


def alpha_at(step, total, cfg):
    """alpha_ramp_tokens (absolute) takes precedence over alpha_ramp_frac (fraction of the run)."""
    if cfg['alpha_ramp_tokens']:
        ramp = cfg['alpha_ramp_tokens'] // cfg['global_batch_tokens']
    else:
        ramp = int(total * cfg['alpha_ramp_frac'])
    return 1.0 if ramp <= 0 else min(1.0, step / ramp)


@torch.no_grad()
def evaluate(model, cfg, device, world, rank):
    """Mean next-token CE per language on fixed validation windows, sharded over ranks."""
    model.eval()
    out = {}
    for lang in cfg.get('eval_langs') or cfg['langs']:
        tot = torch.zeros(2, device=device, dtype=torch.float64)
        for i, batch in enumerate(val_batches(cfg['val_data'], lang, cfg['seq_len'], cfg['val_windows'],
                                              cfg['micro_batch'], manifest=cfg.get('_manifest'))):
            if i % world != rank:
                continue
            batch = batch.to(device, non_blocking=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                logits = lm_logits(model, batch[:, :-1])
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), batch[:, 1:].reshape(-1),
                                   reduction='sum')
            tot += torch.tensor([loss.item(), batch[:, 1:].numel()], device=device, dtype=torch.float64)
        if world > 1:
            dist.all_reduce(tot)
        collective_check(torch.isfinite(tot).all().item() and tot[1].item() > 0,
                         f'Invalid validation loss/count for {lang}', device)
        out[lang] = (tot[0] / tot[1]).item()
    model.train()
    return out


def eval_both(raw_model, cfg, device, world, rank):
    """Validation is observational: it never advances the training RNG stream."""
    rng = capture_rng(device)
    try:
        return _eval_both(raw_model, cfg, device, world, rank)
    finally:
        restore_rng(rng, device)


def _eval_both(raw_model, cfg, device, world, rank):
    """Validation at deployed precision (ternary, alpha=1) and, for reference, in float."""
    res = {}
    if cfg['mode'] == 'qat':
        alpha = raw_model._alpha
        set_alpha(raw_model, 1.0)
        res['val_ternary'] = evaluate(raw_model, cfg, device, world, rank)
        set_alpha(raw_model, alpha)
        set_quantize(raw_model, False)
        res['val_float'] = evaluate(raw_model, cfg, device, world, rank)
        set_quantize(raw_model, True)
    else:
        res['val_float'] = evaluate(raw_model, cfg, device, world, rank)
        set_quantize(raw_model, True)  # bf16 control: what direct ternarization would give now
        res['val_ternary'] = evaluate(raw_model, cfg, device, world, rank)
        set_quantize(raw_model, False)
    return res


def kd_loss(student_logits, teacher_logits, temp):
    """Token-mean KL(teacher || student) at temperature temp, scaled by temp^2."""
    s = F.log_softmax(student_logits.float() / temp, dim=-1)
    t = F.log_softmax(teacher_logits.float() / temp, dim=-1)
    return (t.exp() * (t - s)).sum(-1).mean() * temp * temp


def save_checkpoint(raw_model, optimizer, step, tokens, cfg, tag):
    """Full training state (fp32 master weights + AdamW). Written to a temp file, then renamed."""
    path = Path(cfg['run_dir']) / 'checkpoints' / tag
    path.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({'model': raw_model.state_dict(), 'optimizer': optimizer.state_dict(),
                       'step': step, 'tokens': tokens, 'config': cfg}, path / 'state.pt')
    return path


def save_snapshot(raw_model, step, tokens, cfg):
    """Model-only fp32 weights at a token milestone, kept permanently for the recovery curve."""
    path = Path(cfg['run_dir']) / 'snapshots' / f'tokens_{tokens // 1_000_000}M'
    path.mkdir(parents=True, exist_ok=True)
    atomic_torch_save({'model': raw_model.state_dict(), 'step': step, 'tokens': tokens,
                       'config': cfg}, path / 'model.pt')
    return path


def main():
    cfg = load_config()
    dist.init_process_group('nccl', timeout=timedelta(minutes=30))
    rank, world = dist.get_rank(), dist.get_world_size()
    local = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local)
    device = torch.device('cuda', local)
    # Include Python/NumPy: trusted remote model code may use either during loading.
    random.seed(cfg['seed'])
    np.random.seed(cfg['seed'])
    torch.manual_seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32 = True

    strict = cfg['strict_resume']
    def check_output():
        root = Path(cfg['run_dir']) / 'checkpoints'
        if strict and not cfg['resume'] and any(root.glob('*/state.pt')):
            raise ValueError('Existing strict run requires explicit --resume PATH (or a new run directory)')
    rank_zero_call(check_output)
    if cfg['data_manifest']:
        cfg['_manifest'] = rank_zero_call(lambda: load_manifest(cfg['data_manifest']))
    else:
        cfg['_manifest'] = None
    seqs_per_step = cfg['global_batch_tokens'] // cfg['seq_len']
    accum = seqs_per_step // (cfg['micro_batch'] * world)
    if (accum < 1 or accum * cfg['micro_batch'] * world * cfg['seq_len'] != cfg['global_batch_tokens']):
        raise ValueError('global_batch_tokens must be divisible by seq_len * micro_batch * world')
    total_steps = cfg['total_tokens'] // cfg['global_batch_tokens']
    if total_steps < 1:
        raise ValueError('total_tokens must allow at least one optimizer update')
    sampler = MixedSampler(cfg['data'], cfg['langs'], cfg['weights'], cfg['seq_len'], cfg['seed'],
                           manifest=cfg['_manifest'], canonical_order=strict)
    epochs = sampler.plan(total_steps * seqs_per_step, max_data_passes=cfg['max_data_passes'])
    def check_validation():
        if cfg['val_windows'] <= 0:
            raise ValueError('val_windows must be positive')
        for lang in cfg['eval_langs']:
            stream = TokenStream([validation_path(cfg['val_data'], lang, cfg['_manifest'])], cfg['seq_len'])
            if strict and stream.num_windows < cfg['val_windows']:
                raise ValueError(f'Insufficient fixed validation windows for {lang}: '
                                 f'{stream.num_windows} < {cfg["val_windows"]}')
    rank_zero_call(check_validation)
    def identify():
        identity = {'data': data_identity(sampler, cfg), 'model': model_identity(cfg['model'])}
        if cfg['teacher_ckpt']:
            from data_pipeline.sampling import sha256_file
            identity['teacher_ckpt'] = sha256_file(cfg['teacher_ckpt'])
        return identity
    identity = rank_zero_call(identify) if strict else None

    model = load_model(cfg['model'], device, torch.float32)
    model._alpha = 1.0
    # Must be set before TernaryLinear is instantiated: granularity is module-global
    # so that train, validation and eval_downstream all quantize identically.
    set_quant_group(cfg['quant_group'])
    # Both modes use TernaryLinear so validation can switch precision; bf16 trains with it off.
    quant_suffixes = cfg.get('quant_suffixes') or list(quant.TARGET_SUFFIXES)
    replaced = convert_to_ternary(model, suffixes=quant_suffixes)
    assert len(replaced) == len(quant_suffixes) * model.config.num_hidden_layers, len(replaced)
    if cfg['mode'] == 'bf16':
        set_quantize(model, False)
    teacher = None
    if cfg['kd_weight'] > 0:
        teacher = load_model(cfg['model'], device, torch.bfloat16).eval()
        if cfg['teacher_ckpt']:
            # Float weights from a finished bf16-control run (saved with TernaryLinear names).
            convert_to_ternary(teacher)
            set_quantize(teacher, False)
            state = torch.load(cfg['teacher_ckpt'], map_location='cpu', weights_only=False)
            teacher.load_state_dict(state['model'])
            del state
        teacher.requires_grad_(False)

    eval_every = max(1, cfg['eval_every_tokens'] // cfg['global_batch_tokens'])
    save_every = cfg['save_every_tokens'] // cfg['global_batch_tokens'] if cfg['save_every_tokens'] else 0

    log_path = Path(cfg['run_dir']) / 'metrics.jsonl'
    tb = TBLogger(cfg['run_dir'], cfg) if rank == 0 and cfg['tensorboard'] else None
    def log(rec):
        if rank == 0:
            print(json.dumps(rec), flush=True)
            with log_path.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            if tb is not None:
                tb.log(rec)

    rank_zero_call(lambda: (Path(cfg['run_dir']) / 'resolved_config.json').write_text(
        json.dumps({k: v for k, v in cfg.items() if k != '_manifest'}, indent=2)))

    groups = param_groups(model, cfg)
    optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']),
                                  eps=1e-8, fused=True)

    start_step = 0
    resume_rng = None
    latest = Path(cfg['run_dir']) / 'checkpoints' / 'latest' / 'state.pt'
    final = Path(cfg['run_dir']) / 'checkpoints' / 'final' / 'state.pt'
    # Legacy configs retain automatic recovery, preferring completed final over stale latest.
    resume_path = cfg['resume'] or (str(final if final.exists() else latest)
                                   if not strict and (final.exists() or latest.exists()) else None)
    if resume_path:
        error = None
        try:
            state = torch.load(resume_path, map_location='cpu', weights_only=False)
            if strict:
                validate_resume(state, cfg, identity, world, total_steps, sampler)
            model.load_state_dict(state['model'])
            optimizer.load_state_dict(state['optimizer'])
            start_step = state['step']
            if not 0 <= start_step <= total_steps:
                raise ValueError('Resume step outside schedule')
            if strict and final.exists() and start_step < total_steps and not cfg['eval_only']:
                raise ValueError('Cannot resume an older partial state into a completed run; use a new run directory')
            model._alpha = state.get('alpha', alpha_at(max(0, start_step - 1), total_steps, cfg)
                                     if cfg['mode'] == 'qat' else 1.0)
            set_alpha(model, model._alpha)
            if strict:
                resume_rng = state['rng_by_rank'][rank]
            del state
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
        collective_check(error is None, f'Resume failed on at least one rank: {error}', device)
    if strict and resume_path and not cfg['eval_only']:
        def check_completion():
            if final.exists():
                completed_state = torch.load(final, map_location='cpu', weights_only=False)
                if completed_state['step'] > start_step:
                    raise ValueError('Refusing to rewind a completed run; resume into a new run directory')
        rank_zero_call(check_completion)
    stop_step = total_steps if cfg['max_steps'] is None else min(total_steps, cfg['max_steps'])
    if stop_step < start_step and not cfg['eval_only']:
        raise ValueError('--max-steps is before the restored checkpoint')
    snapshot_steps = {t // cfg['global_batch_tokens'] for t in cfg['snapshot_tokens']}

    if cfg['eval_only']:
        if resume_rng is not None:
            restore_rng(resume_rng, device)
        res = eval_both(model, cfg, device, world, rank)
        if cfg['mode'] == 'qat':
            res['quant'] = quant_stats(model)
        log({'step': start_step, 'tokens': start_step * cfg['global_batch_tokens'],
             'eval_only': True, **res})
        if tb is not None:
            tb.close()
        dist.destroy_process_group()
        return

    if cfg['chunked_ce'] and teacher is not None:
        raise ValueError('chunked_ce does not support KD; KD needs full student logits')
    # With chunked CE, DDP wraps the trunk only; the tied lm_head weight still receives its
    # gradient in the same backward, so DDP's per-parameter hooks reduce it as usual.
    ddp = DDP(FinalHidden(model) if cfg['chunked_ce'] else model, device_ids=[local],
              broadcast_buffers=False)
    if resume_rng is not None:
        restore_rng(resume_rng, device)
    log({'event': 'resume' if resume_path else 'start', 'step': start_step, 'world': world,
         'accum': accum, 'total_steps': total_steps, 'seqs_per_step': seqs_per_step,
         'data': sampler.describe(), 'data_passes': epochs,
         'schedule_usage': sampler.usage(total_steps * seqs_per_step),
         'actual_usage': sampler.usage(start_step * seqs_per_step),
         'param_groups': {g['name']: sum(p.numel() for p in g['params']) for g in groups}})

    if start_step == 0:
        res = eval_both(model, cfg, device, world, rank)
        log({'step': 0, 'tokens': 0, **res,
             **({'quant': quant_stats(model)} if cfg['mode'] == 'qat' else {})})

    t0 = time.time()
    eval_time = 0.0
    for step in range(start_step, stop_step):
        lr_scale = lr_at(step, total_steps, cfg)
        for g in optimizer.param_groups:
            g['lr'] = cfg['lr'] * lr_scale * g['lr_mult']
        if cfg['mode'] == 'qat':
            model._alpha = alpha_at(step, total_steps, cfg)
            set_alpha(model, model._alpha)

        ce_sum = kd_sum = 0.0
        for a in range(accum):
            start = step * seqs_per_step + (a * world + rank) * cfg['micro_batch']
            batch = sampler.batch(start, cfg['micro_batch']).to(device, non_blocking=True)
            inputs, labels = batch[:, :-1], batch[:, 1:]
            sync = a == accum - 1
            ctx = ddp.no_sync() if not sync else torch.enable_grad()
            with ctx, torch.autocast('cuda', dtype=torch.bfloat16):
                if cfg['chunked_ce']:
                    ce = chunked_ce(ddp(inputs), model.lm_head.weight, labels, cfg['ce_chunk_tokens'])
                else:
                    logits = lm_logits(ddp, inputs)
                    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                loss = ce
                if teacher is not None:
                    with torch.no_grad():
                        t_logits = lm_logits(teacher, inputs)
                    kd = kd_loss(logits, t_logits, cfg['kd_temp'])
                    loss = (1 - cfg['kd_weight']) * ce + cfg['kd_weight'] * kd
                    kd_sum += kd.item() / accum
                finite_guard([ce, loss], 'training loss', device)
                (loss / accum).backward()
            ce_sum += ce.item() / accum

        finite_guard((p.grad for p in model.parameters()), 'training gradients', device)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        finite_guard([norm], 'gradient norm', device)
        grad_norm = norm.item()
        optimizer.step()
        finite_guard(model.parameters(), 'updated parameters', device)
        optimizer.zero_grad(set_to_none=True)

        tokens = (step + 1) * cfg['global_batch_tokens']
        if step % 10 == 0 or step == stop_step - 1:
            stats = torch.tensor([ce_sum, kd_sum], device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.AVG)
            elapsed = time.time() - t0
            log({'step': step + 1, 'tokens': tokens, 'loss_ce': stats[0].item(),
                 'loss_kd': stats[1].item() if teacher is not None else None,
                 'lr': cfg['lr'] * lr_scale, 'alpha': model._alpha, 'grad_norm': grad_norm,
                 'tokens_per_s': (tokens - start_step * cfg['global_batch_tokens']) / (elapsed - eval_time),
                 'elapsed_s': elapsed})
        if (step + 1) % eval_every == 0 or step == stop_step - 1:
            te = time.time()
            res = eval_both(model, cfg, device, world, rank)
            eval_time += time.time() - te
            log({'step': step + 1, 'tokens': tokens, **res,
                 **({'quant': quant_stats(model)} if cfg['mode'] == 'qat' else {})})
        if step + 1 in snapshot_steps:
            rank_zero_call(lambda: save_snapshot(model, step + 1, tokens, cfg))
        if save_every and (step + 1) % save_every == 0 and step + 1 < stop_step:
            te = time.time()
            if strict:
                save_training_state(model, optimizer, step + 1, cfg, identity, sampler,
                                    device, world, total_steps)
            else:
                rank_zero_call(lambda: save_checkpoint(model, optimizer, step + 1, tokens, cfg, 'latest'))
            eval_time += time.time() - te
            log({'event': 'checkpoint', 'step': step + 1, 'tokens': tokens,
                 'actual_usage': sampler.usage((step + 1) * seqs_per_step)})

    completed = stop_step == total_steps
    # A completed checkpoint resumed without updates still reports its restored validation.
    if completed and start_step == total_steps:
        res = eval_both(model, cfg, device, world, rank)
        log({'step': total_steps, 'tokens': total_steps * cfg['global_batch_tokens'], **res})
    tags = ('final',) if completed else ('latest',)
    if strict:
        save_training_state(model, optimizer, stop_step, cfg, identity, sampler,
                            device, world, total_steps, tags=tags)
    else:
        for tag in tags:
            rank_zero_call(lambda: save_checkpoint(model, optimizer, stop_step,
                           stop_step * cfg['global_batch_tokens'], cfg, tag))
    path = Path(cfg['run_dir']) / 'checkpoints' / ('final' if completed else 'latest')
    log({'event': 'done' if completed else 'paused', 'step': stop_step,
         'tokens': stop_step * cfg['global_batch_tokens'], 'checkpoint': str(path),
         'actual_usage': sampler.usage(stop_step * seqs_per_step)})
    if tb is not None:
        tb.close()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
