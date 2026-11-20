"""Independent answer-only GSM8K ternary SFT; does NOT launch downstream evaluation.

Full run (run once for each LR 1e-5, 3e-5, 1e-4 on separate GPU pairs)::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 qat/train_gsm8k.py \
        --data-dir data/gsm8k_sft_v1 --run-dir runs/gsm8k_sft_lr1e-5 --lr 1e-5

Resume with the SAME training arguments, adding::

    --resume runs/gsm8k_sft_lr1e-5/checkpoints/latest/state.pt

Manual GPU overfit/smoke test (not run by this script automatically)::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 qat/train_gsm8k.py \
        --data-dir data/gsm8k_sft_v1 --run-dir runs/gsm8k_overfit16 \
        --overfit-samples 16 --global-batch 16 --epochs 100 --lr 1e-4 \
        --max-steps 20 --save-every-steps 5

--max-steps pauses at an optimizer boundary WITHOUT changing the LR schedule;
remove/increase it on resume. --overfit-samples uses only the first N TRAIN rows,
never dev/test. Standard runs require 2 GPUs, global batch 64, and at most 5
epochs. --allow-other-world-size is for explicit diagnostic runs only.

Epoch FP32 master snapshots: snapshots/epoch_N/model.pt (existing eval loader
accepts the {'model': state_dict, ...} format). No best model is selected here:
external dev generation selects by strict accuracy, then CE, then earlier epoch.
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

# Only the trusted project source directory is added, never a downloaded data directory.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json, collate_examples, encode_example, load_prepared, sha256_file
from qat.quant import convert_to_ternary, set_alpha, set_quant_group, ternary_modules

ROOT = Path(__file__).resolve().parent.parent
RESUME_VERSION = 1
# Operational controls do not change the optimizer trajectory.
RESUME_MUTABLE = {'resume', 'max_steps', 'save_every_steps', 'log_every_steps', 'tensorboard',
                  'run_dir', 'allow_other_world_size'}


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
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        dir_fd = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def epoch_order(size, seed, epoch):
    indices = list(range(size))
    random.Random(seed + epoch).shuffle(indices)
    return indices


def rank_microbatches(indices, rank, world, micro_batch):
    """Shard one global update without ever repeating a real sample.

    All ranks execute equal numbers of microbatches; empty slots are None. There
    is deliberately no DistributedSampler, which would duplicate tail examples.
    """
    if not indices or world < 1 or not 0 <= rank < world or micro_batch < 1:
        raise ValueError('Invalid global update or rank/microbatch configuration')
    rounds = math.ceil(len(indices) / (world * micro_batch))
    padded = list(indices) + [None] * (rounds * world * micro_batch - len(indices))
    return [padded[(a * world + rank) * micro_batch:(a * world + rank + 1) * micro_batch]
            for a in range(rounds)]


def shifted_ce_sum(logits, labels):
    """Predict target[t+1] from logits[t]; skip prefix and right-padding labels."""
    targets = labels[:, 1:]
    active = targets.ne(-100)
    count = int(active.sum().item())
    if count == 0:
        # Connected zero: DDP hooks still run on dummy-only ranks, no NaN mean.
        return logits.reshape(-1)[0].float() * 0.0, 0
    selected = logits[:, :-1][active].float()
    return F.cross_entropy(selected, targets[active], reduction='sum'), count


def warmup_linear(step, total_steps, warmup_frac=0.05):
    """0-based optimizer update LR multiplier; warmup then decay to zero boundary."""
    if total_steps < 1 or not 0 <= step < total_steps:
        raise ValueError('LR step outside planned schedule')
    warm = max(1, math.ceil(total_steps * warmup_frac))
    if step < warm:
        return (step + 1) / warm
    # Last applied update remains positive; the post-training boundary is zero.
    return (total_steps - step) / max(1, total_steps - warm)


def assert_fp32_master(model):
    wrong = [name for name, p in model.named_parameters() if p.is_floating_point() and p.dtype != torch.float32]
    if wrong:
        raise ValueError(f'Non-FP32 master parameters: {wrong[:5]}')


def assert_alpha_one(model):
    modules = ternary_modules(model)
    if not modules or any(not m.quantize or m.alpha.item() != 1.0 for m in modules):
        raise RuntimeError('SFT requires enabled ternary quantization with alpha=1 at every update')


def capture_rng(device):
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state(device) if device.type == 'cuda' else None}


def restore_rng(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if device.type == 'cuda':
        torch.cuda.set_rng_state(state['cuda'].cpu(), device)


def validate_resume(saved, cfg, data_hash, world):
    if saved.get('resume_version') != RESUME_VERSION:
        raise ValueError('Not a complete GSM8K resume checkpoint; use --init-ckpt for B initialization')
    if saved['world'] != world or saved['data_hash'] != data_hash:
        raise ValueError('Resume world size or frozen data/tokenization identity changed')
    changed = [key for key in set(saved['config']) | set(cfg)
               if key not in RESUME_MUTABLE and saved['config'].get(key) != cfg.get(key)]
    if changed:
        raise ValueError(f'Resume training configuration changed: {sorted(changed)}')
    for required in ('model', 'optimizer', 'rng_by_rank', 'step', 'samples', 'tokens', 'epoch', 'update_in_epoch'):
        if required not in saved:
            raise ValueError(f'Incomplete resume checkpoint: {required}')
    if len(saved['rng_by_rank']) != world:
        raise ValueError('Incomplete per-rank RNG state')


def save_training_state(raw, optimizer, cfg, progress, data_hash, device, rank, world, tag='latest'):
    init_mode = cfg.get('init_mode', 'qat')
    precision_tag = 'fp32_master_ternary_qat' if init_mode == 'qat' else 'fp32_master_bf16_sft'
    rng = capture_rng(device)
    states = [None] * world
    if world > 1:
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    if rank == 0:
        value = {'resume_version': RESUME_VERSION, 'model': raw.state_dict(),
                 'optimizer': optimizer.state_dict(), 'rng_by_rank': states,
                 'world': world, 'data_hash': data_hash, 'config': cfg,
                 'alpha': 1.0 if init_mode == 'qat' else None,
                 'precision': precision_tag, **progress}
        atomic_torch_save(value, Path(cfg['run_dir']) / 'checkpoints' / tag / 'state.pt')
    if world > 1:
        dist.barrier()


def model_snapshot(raw, cfg, progress, data_hash, epoch, dev_ce):
    path = Path(cfg['run_dir']) / 'snapshots' / f'epoch_{epoch}' / 'model.pt'
    init_mode = cfg.get('init_mode', 'qat')
    precision_tag = 'fp32_master_ternary_qat' if init_mode == 'qat' else 'fp32_master_bf16_sft'
    atomic_torch_save({'model': raw.state_dict(), 'config': cfg, **progress, 'completed_epoch': epoch,
                       'data_hash': data_hash,
                       'alpha': 1.0 if init_mode == 'qat' else None,
                       'dev_answer_ce': dev_ce,
                       'precision': precision_tag}, path)
    return path


def to_device(batch, device):
    return {name: tensor.to(device) for name, tensor in batch.items()}


def autocast_for(device):
    return torch.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else nullcontext()


def train_update(model, raw, optimizer, examples, indices, rank, world, micro_batch,
                 pad_id, dummy_id, device, grad_clip=1.0):
    """One global answer-token-mean update, including all DDP/accumulation tails.

    DDP averages gradients by world. Each rank backprops SUM(loss) * world /
    global_target_count, exactly recovering d(sum losses)/global_target_count.
    There is no division by accumulation count or by (unequal) rank token counts.
    """
    micro_indices = rank_microbatches(indices, rank, world, micro_batch)
    local_tokens = sum(examples[i]['answer_tokens'] for batch in micro_indices for i in batch if i is not None)
    local_samples = sum(i is not None for batch in micro_indices for i in batch)
    counts = torch.tensor([local_tokens, local_samples], dtype=torch.long, device=device)
    if world > 1:
        dist.all_reduce(counts)
    global_tokens, global_samples = counts.tolist()
    if global_tokens <= 0 or global_samples != len(indices):
        raise RuntimeError('Invalid effective global target/sample count')
    optimizer.zero_grad(set_to_none=True)
    local_loss = torch.zeros((), dtype=torch.float64, device=device)
    observed_tokens = 0
    for micro_index, batch_indices in enumerate(micro_indices):
        batch = to_device(collate_examples([examples[i] if i is not None else None for i in batch_indices],
                                          pad_id, dummy_id), device)
        sync = micro_index == len(micro_indices) - 1
        context = model.no_sync() if world > 1 and not sync else nullcontext()
        with context:
            with autocast_for(device):
                logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'],
                               use_cache=False).logits
                loss_sum, count = shifted_ce_sum(logits, batch['labels'])
            # Detect every rank's failure together before entering backward collectives.
            finite = torch.isfinite(loss_sum.detach()).to(torch.int32)
            if world > 1:
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError('Nonfinite answer loss on at least one rank')
            (loss_sum * (world / global_tokens)).backward()
        local_loss += loss_sum.detach().double()
        observed_tokens += count
    if observed_tokens != local_tokens:
        raise RuntimeError('Encoded answer count disagrees with shifted loss target count')
    grad_norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), grad_clip, error_if_nonfinite=True).item()
    optimizer.step()
    if world > 1:
        dist.all_reduce(local_loss)
    return {'loss_ce': local_loss.item() / global_tokens, 'answer_tokens': global_tokens,
            'real_samples': global_samples, 'grad_norm': grad_norm}


@torch.no_grad()
def evaluate_ce(raw, examples, rank, world, micro_batch, pad_id, dummy_id, device):
    """Teacher-forced answer CE only. No generation and no model selection here."""
    was_training = raw.training
    raw.eval()
    total = torch.zeros(3, dtype=torch.float64, device=device)
    local_indices = list(range(rank, len(examples), world))
    for start in range(0, len(local_indices), micro_batch):
        subset = [examples[i] for i in local_indices[start:start + micro_batch]]
        batch = to_device(collate_examples(subset, pad_id, dummy_id), device)
        with autocast_for(device):
            logits = raw(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
            loss, count = shifted_ce_sum(logits, batch['labels'])
        total += torch.tensor([loss.item(), count, len(subset)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(total)
    raw.train(was_training)
    if not torch.isfinite(total).all() or total[1] <= 0 or int(total[2].item()) != len(examples):
        raise FloatingPointError('Invalid dev CE/counts')
    return {'answer_ce': (total[0] / total[1]).item(), 'answer_tokens': int(total[1].item()),
            'real_samples': int(total[2].item())}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--run-dir', required=True, help='New directory; existing directories require --resume')
    parser.add_argument('--model', default=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'))
    parser.add_argument('--init-ckpt', default=str(ROOT / 'runs/s3_B_qat_ramp_lr3e-3/checkpoints/final/state.pt'))
    parser.add_argument('--init-mode', choices=['qat', 'bf16'], default='qat',
                        help='Mode of the init checkpoint; bf16 skips ternary conversion and asserts float weights')
    parser.add_argument('--resume', help='Full GSM8K checkpoint, not an epoch model-only snapshot')
    parser.add_argument('--lr', type=float, choices=[1e-5, 3e-5, 1e-4], default=1e-5)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--global-batch', type=int, default=64)
    parser.add_argument('--micro-batch', type=int, default=2)
    parser.add_argument('--max-length', type=int, default=2048, help='Fail on any longer example; never truncate')
    parser.add_argument('--warmup-frac', type=float, default=0.05)
    parser.add_argument('--weight-decay', type=float, default=0.1)
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad-clip', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--save-every-steps', type=int, default=100)
    parser.add_argument('--log-every-steps', type=int, default=1)
    parser.add_argument('--max-steps', type=int, default=0, help='Pause after this absolute step; 0 means full schedule')
    parser.add_argument('--overfit-samples', type=int, default=0, help='Diagnostic mode, first N frozen training rows')
    parser.add_argument('--allow-other-world-size', action='store_true')
    parser.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--tensorboard', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    for name in ('epochs', 'global_batch', 'micro_batch', 'max_length', 'log_every_steps'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if min(args.save_every_steps, args.max_steps, args.overfit_samples) < 0:
        parser.error('Save/stop/overfit counts must be nonnegative')
    if not 0 < args.warmup_frac < 1 or args.grad_clip <= 0 or args.weight_decay < 0:
        parser.error('Invalid warmup, clipping, or weight decay')
    if not 0 <= args.beta1 < 1 or not 0 <= args.beta2 < 1:
        parser.error('AdamW betas must be in [0,1)')
    if not args.overfit_samples and (args.epochs > 5 or args.global_batch != 64):
        parser.error('Standard protocol uses global batch 64 and at most 5 epochs; override only in --overfit-samples mode')
    for name in ('data_dir', 'run_dir', 'model', 'init_ckpt', 'resume'):
        if getattr(args, name):
            setattr(args, name, str(Path(getattr(args, name)).resolve()))
    return vars(args)


def encoded_hash(train, dev, manifest_hash):
    digest = hashlib.sha256(manifest_hash.encode())
    for name, rows in [('train', train), ('dev', dev)]:
        digest.update(name.encode())
        for row in rows:
            digest.update(json.dumps(row, sort_keys=True, separators=(',', ':')).encode())
    return digest.hexdigest()


def main():
    cfg = parse_args()
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world != 2 and not cfg['allow_other_world_size']:
        raise ValueError('Standard run requires torchrun --nproc_per_node=2; use explicit diagnostic override otherwise')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Trainer requires a BF16-capable CUDA GPU; CPU correctness tests use helper functions')
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if world > 1:
        dist.init_process_group('nccl', timeout=timedelta(minutes=60))
    try:
        run_training(cfg, device, rank, world, local_rank)
    finally:
        # No swallowing exceptions: torchrun/stderr receives original traceback/nonzero exit.
        if dist.is_initialized():
            dist.destroy_process_group()


def run_training(cfg, device, rank, world, local_rank):
    from transformers import AutoTokenizer
    from qat.train import load_model, param_groups
    from qat.tb import TBLogger

    run_dir = Path(cfg['run_dir'])
    if rank == 0:
        if cfg['resume']:
            if not run_dir.is_dir() or not (run_dir / 'resolved_config.json').is_file():
                raise ValueError('--resume requires the existing run directory, not a new experiment')
            if not Path(cfg['resume']).is_relative_to(run_dir / 'checkpoints'):
                raise ValueError('Resume checkpoint must belong to this run directory')
        else:
            run_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    random.seed(cfg['seed'] + rank)
    np.random.seed(cfg['seed'] + rank)
    torch.manual_seed(cfg['seed'] + rank)
    torch.cuda.manual_seed(cfg['seed'] + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    datasets, manifest = load_prepared(cfg['data_dir'])
    if cfg['overfit_samples'] > len(datasets['train']):
        raise ValueError('--overfit-samples exceeds unique training set size')
    rows = datasets['train'][:cfg['overfit_samples']] if cfg['overfit_samples'] else datasets['train']
    tokenizer = AutoTokenizer.from_pretrained(cfg['model'])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dummy_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    train_examples = [encode_example(tokenizer, r, cfg['max_length']) for r in rows]
    dev_examples = [encode_example(tokenizer, r, cfg['max_length']) for r in datasets['dev']]
    data_hash = encoded_hash(train_examples, dev_examples, sha256_file(Path(cfg['data_dir']) / 'manifest.json'))
    if not train_examples:
        raise ValueError('No training examples')
    init_group = quant_group_of(cfg['init_ckpt'])
    cfg = {**cfg, 'mup_lr_scale': True, 'alpha': 1.0, 'precision': 'fp32_master_bf16_autocast',
           'quant_group': init_group}
    # Hash initial checkpoint once on rank 0; mmap below touches only model storage.
    source_hash = [sha256_file(cfg['init_ckpt']) if rank == 0 else None]
    if world > 1:
        dist.broadcast_object_list(source_hash)
    cfg['init_sha256'] = source_hash[0]
    if rank == 0 and not cfg['resume']:
        atomic_json(run_dir / 'resolved_config.json', cfg)
        atomic_json(run_dir / 'data_identity.json', {'data_hash': data_hash, 'manifest': manifest,
                    'train_examples': len(train_examples), 'dev_examples': len(dev_examples),
                    'train_answer_tokens': sum(r['answer_tokens'] for r in train_examples),
                    'max_train_length': max(len(r['input_ids']) for r in train_examples),
                    'max_dev_length': max(len(r['input_ids']) for r in dev_examples)})
    if rank == 0:
        print(json.dumps({'event': 'loading_model', 'init_ckpt': cfg['init_ckpt'],
                          'resume': cfg['resume'], 'data_hash': data_hash}), flush=True)
    raw = load_model(cfg['model'], device, torch.float32)
    if cfg['max_length'] > raw.config.max_position_embeddings:
        raise ValueError('--max-length exceeds model context')
    init_mode = cfg.get('init_mode', 'qat')
    if init_mode == 'qat':
        set_quant_group(init_group)
        replaced = convert_to_ternary(raw)
        if len(replaced) != 7 * raw.config.num_hidden_layers:
            raise ValueError(f'Unexpected number of ternary projections: {len(replaced)}')
    assert_fp32_master(raw)
    groups = param_groups(raw, cfg)
    if not math.isclose(groups[0]['lr_mult'], 0.25):
        raise ValueError('This protocol requires muP hidden matrix LR multiplier 0.25')
    optimizer = torch.optim.AdamW(groups, lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']), eps=1e-8)
    saved = None
    if cfg['resume']:
        saved = torch.load(cfg['resume'], map_location='cpu', weights_only=False, mmap=True)
        validate_resume(saved, cfg, data_hash, world)
        raw.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
    else:
        initial = torch.load(cfg['init_ckpt'], map_location='cpu', weights_only=False, mmap=True)
        if initial.get('config', {}).get('mode') != init_mode or 'optimizer' not in initial:
            raise ValueError(f'Initialization must be a full {init_mode} master checkpoint, not released/fixed weights')
        if any(t.is_floating_point() and t.dtype != torch.float32 for t in initial['model'].values()):
            raise ValueError('Initial model is not FP32 master state')
        raw.load_state_dict(initial['model'], strict=True)
        # Do not restore the optimizer, step, or RNG. Start a new SFT schedule.
        del initial
    if init_mode == 'qat':
        set_alpha(raw, 1.0)
        assert_alpha_one(raw)
    if cfg['gradient_checkpointing']:
        raw.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    raw.config.use_cache = False
    model = DDP(raw, device_ids=[local_rank], broadcast_buffers=False) if world > 1 else raw
    updates_per_epoch = math.ceil(len(train_examples) / cfg['global_batch'])
    total_steps = updates_per_epoch * cfg['epochs']
    progress = ({k: saved[k] for k in ('step', 'samples', 'tokens', 'epoch', 'update_in_epoch')} if saved
                else {'step': 0, 'samples': 0, 'tokens': 0, 'epoch': 0, 'update_in_epoch': 0})
    if (not 0 <= progress['epoch'] <= cfg['epochs'] or
            not 0 <= progress['update_in_epoch'] < updates_per_epoch or
            progress['step'] != progress['epoch'] * updates_per_epoch + progress['update_in_epoch']):
        raise ValueError('Resume data cursor/step is inconsistent')
    if saved:
        restore_rng(saved['rng_by_rank'][rank], device)
        del saved
    tb = TBLogger(run_dir, cfg) if rank == 0 and cfg['tensorboard'] else None

    def log(record):
        if rank != 0:
            return
        record = {'step': progress['step'], 'tokens': progress['tokens'], **record}
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        print(line, flush=True)
        with (run_dir / 'metrics.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(line + '\n')
            handle.flush()
        if tb:
            tb.log(record)

    try:
        log({'event': 'resume' if cfg['resume'] else 'start', 'world': world, 'total_steps': total_steps,
             'train_samples': len(train_examples), 'dev_samples': len(dev_examples),
             'alpha': 1.0 if init_mode == 'qat' else None, 'init_optimizer_reset': not bool(cfg['resume']), 'data_hash': data_hash,
             'sample_order': 'shuffle(seed+epoch), exact-once real samples, dummy-only tail fill'})
        if progress['step'] == 0:
            dev = evaluate_ce(raw, dev_examples, rank, world, cfg['micro_batch'], pad_id, dummy_id, device)
            val_key = 'val_ternary' if init_mode == 'qat' else 'val_float'
            log({'event': 'initial_dev_ce', val_key: {'gsm8k_answer_ce': dev['answer_ce']},
                 'dev_real_samples': dev['real_samples'], 'dev_answer_tokens': dev['answer_tokens']})
        raw.train()
        started = time.monotonic()
        while progress['epoch'] < cfg['epochs']:
            if cfg['max_steps'] and progress['step'] >= cfg['max_steps']:
                break
            epoch = progress['epoch']
            order = epoch_order(len(train_examples), cfg['seed'], epoch)
            for update in range(progress['update_in_epoch'], updates_per_epoch):
                if init_mode == 'qat':
                    assert_alpha_one(raw)
                scale = warmup_linear(progress['step'], total_steps, cfg['warmup_frac'])
                for group in optimizer.param_groups:
                    group['lr'] = cfg['lr'] * scale * group['lr_mult']
                indices = order[update * cfg['global_batch']:(update + 1) * cfg['global_batch']]
                result = train_update(model, raw, optimizer, train_examples, indices, rank, world,
                                      cfg['micro_batch'], pad_id, dummy_id, device, cfg['grad_clip'])
                progress['step'] += 1
                progress['samples'] += result['real_samples']
                progress['tokens'] += result['answer_tokens']
                progress['update_in_epoch'] = update + 1
                if progress['step'] % cfg['log_every_steps'] == 0 or update + 1 == updates_per_epoch:
                    log({'event': 'train', 'epoch': epoch + 1, **result,
                         'alpha': 1.0 if init_mode == 'qat' else None,
                         'samples_seen': progress['samples'], 'lr': cfg['lr'] * scale,
                         'lr_matrices': cfg['lr'] * scale * 0.25, 'elapsed_s': time.monotonic() - started})
                epoch_done = update + 1 == updates_per_epoch
                if epoch_done:
                    progress['epoch'] = epoch + 1
                    progress['update_in_epoch'] = 0
                    dev = evaluate_ce(raw, dev_examples, rank, world, cfg['micro_batch'], pad_id, dummy_id, device)
                    if rank == 0:
                        snapshot = model_snapshot(raw, cfg, progress, data_hash, epoch + 1, dev['answer_ce'])
                        log({'event': 'epoch_complete', 'epoch': epoch + 1, 'snapshot': str(snapshot),
                             'val_ternary' if init_mode == 'qat' else 'val_float': {'gsm8k_answer_ce': dev['answer_ce']},
                             'dev_real_samples': dev['real_samples'], 'dev_answer_tokens': dev['answer_tokens'],
                             'selection': 'external dev generation pending; no best model selected'})
                stop = bool(cfg['max_steps'] and progress['step'] >= cfg['max_steps'])
                periodic = bool(cfg['save_every_steps'] and progress['step'] % cfg['save_every_steps'] == 0)
                if epoch_done or periodic or stop:
                    save_training_state(raw, optimizer, cfg, progress, data_hash, device, rank, world)
                    log({'event': 'checkpoint', 'checkpoint': str(run_dir / 'checkpoints/latest/state.pt')})
                if stop:
                    break
        done = progress['epoch'] == cfg['epochs']
        if done:
            save_training_state(raw, optimizer, cfg, progress, data_hash, device, rank, world, tag='final')
        log({'event': 'done' if done else 'paused', 'completed_epochs': progress['epoch'],
             'samples_seen': progress['samples'], 'downstream_evaluation': 'external, not run by trainer'})
    finally:
        if tb:
            tb.close()


if __name__ == '__main__':
    main()
