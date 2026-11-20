"""Opt-in exact pretraining resume and small all-rank boundary helpers."""
import math
from pathlib import Path

import torch
import torch.distributed as dist

from data_pipeline.sampling import sha256_file, validation_path
# Reuse the existing durable save and Python/NumPy/CPU/CUDA RNG implementation.
# The SFT entry point is guarded; importing it does not run training.
from qat.train_gsm8k import atomic_torch_save, capture_rng, restore_rng

RESUME_VERSION = 1
RESUME_MUTABLE = {'resume', 'max_steps', 'eval_only', 'run_dir', 'tensorboard', '_manifest'}


def distributed():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def rank_zero_call(function):
    """All ranks enter; rank-zero errors are broadcast instead of stranding peers."""
    result = [None, None]
    if not distributed() or dist.get_rank() == 0:
        try:
            result[0] = function()
        except Exception as error:
            result[1] = f'{type(error).__name__}: {error}'
    if distributed():
        dist.broadcast_object_list(result, src=0)
    if result[1] is not None:
        raise RuntimeError(result[1])
    return result[0]


def collective_check(ok, message, device):
    flag = torch.as_tensor(bool(ok), device=device, dtype=torch.int32)
    if distributed():
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    if not flag.item():
        raise RuntimeError(message)


def finite_guard(values, label, device):
    """Check all tensors locally and one flag globally, before any optimizer step."""
    flags = [torch.isfinite(value.detach()).all().to(device) for value in values if value is not None]
    ok = torch.stack(flags).all() if flags else torch.tensor(True, device=device)
    collective_check(ok.item(), f'Non-finite {label} on at least one rank; no checkpoint saved', device)


def model_identity(path):
    """Fingerprint local model config, weights, remote code and tokenizer assets."""
    root = Path(path).resolve()
    if not root.is_dir() or not (root / 'config.json').is_file():
        raise ValueError('strict_resume requires a local frozen model directory with config.json')
    suffixes = {'.json', '.safetensors', '.bin', '.pt', '.pth', '.py', '.model', '.txt', '.tiktoken'}
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix in suffixes
                   and '__pycache__' not in p.parts and '.cache' not in p.parts)
    if not any(p.suffix in {'.safetensors', '.bin', '.pt', '.pth'} for p in files):
        raise ValueError('No model weights found for strict model fingerprint')
    return {'path': str(root), 'files': {str(p.relative_to(root)): sha256_file(p) for p in files}}


def data_identity(sampler, cfg):
    manifest = cfg.get('_manifest')
    if manifest:
        return {'manifest_path': manifest['path'], 'manifest_sha256': manifest['sha256']}
    paths = [p for stream in sampler.streams for p in stream.paths]
    paths += [str(validation_path(cfg['val_data'], lang).resolve()) for lang in cfg['eval_langs']]
    if len(paths) != len(set(paths)):
        raise ValueError('Duplicate training/validation paths')
    return {'files': {p: sha256_file(p) for p in paths}}


def validate_resume(saved, cfg, identity, world, total_steps, sampler):
    if saved.get('resume_version') != RESUME_VERSION:
        raise ValueError('Not a complete strict pretraining resume checkpoint')
    required = ('model', 'optimizer', 'rng_by_rank', 'world', 'identity', 'config', 'step',
                'tokens', 'sample_cursor', 'usage', 'alpha', 'total_steps', 'completed')
    missing = [key for key in required if key not in saved]
    if missing:
        raise ValueError(f'Incomplete resume checkpoint: {missing}')
    if saved['world'] != world:
        raise ValueError('Resume world size changed')
    if saved['identity'] != identity:
        raise ValueError('Resume data/model identity changed')
    changed = [key for key in set(saved['config']) | set(cfg)
               if key not in RESUME_MUTABLE and saved['config'].get(key) != cfg.get(key)]
    if changed:
        raise ValueError(f'Resume configuration changed: {sorted(changed)}')
    step = saved['step']
    if (type(step) is not int or not 0 <= step <= total_steps or saved['total_steps'] != total_steps
            or saved['tokens'] != step * cfg['global_batch_tokens']
            or saved['sample_cursor'] != step * (cfg['global_batch_tokens'] // cfg['seq_len'])
            or saved['completed'] != (step == total_steps)):
        raise ValueError('Resume step/token/cursor/completion inconsistent')
    if saved['usage'] != sampler.usage(saved['sample_cursor']):
        raise ValueError('Resume sampler usage/cursor inconsistent')
    if not math.isfinite(saved['alpha']) or not 0 <= saved['alpha'] <= 1:
        raise ValueError('Resume alpha invalid')
    if len(saved['rng_by_rank']) != world or any(
            not isinstance(rng, dict) or not {'python', 'numpy', 'torch', 'cuda'} <= rng.keys()
            for rng in saved['rng_by_rank']):
        raise ValueError('Incomplete per-rank RNG state')
    opt = saved['optimizer']
    if not opt.get('param_groups') or 'state' not in opt or (step and not opt['state']):
        raise ValueError('Incomplete optimizer state')
    expected_params = {parameter for group in opt['param_groups'] for parameter in group['params']}
    if step and set(opt['state']) != expected_params:
        raise ValueError('Incomplete optimizer per-parameter state')
    for state in opt['state'].values():
        if not {'step', 'exp_avg', 'exp_avg_sq'} <= state.keys() or int(state['step']) != step:
            raise ValueError('Optimizer step inconsistent with resume step')
        if not all(torch.isfinite(state[key]).all() for key in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Non-finite optimizer resume state')
    if any(not torch.isfinite(value).all() for value in saved['model'].values()
           if torch.is_tensor(value) and value.is_floating_point()):
        raise ValueError('Non-finite model resume state')


def save_training_state(raw, optimizer, step, cfg, identity, sampler, device, world,
                        total_steps, tags=('latest',)):
    """All-rank optimizer boundary; RNG gathered once, writes durable before next collective."""
    finite_guard(raw.parameters(), 'checkpoint parameters', device)
    finite_guard((value for state in optimizer.state.values() for value in state.values()
                  if torch.is_tensor(value)), 'checkpoint optimizer', device)
    rng = capture_rng(device)
    states = [None] * world
    if distributed():
        dist.all_gather_object(states, rng)
    else:
        states[0] = rng
    cursor = step * (cfg['global_batch_tokens'] // cfg['seq_len'])

    def write():
        value = {'resume_version': RESUME_VERSION, 'model': raw.state_dict(),
                 'optimizer': optimizer.state_dict(), 'config': {k: v for k, v in cfg.items() if k != '_manifest'},
                 'identity': identity, 'world': world, 'rng_by_rank': states,
                 'step': step, 'tokens': step * cfg['global_batch_tokens'], 'sample_cursor': cursor,
                 'usage': sampler.usage(cursor), 'alpha': raw._alpha, 'total_steps': total_steps,
                 'completed': step == total_steps}
        for tag in tags:
            atomic_torch_save(value, Path(cfg['run_dir']) / 'checkpoints' / tag / 'state.pt')
    rank_zero_call(write)
    if distributed():
        dist.barrier()
