"""CPU-only frozen sampler, strict resume and two-rank gloo regression tests."""
import copy
import json
from pathlib import Path
import random
import sys
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sampling import DOMAIN_ORDER, MixedSampler, load_manifest, sha256_file, val_batches
from qat.checkpoint_utils import (capture_rng, finite_guard, model_identity, rank_zero_call,
                                  restore_rng, save_training_state, validate_resume)
from qat import train

DEVICE = torch.device('cpu')


def fixture_pool(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = {'version': 1, 'streams': {}}
    for i, lang in enumerate(DOMAIN_ORDER):
        stream = {'train': []}
        for split, count in [('train', 240), ('val', 48)]:
            path = root / f'{lang}_{split}.bin'
            (np.arange(count, dtype=np.uint32) % 13 + i).tofile(path)
            entry = {'path': path.name, 'sha256': sha256_file(path), 'tokens': count}
            if split == 'train':
                stream['train'].append(entry)
            else:
                stream['val'] = entry
        manifest['streams'][lang] = stream
    (root / 'manifest.json').write_text(json.dumps(manifest))
    (root / 'READY').write_text(json.dumps({'manifest_sha256': sha256_file(root / 'manifest.json')}))
    return root / 'manifest.json'


def sampler_for(root, weights=(.5, .5, 0, 0)):
    return MixedSampler(str(root), DOMAIN_ORDER, weights, 3, 123,
                        manifest=load_manifest(Path(root) / 'manifest.json'), canonical_order=True)


def test_frozen_sampler_and_usage(tmp_path):
    fixture_pool(tmp_path)
    a = sampler_for(tmp_path)
    passes = a.plan(32, max_data_passes=1)
    # Extra matching files are not included in the frozen manifest.
    np.arange(400, dtype=np.uint32).tofile(tmp_path / 'web_zh_train_extra.bin')
    b = sampler_for(tmp_path)
    b.plan(32)
    assert torch.equal(a.batch(0, 32), b.batch(0, 32))
    assert a.usage(32) == b.usage(32)
    assert passes['math'] == passes['code'] == 0
    assert sum(s['tokens'] for s in a.usage(32).values()) == 96
    assert list(a.usage(32)) == list(DOMAIN_ORDER)
    with pytest.raises(ValueError, match='max_data_passes'):
        a.plan(1000, max_data_passes=1)
    assert all(len(list(val_batches(tmp_path, lang, 3, 4, 2,
                                   manifest=load_manifest(tmp_path / 'manifest.json')))) == 2
               for lang in DOMAIN_ORDER)


@pytest.mark.parametrize('weights', [[0, 0, 0, 0], [-1, 1, 1, 1], [float('nan'), 1, 1, 1],
                                    [1, float('inf'), 0, 0], [1, 2]])
def test_bad_weights(tmp_path, weights):
    fixture_pool(tmp_path)
    with pytest.raises(ValueError, match='weights'):
        sampler_for(tmp_path, weights)


def test_manifest_rejections(tmp_path):
    path = fixture_pool(tmp_path)
    original = json.loads(path.read_text())
    bad = copy.deepcopy(original)
    bad['streams']['math']['train'] = bad['streams']['web_zh']['train']
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='Duplicate'):
        load_manifest(path)
    path.write_text(json.dumps(original))
    target = tmp_path / 'web_zh_train.bin'
    with target.open('r+b') as f:
        f.write(b'bad!')
    with pytest.raises(ValueError, match='SHA256'):
        load_manifest(path)
    (tmp_path / 'READY').unlink()
    with pytest.raises(ValueError, match='READY'):
        load_manifest(path)


def test_ready_binds_manifest(tmp_path):
    path = fixture_pool(tmp_path)
    (tmp_path / 'READY').write_text(json.dumps({'manifest_sha256': '0' * 64}))
    with pytest.raises(ValueError, match='READY manifest'):
        load_manifest(path)


def test_duplicate_and_empty_streams(tmp_path):
    np.arange(12, dtype=np.uint32).tofile(tmp_path / 'zh_train.bin')
    with pytest.raises(ValueError, match='unique'):
        MixedSampler([str(tmp_path), str(tmp_path)], ['zh'], [1], 3, 1)
    np.arange(2, dtype=np.uint32).tofile(tmp_path / 'empty_train.bin')
    with pytest.raises(ValueError, match='Empty domain'):
        MixedSampler(str(tmp_path), ['zh', 'empty'], [1, 0], 3, 1)


def test_model_identity_covers_weights_code_tokenizer(tmp_path):
    (tmp_path / 'config.json').write_text('{}')
    (tmp_path / 'model.safetensors').write_bytes(b'weights')
    (tmp_path / 'modeling_remote.py').write_text('# remote code')
    (tmp_path / 'tokenizer.json').write_text('{}')
    before = model_identity(tmp_path)
    for name in ('model.safetensors', 'modeling_remote.py', 'tokenizer.json', 'config.json'):
        with (tmp_path / name).open('ab') as handle:
            handle.write(b' ')
        after = model_identity(tmp_path)
        assert before != after
        before = after


def small_cfg(root):
    return dict(train.DEFAULTS, run_dir=str(root), langs=list(DOMAIN_ORDER), eval_langs=list(DOMAIN_ORDER),
                weights=[.5, .5, 0, 0], seq_len=3, micro_batch=1, global_batch_tokens=12,
                total_tokens=48, strict_resume=True, resume=None, max_steps=None, eval_only=False,
                alpha_ramp_frac=.75, warmup_frac=.5)


def make_model():
    model = torch.nn.Sequential(torch.nn.Linear(3, 7), torch.nn.Dropout(.3), torch.nn.Linear(7, 1))
    model._alpha = 1.0
    return model


def update(raw, wrapped, optimizer, sampler, cfg, step, rank=0, world=1):
    lr = cfg['lr'] * train.lr_at(step, 4, cfg)
    for group in optimizer.param_groups:
        group['lr'] = lr
    raw._alpha = train.alpha_at(step, 4, cfg)
    # Rank-local random factors exercise Python, NumPy and Torch RNG restoration.
    factor = random.random() + np.random.random() + torch.rand(()).item()
    size = 4 // world
    samples = sampler.batch(step * 4 + rank * size, size)
    loss = wrapped(samples[:, :3].float()).square().mean() * factor
    finite_guard([loss], 'loss', DEVICE)
    loss.backward()
    finite_guard((p.grad for p in raw.parameters()), 'gradients', DEVICE)
    norm = torch.nn.utils.clip_grad_norm_(raw.parameters(), 1.0)
    finite_guard([norm], 'norm', DEVICE)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {'step': step, 'lr': lr, 'alpha': raw._alpha, 'samples': samples.tolist()}


def single_checkpoint(tmp_path):
    fixture_pool(tmp_path / 'data')
    sampler = sampler_for(tmp_path / 'data')
    sampler.plan(16)
    cfg = small_cfg(tmp_path)
    raw = make_model()
    opt = torch.optim.AdamW(raw.parameters())
    update(raw, raw, opt, sampler, cfg, 0)
    identity = {'data': 'frozen', 'model': 'frozen'}
    save_training_state(raw, opt, 1, cfg, identity, sampler, DEVICE, 1, 4)
    path = tmp_path / 'checkpoints/latest/state.pt'
    return torch.load(path, weights_only=False), cfg, identity, sampler


@pytest.mark.parametrize('change', ['world', 'config', 'identity', 'version', 'tokens', 'cursor',
                                   'rng', 'optimizer', 'usage', 'completed', 'nonfinite'])
def test_strict_resume_rejects_inconsistent_state(tmp_path, change):
    saved, cfg, identity, sampler = single_checkpoint(tmp_path)
    validate_resume(saved, cfg, identity, 1, 4, sampler)
    if change == 'world': saved['world'] = 2
    if change == 'config': saved['config']['lr'] *= 2
    if change == 'identity': saved['identity']['model'] = 'changed'
    if change == 'version': saved.pop('resume_version')
    if change == 'tokens': saved['tokens'] += 1
    if change == 'cursor': saved['sample_cursor'] += 1
    if change == 'rng': saved['rng_by_rank'] = []
    if change == 'optimizer': next(iter(saved['optimizer']['state'].values()))['step'] += 1
    if change == 'usage': saved['usage']['math']['tokens'] += 1
    if change == 'completed': saved['completed'] = True
    if change == 'nonfinite': next(iter(saved['model'].values())).fill_(float('nan'))
    with pytest.raises(ValueError):
        validate_resume(saved, cfg, {'data': 'frozen', 'model': 'frozen'}, 1, 4, sampler)


def test_operational_controls_and_eval_rng(tmp_path):
    saved, cfg, identity, sampler = single_checkpoint(tmp_path)
    cfg.update(resume='/explicit/state.pt', max_steps=2, eval_only=True, run_dir='/new/run')
    validate_resume(saved, cfg, identity, 1, 4, sampler)
    model = make_model()
    before = capture_rng(DEVICE)
    with patch.object(train, '_eval_both', side_effect=lambda *a: torch.rand(2)):
        train.eval_both(model, cfg, DEVICE, 1, 0)
    assert torch.equal(before['torch'], capture_rng(DEVICE)['torch'])


def assert_nested_equal(a, b):
    if torch.is_tensor(a):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a: assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b): assert_nested_equal(x, y)
    else:
        assert a == b


def gloo_worker(rank, root, init_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', rank=rank, world_size=2, init_method=f'file://{init_path}')
    try:
        root = Path(root)
        cfg = small_cfg(root)
        sampler = sampler_for(root / 'data')
        sampler.plan(16, max_data_passes=1)
        identity = {'data': 'frozen', 'model': 'synthetic'}
        random.seed(10 + rank)
        np.random.seed(10 + rank)
        torch.manual_seed(10 + rank)
        raw = make_model()
        ddp = DDP(raw)
        opt = torch.optim.AdamW(raw.parameters())
        records = [update(raw, ddp, opt, sampler, cfg, s, rank, 2) for s in range(2)]
        save_training_state(raw, opt, 2, cfg, identity, sampler, DEVICE, 2, 4)
        records += [update(raw, ddp, opt, sampler, cfg, s, rank, 2) for s in range(2, 4)]
        expected = copy.deepcopy(raw.state_dict()), copy.deepcopy(opt.state_dict()), capture_rng(DEVICE)
        saved = torch.load(root / 'checkpoints/latest/state.pt', weights_only=False)
        validate_resume(saved, cfg, identity, 2, 4, sampler)
        resumed = make_model()
        resumed.load_state_dict(saved['model'])
        resumed._alpha = saved['alpha']
        ddp2 = DDP(resumed)
        opt2 = torch.optim.AdamW(resumed.parameters())
        opt2.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_by_rank'][rank], DEVICE)
        replay = [update(resumed, ddp2, opt2, sampler, cfg, s, rank, 2) for s in range(2, 4)]
        assert records[2:] == replay
        assert_nested_equal(expected[0], resumed.state_dict())
        assert_nested_equal(expected[1], opt2.state_dict())
        assert torch.equal(expected[2]['torch'], capture_rng(DEVICE)['torch'])
        save_training_state(resumed, opt2, 4, cfg, identity, sampler, DEVICE, 2, 4,
                            tags=('final', 'latest'))
        final = torch.load(root / 'checkpoints/final/state.pt', weights_only=False)
        latest = torch.load(root / 'checkpoints/latest/state.pt', weights_only=False)
        assert final['completed'] and latest['step'] == final['step'] == 4
        # One bad rank must fail ALL ranks before an optimizer step or checkpoint.
        for label in ('loss', 'gradients'):
            value = torch.tensor(float('nan') if rank else 1.0)
            try:
                finite_guard([value], label, DEVICE)
            except RuntimeError as error:
                assert 'Non-finite' in str(error)
            else:
                raise AssertionError('finite guard did not fail')
        try:
            rank_zero_call(lambda: (_ for _ in ()).throw(OSError('injected I/O failure')))
        except RuntimeError as error:
            assert 'injected I/O failure' in str(error)
        else:
            raise AssertionError('I/O error not broadcast')
        (root / f'rank{rank}.ok').write_text('exact resume; finite and I/O failure collective\n')
    finally:
        dist.destroy_process_group()


def test_two_rank_gloo_exact_resume_and_collective_failures(tmp_path):
    fixture_pool(tmp_path / 'data')
    mp.spawn(gloo_worker, args=(str(tmp_path), str(tmp_path / 'gloo-init')), nprocs=2, join=True)
    assert (tmp_path / 'rank0.ok').is_file() and (tmp_path / 'rank1.ok').is_file()
