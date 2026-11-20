"""CPU-only correctness tests. Run: python -m pytest tests/test_gsm8k_sft.py -q"""
import copy
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import (collate_examples, encode_example, load_prepared, normalize_question,
                                          overlap_audit, prepare, sha256_file, split_rows)
from qat.quant import TernaryLinear, convert_to_ternary, set_alpha
from qat.train_gsm8k import (assert_alpha_one, atomic_torch_save, capture_rng, epoch_order, evaluate_ce,
                             model_snapshot, quant_group_of, rank_microbatches, restore_rng, save_training_state,
                             shifted_ce_sum, train_update, validate_resume, warmup_linear)


def test_initial_checkpoint_determines_sft_quant_group(tmp_path):
    row = tmp_path / 'runs/row/checkpoints/final/state.pt'
    row.parent.mkdir(parents=True)
    row.write_bytes(b'fixture')
    (tmp_path / 'runs/row/resolved_config.json').write_text('{"mode": "qat"}', encoding='utf-8')
    g32 = tmp_path / 'runs/g32/checkpoints/final/state.pt'
    g32.parent.mkdir(parents=True)
    g32.write_bytes(b'fixture')
    (tmp_path / 'runs/g32/resolved_config.json').write_text(
        '{"mode": "qat", "quant_group": 32}', encoding='utf-8')
    assert quant_group_of(row) is None
    assert quant_group_of(g32) == 32


class Tokenizer:
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text, add_special_tokens=False, truncation=False):
        assert not truncation
        return ([self.bos_token_id] if add_special_tokens else []) + [3 + ord(c) % 25 for c in text]


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, 8)
        self.layers = nn.ModuleList([nn.ModuleDict({'q_proj': nn.Linear(8, 8)})])
        self.lm_head = nn.Linear(8, 32)
        self.config = SimpleNamespace(dim_model_base=2, hidden_size=8)

    def forward(self, input_ids, attention_mask, use_cache=False):
        assert not use_cache
        hidden = self.embed_tokens(input_ids) * attention_mask.unsqueeze(-1)
        hidden = hidden.cumsum(dim=1) / attention_mask.cumsum(dim=1).clamp(min=1).unsqueeze(-1)
        hidden = self.layers[0]['q_proj'](hidden).tanh()
        return SimpleNamespace(logits=self.lm_head(hidden))


def tiny():
    torch.manual_seed(42)
    model = TinyLM()
    assert convert_to_ternary(model) == ['layers.0.q_proj']
    set_alpha(model, 1)
    return model


def examples():
    return [encode_example(Tokenizer(), {'question': str(i), 'answer': 'x' * (i * 3 + 1),
                                        'source_id': str(i)}, 200) for i in range(11)]


def rows(size=40):
    return [{'question': f'Question {i}?', 'answer': f'Reason\n#### {i}'} for i in range(size)]


def write_rows(path, records):
    path.write_text(''.join(json.dumps(r) + '\n' for r in records))


def test_split_deterministic_reserved_dedup_and_audit():
    source = rows()
    source.append({'question': '  ＱＵＥＳＴＩＯＮ\t0? ', 'answer': source[0]['answer']})
    source.append({'question': 'question 12?', 'answer': 'different'})
    train, dev, few, dup = split_rows(source, seed=17, dev_size=7)
    assert len(train) == 33 and len(dev) == 7
    assert [r['source_index'] for r in few] == list(range(8))
    assert all(r in train for r in few)
    assert not overlap_audit(train, dev)
    assert len(dup) == 2 and dup[1]['answer_conflict']
    assert (train, dev, few, dup) == split_rows(source, seed=17, dev_size=7)
    assert [r['source_index'] for r in dev] != [r['source_index'] for r in split_rows(source, 18, 7)[1]]
    assert normalize_question('Ａ  B\nC') == 'a b c'
    source[2] = source[0]
    with pytest.raises(ValueError, match='reserved few-shot'):
        split_rows(source, dev_size=7)


def test_prepare_read_only_hashes_and_test_independence(tmp_path):
    train_file, test_file = tmp_path / 'source_train.jsonl', tmp_path / 'source_test.jsonl'
    write_rows(train_file, rows())
    write_rows(test_file, [rows()[0]])
    original_hash = sha256_file(train_file)
    audit = prepare(train_file, test_file, tmp_path / 'new', seed=31, dev_size=7)
    assert sha256_file(train_file) == original_hash
    assert audit['train_test_overlap'] == [{'left_source_index': 0, 'right_indices': [0]}]
    datasets, manifest = load_prepared(tmp_path / 'new')
    assert len(datasets['train']) == 33 and manifest == audit
    write_rows(test_file, [{'question': 'No overlap', 'answer': '#### 3'}])
    other = prepare(train_file, test_file, tmp_path / 'different_test', seed=31, dev_size=7)
    assert other['indices'] == audit['indices']
    with pytest.raises(FileExistsError):
        prepare(train_file, test_file, tmp_path / 'new', dev_size=7)
    with (tmp_path / 'new/train.jsonl').open('a') as handle:
        handle.write('\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_prepared(tmp_path / 'new')


def test_answer_mask_padding_eos_shift_and_no_truncation():
    tok = Tokenizer()
    record = {'question': '2+2?', 'answer': 'work\n#### 4', 'source_id': 'train_0'}
    encoded = encode_example(tok, record, 100)
    prefix_len = len(tok.encode('Question: 2+2?\nAnswer:', add_special_tokens=True))
    assert encoded['labels'][:prefix_len] == [-100] * prefix_len
    assert encoded['labels'][prefix_len:] == tok.encode(' ' + record['answer']) + [tok.eos_token_id]
    assert encoded['input_ids'][-1] == tok.eos_token_id
    short = encode_example(tok, {'question': 'x', 'answer': 'a'}, 100)
    batch = collate_examples([encoded, short, None], 0, 1)
    n = len(short['input_ids'])
    assert batch['attention_mask'][1, n:].eq(0).all()
    assert batch['labels'][1, n:].eq(-100).all()
    assert batch['labels'][2].eq(-100).all()
    # First answer token must be predicted at the last PREFIX position, not at itself.
    logits = torch.full((*batch['labels'].shape, 32), -10.0)
    for b in range(3):
        for t in range(1, batch['labels'].shape[1]):
            target = batch['labels'][b, t].item()
            if target != -100:
                logits[b, t - 1, target] = 10
    ce, count = shifted_ce_sum(logits, batch['labels'])
    assert ce.item() < 1e-4
    assert count == encoded['answer_tokens'] + short['answer_tokens']
    with pytest.raises(ValueError, match='no truncation performed'):
        encode_example(tok, record, len(encoded['input_ids']) - 1)


def test_dummy_loss_connected_zero_and_mask_invariance():
    model = tiny()
    batch = collate_examples([None], 0, 1)
    ce, count = shifted_ce_sum(model(**{k: v for k, v in batch.items() if k != 'labels'}).logits, batch['labels'])
    ce.backward()
    assert count == 0 and ce.item() == 0
    assert all(p.grad is not None and p.grad.eq(0).all() for p in model.parameters())
    short, long = examples()[0], examples()[-1]
    one = collate_examples([short], 0, 1)
    padded = collate_examples([short, long], 0, 1)
    a, _ = shifted_ce_sum(model(input_ids=one['input_ids'], attention_mask=one['attention_mask']).logits, one['labels'])
    b, _ = shifted_ce_sum(model(input_ids=padded['input_ids'], attention_mask=padded['attention_mask']).logits[:1], padded['labels'][:1])
    torch.testing.assert_close(a, b)


@pytest.mark.parametrize('size,world,micro', [(1, 2, 2), (65, 2, 2), (64, 2, 2), (7, 3, 2), (11, 2, 2)])
def test_sampler_no_repeats_and_equal_rank_work(size, world, micro):
    order = epoch_order(size, 1234, 0)
    assert sorted(order) == list(range(size))
    ranks = [rank_microbatches(order, rank, world, micro) for rank in range(world)]
    assert len({len(r) for r in ranks}) == 1
    actual = [index for rank in ranks for batch in rank for index in batch if index is not None]
    assert sorted(actual) == list(range(size))
    assert all(len(batch) == micro for rank in ranks for batch in rank)


def ddp_worker(rank, world, rendezvous, output):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method=f'file://{rendezvous}', rank=rank, world_size=world)
    try:
        model = tiny()
        ddp = DDP(model, broadcast_buffers=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
        data = examples()
        records = []
        # Unequal answer lengths, accumulation, a partly filled last microbatch,
        # then an entire dummy-only rank (tail of one real sample).
        for ids in [list(range(11)), [0]]:
            records.append(train_update(ddp, model, optimizer, data, ids, rank, world, 2,
                                        0, 1, torch.device('cpu'), grad_clip=100))
        dev = evaluate_ce(model, data, rank, world, 2, 0, 1, torch.device('cpu'))
        if rank == 0:
            atomic_torch_save({'model': model.state_dict(), 'records': records, 'dev': dev}, output)
    finally:
        dist.destroy_process_group()


def test_two_rank_global_token_normalization_matches_single_process(tmp_path):
    torch.set_num_threads(1)
    output = str(tmp_path / 'ddp.pt')
    mp.spawn(ddp_worker, args=(2, str(tmp_path / 'rendezvous'), output), nprocs=2, join=True)
    actual = torch.load(output, weights_only=True)
    model = tiny()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.03)
    records = []
    for ids in [list(range(11)), [0]]:
        records.append(train_update(model, model, optimizer, examples(), ids, 0, 1, 20,
                                    0, 1, torch.device('cpu'), grad_clip=100))
    for key, value in model.state_dict().items():
        torch.testing.assert_close(actual['model'][key], value, rtol=1e-5, atol=2e-7)
    for a, b in zip(actual['records'], records):
        assert a['answer_tokens'] == b['answer_tokens'] and a['real_samples'] == b['real_samples']
        assert a['loss_ce'] == pytest.approx(b['loss_ce'], rel=1e-6)
        assert a['grad_norm'] == pytest.approx(b['grad_norm'], rel=1e-5)
    dev = evaluate_ce(model, examples(), 0, 1, 3, 0, 1, torch.device('cpu'))
    assert actual['dev']['answer_ce'] == pytest.approx(dev['answer_ce'], rel=1e-6)
    assert actual['dev']['real_samples'] == 11


def test_atomic_save_preserves_previous_on_failure(tmp_path):
    path = tmp_path / 'state.pt'
    atomic_torch_save({'value': 1}, path)
    with patch('qat.train_gsm8k.torch.save', side_effect=OSError('disk full')):
        with pytest.raises(OSError, match='disk full'):
            atomic_torch_save({'value': 2}, path)
    assert torch.load(path, weights_only=True) == {'value': 1}
    assert list(tmp_path.iterdir()) == [path]


def test_full_resume_optimizer_rng_cursor_and_model_only(tmp_path):
    torch.set_num_threads(1)
    device = torch.device('cpu')
    model = tiny()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    data = examples()
    cfg = {'run_dir': str(tmp_path), 'lr': 0.005, 'epochs': 5, 'resume': None, 'max_steps': 1}
    result = train_update(model, model, optimizer, data, list(range(7)), 0, 1, 2, 0, 1, device)
    progress = {'step': 1, 'tokens': result['answer_tokens'], 'samples': 7, 'epoch': 0, 'update_in_epoch': 1}
    save_training_state(model, optimizer, cfg, progress, 'hash', device, 0, 1)
    snapshot = model_snapshot(model, cfg, progress, 'hash', 1, 3.2)
    model_only = torch.load(snapshot, weights_only=False)
    assert 'optimizer' not in model_only
    evaluator = tiny()
    evaluator.load_state_dict(model_only['model'], strict=True)
    assert 'layers.0.q_proj.alpha' not in model_only['model']  # existing quant.py uses nonpersistent alpha
    saved = torch.load(tmp_path / 'checkpoints/latest/state.pt', weights_only=False)
    validate_resume(saved, {**cfg, 'resume': 'latest', 'max_steps': 20}, 'hash', 1)
    with pytest.raises(ValueError, match='configuration changed'):
        validate_resume(saved, {**cfg, 'lr': 0.01}, 'hash', 1)
    with pytest.raises(ValueError, match='identity changed'):
        validate_resume(saved, cfg, 'changed-data', 1)
    with pytest.raises(ValueError, match='identity changed'):
        validate_resume(saved, cfg, 'hash', 2)
    with pytest.raises(ValueError, match='complete GSM8K'):
        validate_resume(model_only, cfg, 'hash', 1)
    expected_rng = (random.random(), np.random.rand(), torch.rand(5))
    resumed = tiny()
    resumed.load_state_dict(saved['model'])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.2)
    resumed_optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng_by_rank'][0], device)
    assert random.random() == expected_rng[0] and np.random.rand() == expected_rng[1]
    torch.testing.assert_close(torch.rand(5), expected_rng[2], rtol=0, atol=0)
    for m, opt in [(model, optimizer), (resumed, resumed_optimizer)]:
        train_update(m, m, opt, data, list(range(7, 11)), 0, 1, 2, 0, 1, device)
        assert_alpha_one(m)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, resumed.state_dict()[key], rtol=0, atol=0)
    for left, right in zip(optimizer.state.values(), resumed_optimizer.state.values()):
        for key in left:
            torch.testing.assert_close(left[key], right[key], rtol=0, atol=0)


def test_nonfinite_loss_raises_without_optimizer_update():
    model = tiny()
    with torch.no_grad():
        model.lm_head.bias.fill_(float('nan'))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError, match='Nonfinite answer loss'):
        train_update(model, model, optimizer, examples(), [0], 0, 1, 2,
                     0, 1, torch.device('cpu'))


def test_tensorboard_training_and_dev_scalars(tmp_path):
    from qat.tb import TBLogger
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    logger = TBLogger(tmp_path, {'task': 'gsm8k_sft'})
    logger.log({'step': 1, 'tokens': 123, 'loss_ce': 2.0, 'alpha': 1.0,
                'real_samples': 7, 'answer_tokens': 123, 'lr_matrices': 2.5e-6})
    logger.log({'step': 1, 'tokens': 123, 'val_ternary': {'gsm8k_answer_ce': 2.1}})
    logger.close()
    events = EventAccumulator(str(tmp_path / 'tb')).Reload()
    assert events.Scalars('train/loss_ce')[0].step == 123
    assert events.Scalars('train/real_samples')[0].value == 7
    assert events.Scalars('val_ternary/gsm8k_answer_ce')[0].value == pytest.approx(2.1)


def test_scheduler_and_mup_groups():
    from qat.train import param_groups
    model = tiny()
    groups = param_groups(model, {'mup_lr_scale': True, 'weight_decay': 0.1})
    assert [g['lr_mult'] for g in groups] == [0.25, 1, 1]
    assert [g['weight_decay'] for g in groups] == [0.1, 0, 0]
    values = [warmup_linear(i, 100) for i in range(100)]
    assert values[:5] == pytest.approx([0.2, 0.4, 0.6, 0.8, 1])
    assert values[-1] == pytest.approx(1 / 95)
    assert all(a >= b for a, b in zip(values[5:], values[6:]))
    assert warmup_linear(0, 1) == 1
