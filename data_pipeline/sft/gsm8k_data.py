"""Train-only GSM8K preparation and answer-only SFT examples.

Read original datasets in isolated Python mode (never execute code beside them)::

    python -I data_pipeline/sft/gsm8k_data.py --train-file eval_data/gsm8k/train.jsonl \
        --test-file eval_data/gsm8k/test.jsonl --output-dir data/gsm8k_sft_v1

The output directory MUST be new. Test is read only for the overlap audit; no
split, filtering, or training decision depends on its contents. The first eight
original training rows remain the few-shot examples and are excluded from dev.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import unicodedata


FORMAT_VERSION = 1


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    """Atomic replacement, including fsync; never leave a partially written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f'.{path.name}.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def normalize_question(question):
    """NFKC + casefold + whitespace folding; do not erase numbers/punctuation."""
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().split())


def read_rows(path):
    rows = []
    with Path(path).open(encoding='utf-8') as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f'{path}:{line_no}: blank JSONL row')
            row = json.loads(line)
            for field in ('question', 'answer'):
                if not isinstance(row.get(field), str) or not row[field].strip():
                    raise ValueError(f'{path}:{line_no}: missing/non-string/empty {field}')
            rows.append(row)
    return rows


def split_rows(rows, seed=1234, dev_size=500, fewshot_size=8):
    """Deduplicate in source order, retaining canonical source IDs and audit trails."""
    if dev_size < 1 or fewshot_size < 1 or len(rows) < fewshot_size:
        raise ValueError('Need positive dev/few-shot sizes and enough source rows')
    canonical, unique, duplicates = {}, [], []
    for index, row in enumerate(rows):
        key = normalize_question(row['question'])
        if not key:
            raise ValueError(f'Empty normalized question at source index {index}')
        if key in canonical:
            first = canonical[key]
            if index < fewshot_size:
                raise ValueError('Duplicate within reserved few-shot rows: cannot preserve all and deduplicate')
            duplicates.append({'kept_source_index': first, 'dropped_source_index': index,
                               'answer_conflict': row['answer'] != rows[first]['answer']})
            continue
        canonical[key] = index
        unique.append({**row, 'source_index': index, 'source_id': f'train_{index}',
                       'question_sha256': hashlib.sha256(key.encode()).hexdigest()})
    candidates = [row['source_index'] for row in unique if row['source_index'] >= fewshot_size]
    if len(candidates) < dev_size:
        raise ValueError(f'Only {len(candidates)} unique non-few-shot rows for dev_size={dev_size}')
    dev_ids = set(random.Random(seed).sample(candidates, dev_size))
    train = [row for row in unique if row['source_index'] not in dev_ids]
    dev = [row for row in unique if row['source_index'] in dev_ids]
    fewshot = unique[:fewshot_size]
    return train, dev, fewshot, duplicates


def overlap_audit(left, right):
    lookup = {}
    for index, row in enumerate(right):
        lookup.setdefault(normalize_question(row['question']), []).append(index)
    return [{'left_source_index': row.get('source_index', index), 'right_indices': lookup[key]}
            for index, row in enumerate(left)
            if (key := normalize_question(row['question'])) in lookup]


def prepare(train_file, test_file, output_dir, seed=1234, dev_size=500):
    # All source reads precede directory creation, and never modify source files.
    source = read_rows(train_file)
    test = read_rows(test_file)
    train, dev, fewshot, duplicates = split_rows(source, seed, dev_size)
    audit = {'format_version': FORMAT_VERSION, 'seed': seed, 'dev_size': dev_size,
             'normalization': 'Unicode NFKC; casefold; collapse whitespace',
             'source_files': {'train': {'path': str(Path(train_file).resolve()), 'sha256': sha256_file(train_file)},
                              'test': {'path': str(Path(test_file).resolve()), 'sha256': sha256_file(test_file)}},
             'counts': {'source_train': len(source), 'source_test': len(test), 'train': len(train),
                        'dev': len(dev), 'fewshot': len(fewshot), 'duplicates_removed': len(duplicates)},
             'duplicates': duplicates, 'train_dev_overlap': overlap_audit(train, dev),
             'train_test_overlap': overlap_audit(train, test), 'dev_test_overlap': overlap_audit(dev, test),
             'indices': {'train': [r['source_index'] for r in train], 'dev': [r['source_index'] for r in dev],
                         'fewshot': [r['source_index'] for r in fewshot]},
             'test_policy': 'Read-only audit. Test does not influence splitting or deduplication.'}
    if audit['train_dev_overlap']:
        raise AssertionError('Internal error: normalized train/dev intersection')
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    audit['files'] = {}
    for name, rows in [('train', train), ('dev', dev), ('fewshot', fewshot)]:
        path = out / f'{name}.jsonl'
        with path.open('x', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + '\n')
            handle.flush()
            os.fsync(handle.fileno())
        audit['files'][path.name] = sha256_file(path)
    # Manifest is the completion marker. An interrupted preparation is rejected by the trainer.
    atomic_json(out / 'manifest.json', audit)
    return audit


def load_prepared(directory):
    """Verify frozen prepared bytes, counts, and split invariants before training."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['format_version'] != FORMAT_VERSION:
        raise ValueError('Unsupported GSM8K data format')
    datasets = {}
    for split in ('train', 'dev', 'fewshot'):
        name = f'{split}.jsonl'
        if sha256_file(directory / name) != manifest['files'][name]:
            raise ValueError(f'Data hash mismatch: {name}')
        datasets[split] = read_rows(directory / name)
        if len(datasets[split]) != manifest['counts'][split]:
            raise ValueError(f'Data count mismatch: {name}')
    train, dev, fewshot = (datasets[s] for s in ('train', 'dev', 'fewshot'))
    for name, rows in [('train', train), ('dev', dev)]:
        if len({normalize_question(r['question']) for r in rows}) != len(rows):
            raise ValueError(f'Duplicate normalized question in {name}')
    if overlap_audit(train, dev):
        raise ValueError('Train/dev leakage')
    by_index = {r['source_index']: r for r in train}
    if [r['source_index'] for r in fewshot] != list(range(8)):
        raise ValueError('Few-shot examples must be the first eight source rows')
    if any(by_index.get(r['source_index']) != r for r in fewshot):
        raise ValueError('Few-shot example absent from training')
    return datasets, manifest


def encode_example(tokenizer, row, max_length):
    """Explicit prompt/completion token boundary; only completion + EOS are targets.

    Prompt and completion are tokenized separately, avoiding BPE tokens straddling
    the supervision boundary. No truncation is ever requested from the tokenizer.
    """
    prefix = f"Question: {row['question']}\nAnswer:"
    prompt = tokenizer.encode(prefix, add_special_tokens=True, truncation=False)
    answer = tokenizer.encode(' ' + row['answer'], add_special_tokens=False, truncation=False)
    eos = tokenizer.eos_token_id
    if eos is None or not prompt or not answer:
        raise ValueError(f"Missing EOS or empty tokens for {row.get('source_id', 'unknown')}")
    # Some tokenizers append EOS to prompts by default. It is not part of the prompt.
    if prompt[-1] == eos:
        prompt = prompt[:-1]
    if not prompt:
        raise ValueError('Prompt has no tokens after removing automatic EOS')
    if answer[-1] != eos:
        answer.append(eos)
    ids = prompt + answer
    if len(ids) > max_length:
        raise ValueError(f"{row.get('source_id', 'unknown')}: {len(ids)} tokens exceeds --max-length "
                         f'{max_length} (prompt={len(prompt)}, answer+EOS={len(answer)}); '
                         'no truncation performed. Increase the explicit limit within model context.')
    return {'input_ids': ids, 'labels': [-100] * len(prompt) + answer,
            'answer_tokens': len(answer), 'source_id': row.get('source_id', 'unknown')}


def collate_examples(examples, pad_id, dummy_id):
    """Right padding with true masks; None denotes a zero-target dummy, not a repeat.

    A dummy has two attended tokens to keep fused attention kernels valid, but all
    labels are -100. Its loss/gradient contribution and real-example count are zero.
    """
    import torch
    if not examples:
        raise ValueError('Empty microbatch')
    rows = [e if e is not None else {'input_ids': [dummy_id, dummy_id], 'labels': [-100, -100]}
            for e in examples]
    length = max(len(e['input_ids']) for e in rows)
    ids = torch.full((len(rows), length), pad_id, dtype=torch.long)
    labels = torch.full_like(ids, -100)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        n = len(row['input_ids'])
        ids[index, :n] = torch.tensor(row['input_ids'])
        labels[index, :n] = torch.tensor(row['labels'])
        mask[index, :n] = 1
    return {'input_ids': ids, 'attention_mask': mask, 'labels': labels}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--train-file', required=True)
    parser.add_argument('--test-file', required=True, help='Read-only overlap audit, never used for splitting')
    parser.add_argument('--output-dir', required=True, help='Must not already exist')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dev-size', type=int, default=500)
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error('Run this raw-data reader with python -I')
    audit = prepare(args.train_file, args.test_file, args.output_dir, args.seed, args.dev_size)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)
    if audit['duplicates'] or audit['train_test_overlap'] or audit['dev_test_overlap']:
        print('WARNING: duplicates/overlaps found; inspect manifest.json before training.', file=sys.stderr)


if __name__ == '__main__':
    main()
