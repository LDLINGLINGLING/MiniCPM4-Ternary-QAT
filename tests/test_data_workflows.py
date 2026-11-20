"""Small real CLI workflows using only synthetic local data."""

import json
import os
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data_pipeline.sft import prepare_sft_v2, prepare_sft_v3


ROOT = Path(__file__).resolve().parents[1]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows), encoding='utf-8')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]


def run_cli(script, *args):
    env = dict(
        os.environ,
        PYTHONDONTWRITEBYTECODE='1',
        HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',
    )
    result = subprocess.run(
        [sys.executable, '-I', str(script), *map(str, args)],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return result


def local_tokenizer(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {'[UNK]': 0, '<s>': 1, '</s>': 2, 'alpha': 3, 'beta': 4, 'gamma': 5}
    tokenizer = Tokenizer(WordLevel(vocabulary, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='[UNK]',
        bos_token='<s>',
        eos_token='</s>',
    )
    fast.save_pretrained(path)
    return vocabulary


def test_prepare_data_parquet_to_uint32_with_real_local_tokenizer(tmp_path):
    model = tmp_path / 'tokenizer'
    vocabulary = local_tokenizer(model)
    documents = ['alpha beta', 'gamma', '', 'beta gamma alpha', 'alpha', None]
    source = tmp_path / 'toy.parquet'
    pq.write_table(pa.table({'content': documents}), source)
    output = tmp_path / 'tokens'

    run_cli(
        ROOT / 'data_pipeline/pretrain/prepare_data.py',
        '--tokenizer', model,
        '--input', f'toy={source}',
        '--output', output,
        '--val-docs', 2,
        '--min-chars', 1,
        '--workers', 2,
        '--batch-docs', 2,
        '--seed', 17,
    )

    kept = [text for text in documents if isinstance(text, str) and len(text.strip()) >= 1]
    order = np.random.default_rng(17).permutation(len(kept))
    shuffled = [kept[index] for index in order]
    metadata = json.loads((output / 'meta.json').read_text(encoding='utf-8'))
    for split, texts in (('val', shuffled[:2]), ('train', shuffled[2:])):
        expected_documents = [
            np.asarray([1, *(vocabulary[word] for word in text.split()), 2], dtype=np.uint32)
            for text in texts
        ]
        expected = np.concatenate(expected_documents)
        actual = np.fromfile(output / f'toy_{split}.bin', dtype='<u4')
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(
            np.load(output / f'toy_{split}_doclens.npy'),
            np.asarray([len(document) for document in expected_documents], dtype=np.int64),
        )
        assert metadata['splits'][f'toy_{split}']['docs'] == 2
        assert metadata['splits'][f'toy_{split}']['tokens'] == len(expected)


def general_record(index, split='train'):
    return {
        'prompt': f'General instruction {split} {index}\n',
        'response': f'General response {index}',
        'source': 'general/synthetic',
        'id': f'general_{split}_{index}',
    }


def gsm_record(index, split='train'):
    return {
        'question': f'What is synthetic math question {split} number {index}?',
        'answer': f'Synthetic answer {index}',
        'source_id': f'{split}_{index}',
    }


def write_general_and_gsm(tmp_path, general_count=12):
    general = tmp_path / 'general'
    gsm = tmp_path / 'gsm'
    write_jsonl(general / 'train.jsonl', [general_record(i) for i in range(general_count)])
    write_jsonl(general / 'dev.jsonl', [general_record(0, 'dev')])
    write_jsonl(gsm / 'train.jsonl', [gsm_record(i) for i in range(2)])
    write_jsonl(gsm / 'dev.jsonl', [gsm_record(0, 'dev')])
    return general, gsm


def benchmark_root(tmp_path):
    root = tmp_path / 'benchmarks'
    for split in ('train', 'test'):
        write_jsonl(root / f'eval_data/gsm8k/{split}.jsonl', [{'question': f'Benchmark GSM8K {split}?'}])
    write_jsonl(root / 'eval_data/gsm8k_variants_v1/numeric.jsonl', [{'question': 'Benchmark variant?'}])
    for benchmark, splits in (
        ('mmlu/data', ('dev', 'val', 'test')),
        ('cmmlu', ('dev', 'test')),
        ('ceval', ('dev', 'val', 'test')),
    ):
        for split in splits:
            path = root / f'eval_data/{benchmark}/{split}/sample.csv'
            path.parent.mkdir(parents=True, exist_ok=True)
            if benchmark == 'mmlu/data':
                path.write_text('Benchmark MMLU question?,one,two,three,four,A\n', encoding='utf-8')
            else:
                field = 'Question' if benchmark == 'cmmlu' else 'question'
                path.write_text(f'{field},A,B\nBenchmark {benchmark} {split}?,one,two\n', encoding='utf-8')
    return root


def test_mix_gsm8k_sft_cli(tmp_path):
    general, gsm = write_general_and_gsm(tmp_path, general_count=2)
    output = tmp_path / 'mixed'
    run_cli(
        ROOT / 'data_pipeline/sft/mix_gsm8k_sft.py',
        '--alpaca', general,
        '--gsm', gsm,
        '--output', output,
        '--repeat', 3,
        '--seed', 23,
    )
    train = read_jsonl(output / 'train.jsonl')
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    assert len(train) == 8
    assert len(read_jsonl(output / 'dev.jsonl')) == 2
    assert len({row['id'] for row in train}) == 8
    assert manifest['train_size'] == 8
    assert manifest['sources']['gsm8k_repeated'] == 6
    assert manifest['gsm8k_share'] == 0.75


def test_sft_v2_and_v3_cli_workflows(tmp_path):
    general, gsm = write_general_and_gsm(tmp_path)
    benchmarks = benchmark_root(tmp_path)
    code = tmp_path / 'code'
    write_jsonl(code / 'python.jsonl', [
        {'lang': 'python', 'text': f'Write synthetic function {index}\n```python\nreturn {index}\n```'}
        for index in range(6)
    ])
    firefly = tmp_path / 'firefly.jsonl'
    write_jsonl(firefly, [
        {'kind': 'Program', 'input': 'Write a Firefly synthetic program', 'target': 'print("synthetic")'},
        {'kind': 'OpenQA', 'input': 'Excluded from code', 'target': 'not a program'},
    ])
    v2 = tmp_path / 'sft_v2'
    run_cli(
        ROOT / 'data_pipeline/sft/prepare_sft_v2.py',
        '--general', general,
        '--gsm', gsm,
        '--code-dir', code,
        '--firefly', firefly,
        '--benchmark-root', benchmarks,
        '--output', v2,
        '--total', 20,
        '--code-frac', 0.25,
        '--math-frac', 0.15,
        '--math-repeat', 2,
        '--seed', 23,
    )
    v2_train = read_jsonl(v2 / 'train.jsonl')
    v2_manifest = json.loads((v2 / 'manifest.json').read_text(encoding='utf-8'))
    assert v2_manifest['train_size'] == 20
    assert v2_manifest['categories'] == {'general': 11, 'code': 5, 'gsm8k': 4}
    assert len(v2_train) == 20
    assert len(read_jsonl(v2 / 'dev.jsonl')) == 2

    ultra = tmp_path / 'ultra'
    ultra.mkdir()
    content = (
        'Calculate the synthetic value of this sufficiently long mathematical problem?\n'
        + 'We show every synthetic reasoning step clearly. ' * 8
    )
    second_content = content.replace('Calculate', 'Compute', 1).replace('value', 'result', 1)
    pq.write_table(pa.table({'content': [content, second_content]}), ultra / 'shard.parquet')
    v3 = tmp_path / 'sft_v3'
    run_cli(
        ROOT / 'data_pipeline/sft/prepare_sft_v3.py',
        '--v2', v2,
        '--gsm', gsm,
        '--ultra', ultra,
        '--benchmark-root', benchmarks,
        '--output', v3,
        '--math-size', 6,
        '--gsm-repeat', 2,
        '--seed', 23,
    )
    v3_train = read_jsonl(v3 / 'train.jsonl')
    v3_manifest = json.loads((v3 / 'manifest.json').read_text(encoding='utf-8'))
    assert v3_manifest['train_size'] == 22
    assert v3_manifest['categories'] == {'general': 11, 'code': 5, 'math': 6}
    assert v3_manifest['sources']['math/ultradata_l3_qa'] == 2
    assert v3_manifest['sources']['gsm8k/train'] == 4
    assert {row['id'] for row in v3_train if row['source'].startswith(('general/', 'code/'))} == {
        row['id'] for row in v2_train if row['source'].startswith(('general/', 'code/'))
    }


def test_sft_quota_edges_fail_early_or_skip_empty_sources(tmp_path):
    rng = random.Random(1)
    with pytest.raises(ValueError, match='General quota is negative'):
        prepare_sft_v2.load_general(-1, rng, tmp_path / 'missing')
    with pytest.raises(ValueError, match='OPC quota is negative'):
        prepare_sft_v2.load_code(-1, rng, tmp_path / 'missing')
    assert prepare_sft_v2.load_code(0, rng, tmp_path / 'missing') == []
    with pytest.raises(ValueError, match='No nonempty OPC'):
        prepare_sft_v2.load_code(1, rng, tmp_path)
    with pytest.raises(ValueError, match='UltraData quota'):
        prepare_sft_v3.load_ultradata(-1, rng, tmp_path / 'missing')
    assert prepare_sft_v3.load_ultradata(0, rng, tmp_path / 'missing') == []


def test_training_config_resolves_portable_manifest_path(tmp_path, monkeypatch):
    from qat.train import load_config

    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'data_manifest': './data/mix/manifest.json'}), encoding='utf-8')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, 'argv', ['train.py', '--config', str(config)])
    parsed = load_config()
    assert parsed['data_manifest'] == str(tmp_path / 'data/mix/manifest.json')
    assert parsed['quant_group'] is None
