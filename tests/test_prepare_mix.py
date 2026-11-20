"""Synthetic/no-network tests; run python -I -m pytest tests/test_prepare_mix.py."""
from array import array
import json
from pathlib import Path
import sqlite3
import struct
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.pretrain import decontam as dc
from data_pipeline.pretrain import prepare_mix as pm


def info(index=0):
    return {'benchmark': 'synthetic', 'split': 'test', 'sample_id': str(index)}


def fake_encode(texts):
    return [struct.pack('<' + 'I' * (len(text) + 2), 1, *map(ord, text), 2) for text in texts]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))


def synthetic(tmp_path, parquet=False):
    root = tmp_path / 'project'
    root.mkdir()
    tokenizer = root / 'model'
    tokenizer.mkdir()
    (tokenizer / 'tokenizer.json').write_text('{}')
    benchmark = root / 'bench.jsonl'
    write_jsonl(benchmark, [{'question': 'What is the forbidden benchmark question?'}])
    plans = []
    for domain in pm.DOMAINS:
        directory = root / domain
        directory.mkdir()
        # Guaranteed fixed 1% split without making the test scan thousands of docs.
        rows = ['What is the forbidden benchmark question?', '', 'DUPLICATE SHARED DOCUMENT']
        for split in ('train', 'val'):
            candidates = (f'{domain} independent original document {index}' for index in range(10000))
            rows.extend(text for _, text in zip(range(12),
                        (t for t in candidates if pm.split_for(pm.document_hash(t)) == split)))
        if parquet:
            import pyarrow as pa
            import pyarrow.parquet as pq
            path = directory / 'shard.parquet'
            pq.write_table(pa.table({'content': rows + [None]}), path, row_group_size=4)
        else:
            path = directory / 'shard.jsonl'
            write_jsonl(path, [{'text': row} for row in rows] + [{'other': 1}, {'text': None}])
        spec = pm.fingerprint(path)
        item = {'path': path.name, 'sha256': spec['sha256'], 'size': spec['size']}
        if domain.startswith('web_'):
            item['lang'] = domain[4:]
            repo = 'test/web'
        else:
            repo = f'test/{domain}'
        plan = directory / 'plan.json'
        plan.write_text(json.dumps({'repo': repo, 'queue': [item]}))
        plans.append(plan)
    cfg = pm.config()
    cfg.update(train_caps={d: 150 for d in pm.DOMAINS}, val_windows=2, window_tokens=20,
               batch_docs=8, max_doc_bytes=1024, batch_bytes=4096, workers=2)
    return root, tokenizer, plans, [{'path': str(benchmark), 'benchmark': 'synthetic',
                                   'split': 'test', 'format': 'jsonl'}], cfg


def frozen(tmp_path, parquet=False):
    root, tokenizer, plans, benchmarks, cfg = synthetic(tmp_path, parquet)
    output = tmp_path / 'output'
    pm.freeze(root, output, tokenizer, cfg, plans, benchmarks)
    return output


@pytest.mark.parametrize('native', [False, True])
def test_normalized_stems_and_units(native):
    records = [('How MANY apples cost １２ dollars?', info()), ('猫有几只？', info(1)),
               ('one two three four five six seven eight nine ten eleven twelve thirteen fourteen', info(2)),
               ('甲乙丙丁戊己庚辛壬癸子丑寅卯', info(3))]
    index = dc.ExclusionIndex(records, require_fast=native)
    if not native:
        index.automaton = None
    assert index.match('prefix HOW many\n apples cost 12 dollars? suffix')['position'] == 7
    assert index.match('正文猫有几只?续文')['sample_id'] == '1'
    assert index.match('one,two,three,four,five,six,seven,eight,nine,ten,eleven,twelve,thirteen changed')['rule'] == '13_unit'
    assert index.match('one two three four five six seven eight nine ten eleven twelve changed') is None
    assert index.match('甲 乙 丙 丁 戊 己 庚 辛 壬 癸 子 丑 寅 改')['rule'] == '13_unit'
    assert index.match('A B C D 12') is None
    assert dc.units('ＡｂＣ １２.34 中文') == ['abc', '12', '34', '中', '文']


def test_answerless_ceval_and_mmlu_cmmlu(tmp_path):
    specs = []
    data = {'ceval': 'id,question,A,B,C,D\n0,无标签问题？,苹果,香蕉,树,花\n',
            'cmmlu': ',Question,A,B,C,D,Answer\n0,中文题干？,一,二,三,四,B\n',
            'mmlu': 'What is four?,one,two,three,four,D\n'}
    for benchmark, text in data.items():
        path = tmp_path / f'{benchmark}.csv'
        path.write_text(text)
        specs.append({'path': str(path), 'benchmark': benchmark, 'split': 'test', 'format': 'csv'})
    index = dc.ExclusionIndex(dc.questions(specs))
    assert index.sample_count == 3
    assert index.match('prefix无标签问题?')['benchmark'] == 'ceval'
    assert index.match('香蕉 苹果 B') is None


def test_discovery_all_splits_and_variants(tmp_path):
    base = tmp_path / 'eval_data'
    for split in ('train', 'test', 'train_socratic', 'test_socratic'):
        write_jsonl(base / 'gsm8k' / f'{split}.jsonl', [{'question': split}])
    write_jsonl(base / 'gsm8k_variants_v1' / 'numeric.jsonl', [{'question': 'variant'}])
    write_jsonl(base / 'gsm8k_variants_v1' / 'review_audit.jsonl', [{'answer': 'NOT A QUESTION'}])
    for benchmark, splits in [('mmlu/data', ('dev', 'val', 'test')), ('cmmlu', ('dev', 'test')),
                              ('ceval', ('dev', 'val', 'test'))]:
        for split in splits:
            path = base / benchmark / split / 'sample.csv'
            path.parent.mkdir(parents=True)
            path.write_text('question,A,B,C,D\nQuestion?,1,2,3,4\n')
    specs = dc.discover_benchmarks(tmp_path)
    assert len(specs) == 13
    assert all('review_audit' not in s['path'] for s in specs)
    (base / 'ceval/test/sample.csv').unlink()
    with pytest.raises(ValueError, match='ceval test'):
        dc.discover_benchmarks(tmp_path)


def test_freeze_completed_only_and_sha(tmp_path):
    root, tokenizer, plans, benchmarks, cfg = synthetic(tmp_path)
    plan = json.loads(plans[0].read_text())
    for name in ('unplanned.jsonl', 'missing.jsonl.partial'):
        (plans[0].parent / name).write_text('junk')
    plan['queue'] += [{'path': 'missing.jsonl', 'size': 10, 'sha256': '0' * 64},
                      {'path': 'missing.jsonl.partial', 'size': 4, 'sha256': '0' * 64}]
    plans[0].write_text(json.dumps(plan))
    snap = pm.freeze(root, tmp_path / 'out', tokenizer, cfg, plans, benchmarks)
    assert len(snap['sources']) == 4
    assert len(snap['excluded_at_freeze']) == 2
    assert snap['uncovered_benchmarks'] == ['MATH', 'HumanEval', 'MBPP']
    plan['queue'][0]['sha256'] = '0' * 64
    plans[0].write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='SHA256'):
        pm.freeze(root, tmp_path / 'bad', tokenizer, cfg, plans, benchmarks)


@pytest.mark.parametrize('parquet', [False, True])
def test_streams_dedup_split_counts_and_ready(tmp_path, parquet):
    output = frozen(tmp_path, parquet)
    manifest = pm.run(output, encoder=fake_encode)
    assert (output / 'READY').exists()
    assert manifest['version'] == 1
    assert manifest['statistics']['counts']['duplicates'] >= 3
    assert manifest['statistics']['counts']['contaminated'] == 4
    global_documents = set()
    for domain, stream in manifest['streams'].items():
        for split, entries in [('train', stream['train']), ('val', [stream['val']])]:
            for entry in entries:
                path = output / entry['path']
                assert pm.sha256_file(path) == entry['sha256']
                assert path.stat().st_size == entry['tokens'] * 4
                values = list(struct.unpack('<' + 'I' * entry['tokens'], path.read_bytes()))
                while values:
                    assert values.pop(0) == 1
                    end = values.index(2)
                    text = ''.join(map(chr, values[:end]))
                    del values[:end + 1]
                    assert 'forbidden benchmark' not in text
                    assert pm.split_for(pm.document_hash(text)) == split
                    assert text not in global_documents
                    global_documents.add(text)
    assert pm.run(output, encoder=fake_encode) == manifest
    with sqlite3.connect(output / 'staging/state.sqlite') as db:
        hits = db.execute('SELECT details FROM hits').fetchall()
    assert len(hits) == 4
    assert 'position' in json.loads(hits[0][0])


@pytest.mark.parametrize('parquet', [False, True])
def test_resume_byte_identical_and_repair_tails(tmp_path, parquet):
    first = frozen(tmp_path, parquet)
    second = tmp_path / 'second'
    second.mkdir()
    (second / 'freeze.json').write_bytes((first / 'freeze.json').read_bytes())
    expected = pm.run(first, encoder=fake_encode)
    paused = pm.run(second, max_batches=2, encoder=fake_encode)
    assert paused['paused'] and not (second / 'READY').exists()
    for path in (second / 'staging').glob('*.bin'):
        with path.open('ab') as handle:
            handle.write(b'uncommitted crash tail')
    actual = pm.run(second, encoder=fake_encode)
    assert actual == expected
    assert (first / 'manifest.json').read_bytes() == (second / 'manifest.json').read_bytes()


def test_transaction_failure_replay(tmp_path):
    output = frozen(tmp_path)
    with pytest.raises(RuntimeError, match='injected'):
        pm.run(output, encoder=lambda _: (_ for _ in ()).throw(RuntimeError('injected')))
    with sqlite3.connect(output / 'staging/state.sqlite') as db:
        assert db.execute('SELECT count(*) FROM seen').fetchone()[0] == 0
        # The first batch contains only rejected/empty records and committed before
        # the first encoder invocation. The failed second batch must not commit.
        assert json.loads(db.execute('SELECT payload FROM state').fetchone()[0])['batches'] == 1
    assert pm.run(output, encoder=fake_encode)['version'] == 1


def test_partial_publication_recovery(tmp_path, monkeypatch):
    output = frozen(tmp_path)
    real_replace = pm.os.replace
    counter = 0
    def fail(src, dst):
        nonlocal counter
        if str(dst).endswith('.bin'):
            counter += 1
            if counter == 3:
                raise OSError('publish interruption')
        return real_replace(src, dst)
    with monkeypatch.context() as context:
        context.setattr(pm.os, 'replace', fail)
        with pytest.raises(OSError, match='publish interruption'):
            pm.run(output, encoder=fake_encode)
    assert not (output / 'READY').exists()
    assert pm.run(output, encoder=fake_encode)['version'] == 1


def test_insufficient_and_identity_rejected(tmp_path):
    root, tokenizer, plans, benchmarks, cfg = synthetic(tmp_path)
    cfg['train_caps'] = {d: 10**9 for d in pm.DOMAINS}
    output = tmp_path / 'out'
    pm.freeze(root, output, tokenizer, cfg, plans, benchmarks)
    with pytest.raises(RuntimeError, match='exhausted'):
        pm.run(output, encoder=fake_encode)
    assert not (output / 'READY').exists()
    assert (output / 'INSUFFICIENT.json').exists()
    snapshot = json.loads((output / 'freeze.json').read_text())
    snapshot['config']['seed'] += 1
    pm.atomic_json(output / 'freeze.json', snapshot)
    with pytest.raises(ValueError, match='identity'):
        pm.run(output, encoder=fake_encode)


def test_jsonl_bounded_bad_empty_oversize_and_cursor(tmp_path):
    path = tmp_path / 'rows.jsonl'
    path.write_bytes(b'bad\n\n' + b'x' * 3000 + b'\n' + b'{"text":"hello"}\n' + b'{"text":"last"}')
    cfg = pm.config()
    cfg['max_doc_bytes'] = 100
    spec = {'path': str(path), 'format': 'jsonl', 'field': 'text'}
    records = list(pm.iter_records(spec, 0, cfg))
    assert [r[2] for r in records] == ['invalid', 'invalid', 'oversize', 'ok', 'ok']
    assert records[-1][1] == 'last'
    assert list(pm.iter_records(spec, records[2][0], cfg)) == records[3:]


def test_actual_tokenizer_encoding_without_network(monkeypatch):
    class Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs == {'add_special_tokens': False, 'truncation': False}
            return {'input_ids': [[8, 9] for _ in texts]}
    monkeypatch.setattr(pm, '_worker_tokenizer', Tokenizer())
    assert pm.encode_texts(['input']) == [struct.pack('<4I', 1, 8, 9, 2)]


def test_spawn_local_tokenizer_no_network(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast
    tokenizer = Tokenizer(WordLevel({'[UNK]': 0, '<s>': 1, '</s>': 2, 'hello': 3}, unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token='[UNK]', bos_token='<s>', eos_token='</s>')
    fast.save_pretrained(tmp_path)
    with pm.ProcessPoolExecutor(max_workers=2, mp_context=pm.mp.get_context('spawn'),
                               initializer=pm.init_tokenizer, initargs=(str(tmp_path),)) as pool:
        outputs = list(pool.map(pm.encode_texts, [['hello'], ['hello hello']]))
    assert outputs == [[struct.pack('<3I', 1, 3, 2)], [struct.pack('<4I', 1, 3, 3, 2)]]
