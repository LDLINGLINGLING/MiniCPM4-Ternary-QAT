"""Offline tests for data_pipeline/download; no real download is performed."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest
import requests

DOWNLOAD_DIR = Path(__file__).resolve().parents[1] / 'data_pipeline' / 'download'
sys.path.insert(0, str(DOWNLOAD_DIR))

import download_alpaca as alpaca
import download_eval_data as eval_data
import download_minicpm as minicpm
import download_modelscope_files as ms_files
import download_modelscope_ufw as ms_ufw
import download_ranges as ranges
import download_ultrafineweb as ufw

MODULES = (minicpm, ufw, ms_ufw, ms_files, ranges, alpaca, eval_data)


def fail_network(*args, **kwargs):
    raise AssertionError('Network access is forbidden in offline downloader tests')


def test_import_has_no_cli_network_or_environment_side_effects(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['pytest', '--not-a-downloader-option'])
    monkeypatch.setattr(requests, 'Session', fail_network)
    monkeypatch.setattr(requests, 'get', fail_network)
    names = (
        'HF_ENDPOINT', 'HF_HUB_DISABLE_IMPLICIT_TOKEN', 'HF_HUB_DISABLE_XET',
        'HF_HUB_ENABLE_HF_TRANSFER', 'HF_HUB_DOWNLOAD_TIMEOUT', 'HF_HUB_ETAG_TIMEOUT',
    )
    before = {name: os.environ.get(name) for name in names}
    for module in MODULES:
        importlib.reload(module)
    assert {name: os.environ.get(name) for name in names} == before


@pytest.mark.parametrize('module', MODULES, ids=lambda module: module.__name__)
def test_cli_help_is_offline(module, monkeypatch, capsys):
    monkeypatch.setattr(requests, 'Session', fail_network)
    monkeypatch.setattr(requests, 'get', fail_network)
    with pytest.raises(SystemExit) as caught:
        module.main(['--help'])
    assert caught.value.code == 0
    help_text = capsys.readouterr().out
    assert '--output' in help_text
    assert 'usage:' in help_text


def test_minicpm_manifest_hash_and_resume_name(tmp_path):
    payload = {'repo': 'owner/model', 'revision': 'abc', 'files': []}
    minicpm.save_manifest(tmp_path, payload)
    payload['files'].append({'path': 'weights.bin', 'status': 'verified'})
    minicpm.save_manifest(tmp_path, payload)
    assert json.loads((tmp_path / 'download-manifest.json').read_text(encoding='utf-8')) == payload
    assert not (tmp_path / 'download-manifest.json.tmp').exists()

    content = b'synthetic downloader payload'
    path = tmp_path / 'weights.bin.partial'
    path.write_bytes(content)
    assert minicpm.sha256_file(path) == hashlib.sha256(content).hexdigest()


@pytest.mark.parametrize('module', (minicpm, ms_ufw, ms_files, ranges))
def test_remote_paths_cannot_escape_output(module, tmp_path):
    assert module.safe_target(tmp_path, 'nested/file.bin') == tmp_path / 'nested' / 'file.bin'
    for unsafe in ('../escape.bin', '/absolute.bin'):
        with pytest.raises(ValueError, match='Unsafe filename'):
            module.safe_target(tmp_path, unsafe)


def test_ultrafineweb_lfs_hash_normalization():
    digest = 'a' * 64
    assert ufw.normalize_lfs_sha256(f'sha256:{digest}') == digest
    assert ufw.normalize_lfs_sha256(digest) == digest
    assert ufw.normalize_lfs_sha256(None) is None


def test_modelscope_ufw_reuses_plan_and_applies_limit_without_network(tmp_path):
    saved = {
        'source': ms_ufw.API,
        'revision': 'master',
        'seed': 17,
        'per_language': 2,
        'queue': [{'path': f'data/shard-{index}.parquet'} for index in range(4)],
    }
    (tmp_path / 'plan.json').write_text(json.dumps(saved), encoding='utf-8')
    assert ms_ufw.plan(None, tmp_path, 17, 2) == saved
    assert ms_ufw.apply_limit(saved['queue'], 0) == saved['queue']
    assert ms_ufw.apply_limit(saved['queue'], 2) == saved['queue'][:2]
    with pytest.raises(ValueError, match='limit'):
        ms_ufw.apply_limit(saved['queue'], -1)
    with pytest.raises(ValueError, match='different seed'):
        ms_ufw.plan(None, tmp_path, 18, 2)


def plan_args(**overrides):
    values = {
        'repo': 'owner/dataset',
        'root': ['data/train'],
        'seed': 23,
        'per_root': 1,
        'suffix': '.jsonl',
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_modelscope_files_reuses_matching_plan_without_listing(tmp_path, monkeypatch):
    args = plan_args()
    saved = {
        'source': ms_files.API,
        'repo': args.repo,
        'revision': 'master',
        'roots': args.root,
        'suffix': args.suffix,
        'seed': args.seed,
        'per_root': args.per_root,
        'queue': [],
    }
    (tmp_path / 'plan.json').write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(ms_files, 'list_files', fail_network)
    assert ms_files.make_plan(tmp_path, args) == saved
    with pytest.raises(ValueError, match='differs'):
        ms_files.make_plan(tmp_path, plan_args(suffix='.parquet'))


def test_modelscope_files_validates_synthetic_jsonl(tmp_path):
    valid = tmp_path / 'shard.jsonl.partial'
    valid.write_text('{"text": "synthetic text", "id": 1}\n', encoding='utf-8')
    assert 'jsonl keys=' in ms_files.validate(valid)

    invalid = tmp_path / 'invalid.jsonl.partial'
    invalid.write_text('{"text": "  "}\n', encoding='utf-8')
    with pytest.raises(ValueError, match='supported schema'):
        ms_files.validate(invalid)


@pytest.mark.parametrize('record', [
    {'kind': 'OpenQA', 'input': 'question', 'target': 'answer'},
    {'question': 'question', 'chatgpt_answers': ['answer']},
    {'instruction': 'instruction', 'input': '', 'output': 'answer'},
])
def test_modelscope_files_accepts_sft_jsonl_schemas(tmp_path, record):
    path = tmp_path / 'sft.jsonl.partial'
    path.write_text(json.dumps(record) + '\n', encoding='utf-8')
    assert 'jsonl keys=' in ms_files.validate(path)


def test_modelscope_files_validates_instruction_csv(tmp_path):
    path = tmp_path / 'alpaca.csv.partial'
    path.write_text('instruction,input,output\nquestion,,answer\n', encoding='utf-8')
    assert 'csv columns=' in ms_files.validate(path)
    path.write_text('question,answer\nquestion,answer\n', encoding='utf-8')
    with pytest.raises(ValueError, match='instruction/output'):
        ms_files.validate(path)


def test_ranges_limit_and_manifest_use_fake_fetch(tmp_path, monkeypatch):
    records = [
        {'language': 'zh', 'path': f'data/shard-{index}.parquet', 'expected_bytes': 1}
        for index in range(3)
    ]
    original = {
        'repo': 'openbmb/Ultra-FineWeb',
        'revision': 'resolved-commit',
        'endpoint': ranges.DEFAULT_ENDPOINT,
        'selection': 'synthetic test selection',
        'files': records,
    }
    (tmp_path / 'download-manifest.json').write_text(json.dumps(original), encoding='utf-8')
    calls = []

    def fake_safe_fetch(record, out, repo, revision, endpoint):
        calls.append(record['path'])
        return {'path': record['path'], 'status': 'verified'}

    monkeypatch.setattr(ranges, 'safe_fetch', fake_safe_fetch)
    assert ranges.main(['--output', str(tmp_path), '--workers', '1', '--limit', '2']) == 0
    assert calls == [record['path'] for record in records[:2]]
    result = json.loads((tmp_path / 'range-download-manifest.json').read_text(encoding='utf-8'))
    assert result['repo'] == original['repo']
    assert result['revision'] == original['revision']
    assert result['files'] == [{'path': record['path'], 'status': 'verified'} for record in records[:2]]
    assert not (tmp_path / 'range-download-manifest.json.tmp').exists()


def test_ranges_empty_manifest_never_opens_a_session(tmp_path, monkeypatch):
    original = {
        'repo': 'openbmb/Ultra-FineWeb',
        'revision': 'resolved-commit',
        'selection': 'empty synthetic test',
        'files': [],
    }
    (tmp_path / 'download-manifest.json').write_text(json.dumps(original), encoding='utf-8')
    monkeypatch.setattr(ranges.requests, 'Session', fail_network)
    assert ranges.main(['--output', str(tmp_path), '--workers', '1']) == 0
    result = json.loads((tmp_path / 'range-download-manifest.json').read_text(encoding='utf-8'))
    assert result['files'] == []


def test_alpaca_json_to_csv_and_manifest_without_network(tmp_path, monkeypatch):
    records = [
        {'instruction': 'Translate "hello"', 'input': '', 'output': '你好'},
        {'instruction': 'Add', 'input': '1\n2', 'output': '3'},
    ]

    def fake_fetch(url, target, timeout):
        Path(target).write_text(json.dumps(records, ensure_ascii=False), encoding='utf-8')

    monkeypatch.setattr(alpaca, 'fetch', fake_fetch)
    assert alpaca.main(['--output', str(tmp_path), '--languages', 'en']) == 0
    lines = (tmp_path / 'alpaca_en.csv').read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'instruction,input,output'
    manifest = json.loads((tmp_path / 'download-manifest.json').read_text(encoding='utf-8'))
    assert manifest['files'][0]['records'] == 2
    assert len(manifest['files'][0]['csv']['sha256']) == 64


def test_alpaca_rejects_invalid_records(tmp_path):
    raw = tmp_path / 'bad.json'
    raw.write_text('[{"instruction": "missing output"}]', encoding='utf-8')
    with pytest.raises(ValueError, match='lacks output'):
        alpaca.validate_records(raw)


def test_eval_zip_materializes_only_mapped_csvs(tmp_path):
    archive = tmp_path / 'data.zip'
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'subject': {'name': 'Subject', 'category': 'Other'}}), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('nested/dev/subject.csv', 'Question,A,B,C,D,Answer\nQ,a,b,c,d,A\n')
        handle.writestr('nested/test/subject.csv', 'Question,A,B,C,D,Answer\nQ2,a,b,c,d,D\n')
        handle.writestr('nested/test/ignored.csv', 'not,a,mapped,file,with,invalid,data\n')
    output = tmp_path / 'output'
    records = eval_data.materialize_archive('cmmlu', archive, mapping, output)
    assert len(records) == 2
    assert (output / 'dev/subject.csv').is_file()
    assert not (output / 'test/ignored.csv').exists()
    repeated = eval_data.materialize_archive('cmmlu', archive, mapping, output)
    assert {record['status'] for record in repeated} == {'existing'}


def test_eval_rejects_traversal_and_missing_required_files(tmp_path):
    with pytest.raises(ValueError, match='Unsafe archive member'):
        eval_data.safe_member_name('../escape.csv')
    archive = tmp_path / 'incomplete.zip'
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'subject': {}}), encoding='utf-8')
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('dev/subject.csv', 'Question,A,B,C,D,Answer\nQ,a,b,c,d,A\n')
    with pytest.raises(ValueError, match='Archive lacks 1 required CSVs'):
        eval_data.materialize_archive('cmmlu', archive, mapping, tmp_path / 'out')


def test_eval_rejects_duplicate_mapped_members(tmp_path):
    archive = tmp_path / 'duplicate.zip'
    mapping = tmp_path / 'mapping.json'
    mapping.write_text(json.dumps({'subject': {}}), encoding='utf-8')
    row = 'Question,A,B,C,D,Answer\nQ,a,b,c,d,A\n'
    with zipfile.ZipFile(archive, 'w') as handle:
        handle.writestr('one/dev/subject.csv', row)
        handle.writestr('two/dev/subject.csv', row)
        handle.writestr('test/subject.csv', row)
    with pytest.raises(ValueError, match='Duplicate archive member'):
        eval_data.materialize_archive('cmmlu', archive, mapping, tmp_path / 'out')


def test_mmlu_and_gsm8k_schema_validation(tmp_path):
    assert eval_data.validate_mc_csv(b'Q,a,b,c,d,B\n', 'mmlu', 'fixture') == 1
    with pytest.raises(ValueError, match='Invalid question'):
        eval_data.validate_mc_csv(b'Q,a,b,c,d,E\n', 'mmlu', 'fixture')
    gsm = tmp_path / 'train.jsonl'
    gsm.write_text(json.dumps({'question': '1+1?', 'answer': '#### 2'}) + '\n', encoding='utf-8')
    assert eval_data.validate_gsm8k(gsm) == 1
