"""Selection must use complete, matching final development results only."""
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from qat import run_mix as mix


def write_runs(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, 'ROOT', tmp_path)
    data = tmp_path / 'data/mix_v1'
    data.mkdir(parents=True)
    (data / 'manifest.json').write_text('{}')
    for i, (name, mode, _) in enumerate(mix.SPECS):
        run = tmp_path / 'runs' / name
        (run / 'checkpoints/final').mkdir(parents=True)
        (run / 'checkpoints/final/state.pt').write_bytes(b'fixture')
        (run / 'config.json').write_text(json.dumps(mix.configs(data)[name]))
        key = 'val_ternary' if mode == 'qat' else 'val_float'
        records = [{'tokens': mix.TOTAL, key: {d: 3.0 - i * 0.1 for d in mix.DOMAINS}},
                   {'event': 'done'}]
        (run / 'metrics.jsonl').write_text('\n'.join(json.dumps(r) for r in records))
    return data


def test_code_identity_includes_training_and_data_pipeline(tmp_path, monkeypatch):
    monkeypatch.setattr(mix, 'ROOT', tmp_path)
    (tmp_path / 'qat').mkdir()
    (tmp_path / 'data_pipeline').mkdir()
    (tmp_path / 'qat/train.py').write_text('train = True\n', encoding='utf-8')
    (tmp_path / 'data_pipeline/sampling.py').write_text('sample = True\n', encoding='utf-8')
    (tmp_path / 'qat/__pycache__').mkdir()
    (tmp_path / 'qat/__pycache__/ignored.py').write_text('ignored = True\n', encoding='utf-8')
    identity = mix.code_identity()
    assert set(identity) == {'qat/train.py', 'data_pipeline/sampling.py'}
    assert all(len(digest) == 64 for digest in identity.values())


def test_final_dev_selects_qat_not_better_bf16(tmp_path, monkeypatch):
    data = write_runs(tmp_path, monkeypatch)
    result = mix.summarize(data)
    assert result['selected']['run'] == mix.SPECS[2][0]
    assert len(result['candidates']) == 3


def test_missing_done_rejected(tmp_path, monkeypatch):
    write_runs(tmp_path, monkeypatch)
    run = tmp_path / 'runs' / mix.SPECS[0][0]
    path = run / 'metrics.jsonl'
    path.write_text(path.read_text().splitlines()[0] + '\n')
    with pytest.raises(ValueError, match='completion'):
        mix.final_record(run)


def test_nonfinite_or_missing_domain_rejected(tmp_path, monkeypatch):
    write_runs(tmp_path, monkeypatch)
    run = tmp_path / 'runs' / mix.SPECS[0][0]
    path = run / 'metrics.jsonl'
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['val_ternary']['math'] = float('nan')
    path.write_text('\n'.join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match='nonfinite'):
        mix.final_record(run)


def test_config_change_rejected(tmp_path, monkeypatch):
    data = write_runs(tmp_path, monkeypatch)
    path = tmp_path / 'runs' / mix.SPECS[0][0] / 'config.json'
    cfg = json.loads(path.read_text())
    cfg['lr'] = 1e-2
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match='mismatch'):
        mix.summarize(data)


def test_tie_preserves_declared_order(tmp_path, monkeypatch):
    data = write_runs(tmp_path, monkeypatch)
    for name, mode, _ in mix.SPECS[:3]:
        path = tmp_path / 'runs' / name / 'metrics.jsonl'
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        rows[0]['val_ternary'] = {d: 3.0 for d in mix.DOMAINS}
        path.write_text('\n'.join(json.dumps(r) for r in rows))
    assert mix.summarize(data)['selected']['run'] == mix.SPECS[0][0]
