"""Download GSM8K, MMLU, CMMLU and C-Eval into qat/eval_downstream.py's layout.

Archives stay in a separate work directory. Only the CSVs named by the bundled
mapping files are read in memory; no archive member is extracted to a member-
controlled path. Existing different outputs are never overwritten.
"""

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import zipfile

import requests


ROOT = Path(__file__).resolve().parents[2]
ARCHIVES = {
    'mmlu': 'https://people.eecs.berkeley.edu/~hendrycks/data.tar',
    'cmmlu': 'https://raw.githubusercontent.com/haonan-li/CMMLU/master/data.zip',
    'ceval': 'https://raw.githubusercontent.com/hkust-nlp/ceval/main/data.zip',
}
GSM8K = {
    name: f'https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/{name}.jsonl'
    for name in ('train', 'test', 'socratic_train', 'socratic_test')
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--tasks', nargs='+', choices=['gsm8k', *sorted(ARCHIVES)], default=['gsm8k', *sorted(ARCHIVES)])
    parser.add_argument('--work-dir', type=Path, required=True,
                        help='Separate directory for downloaded archives and raw JSONL.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Evaluation root, normally <repository>/eval_data.')
    parser.add_argument('--mapping-root', type=Path, default=ROOT / 'configs/eval',
                        help='Directory containing <task>/mapping.json.')
    parser.add_argument('--timeout', type=int, default=120, help='Per-response read timeout in seconds.')
    return parser


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def fetch(url, target, timeout):
    target = Path(target)
    if target.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    for attempt in range(6):
        try:
            with requests.get(url, stream=True, timeout=(20, timeout)) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(8 * 1024 * 1024):
                        if block:
                            handle.write(block)
            partial.replace(target)
            return
        except Exception:
            if attempt == 5:
                raise
            time.sleep(min(2 ** attempt, 20))


def install_bytes(target, payload):
    target = Path(target)
    digest = sha256_bytes(payload)
    if target.exists():
        if sha256_bytes(target.read_bytes()) != digest:
            raise FileExistsError(f'Existing output differs from downloaded source: {target}')
        return {'path': target.name, 'bytes': len(payload), 'sha256': digest, 'status': 'existing'}
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.tmp')
    temporary.write_bytes(payload)
    temporary.replace(target)
    return {'path': target.name, 'bytes': len(payload), 'sha256': digest, 'status': 'written'}


def validate_mc_csv(payload, task, member):
    if len(payload) > 32 * 1024 * 1024:
        raise ValueError(f'Unexpectedly large evaluation CSV: {member}')
    try:
        rows = list(csv.reader(io.StringIO(payload.decode('utf-8-sig'))))
    except (UnicodeError, csv.Error) as exc:
        raise ValueError(f'Invalid CSV {member}: {exc}') from exc
    if not rows:
        raise ValueError(f'Empty CSV: {member}')
    header = task != 'mmlu'
    if header:
        columns = [column.strip().lower() for column in rows[0]]
        if 'question' not in columns or 'answer' not in columns:
            raise ValueError(f'Missing question/answer columns: {member}')
        rows = rows[1:]
    if not rows:
        raise ValueError(f'CSV has no questions: {member}')
    for row in rows:
        if len(row) < 6 or not row[0].strip() or row[5].strip() not in 'ABCD':
            raise ValueError(f'Invalid question/options/answer row in {member}: {row}')
    return len(rows)


def safe_member_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts:
        raise ValueError(f'Unsafe archive member: {name}')
    return path


def archive_members(archive):
    archive = Path(archive)
    if archive.suffix == '.zip':
        with zipfile.ZipFile(archive) as handle:
            for info in handle.infolist():
                if info.is_dir():
                    continue
                path = safe_member_name(info.filename)
                if info.file_size > 32 * 1024 * 1024:
                    raise ValueError(f'Unexpectedly large archive member: {info.filename}')
                yield path, info.file_size, lambda info=info: handle.read(info)
    else:
        with tarfile.open(archive, mode='r:*') as handle:
            for info in handle.getmembers():
                if not info.isfile():
                    continue
                path = safe_member_name(info.name)
                if info.size > 32 * 1024 * 1024:
                    raise ValueError(f'Unexpectedly large archive member: {info.name}')
                yield path, info.size, lambda info=info: handle.extractfile(info).read()


def materialize_archive(task, archive, mapping_path, output):
    mapping = json.loads(Path(mapping_path).read_text(encoding='utf-8'))
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f'Empty subject mapping: {mapping_path}')
    split = 'val' if task == 'ceval' else 'test'
    expected = {}
    for subject in mapping:
        base = f'{subject}.csv' if task == 'cmmlu' else f'{subject}_{{split}}.csv'
        for part in ('dev', split):
            filename = base.format(split=part)
            expected[f'{part}/{filename}'] = output / part / filename
    pending = dict(expected)
    records = []
    for member, size, read in archive_members(archive):
        if len(member.parts) < 2:
            continue
        key = '/'.join(member.parts[-2:])
        if key not in pending:
            continue
        if pending[key] is None:
            raise ValueError(f'Duplicate archive member for {key}')
        payload = read()
        if len(payload) != size:
            raise ValueError(f'Short archive member: {member}')
        count = validate_mc_csv(payload, task, str(member))
        record = install_bytes(pending[key], payload)
        record.update({'split': key.split('/')[0], 'questions': count, 'archive_member': str(member)})
        records.append(record)
        pending[key] = None
    missing = sorted(key for key, target in pending.items() if target is not None)
    if missing:
        raise ValueError(f'Archive lacks {len(missing)} required CSVs; first: {missing[:5]}')
    return records


def validate_gsm8k(path):
    count = 0
    with Path(path).open(encoding='utf-8') as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record.get('question'), str) or not record['question'].strip():
                raise ValueError(f'{path}:{line_number}: missing question')
            if not isinstance(record.get('answer'), str) or not record['answer'].strip():
                raise ValueError(f'{path}:{line_number}: missing answer')
            count += 1
    if not count:
        raise ValueError(f'No GSM8K records in {path}')
    return count


def prepare_gsm8k(work, output, timeout):
    records = []
    for name, url in GSM8K.items():
        raw = work / f'{name}.jsonl'
        fetch(url, raw, timeout)
        count = validate_gsm8k(raw)
        record = install_bytes(output / f'{name}.jsonl', raw.read_bytes())
        record.update({'source': url, 'records': count})
        records.append(record)
    return records


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.timeout < 1:
        raise ValueError('--timeout must be positive')
    work_root = args.work_dir.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    mapping_root = args.mapping_root.expanduser().resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    for task in args.tasks:
        work = work_root / task
        output = output_root / task
        work.mkdir(parents=True, exist_ok=True)
        if task == 'gsm8k':
            source = dict(GSM8K)
            files = prepare_gsm8k(work, output, args.timeout)
        else:
            url = ARCHIVES[task]
            archive = work / ('data.zip' if task != 'mmlu' else 'data.tar')
            fetch(url, archive, args.timeout)
            mapping = mapping_root / task / 'mapping.json'
            files = materialize_archive(task, archive, mapping, output)
            source = {'archive': url, 'archive_sha256': sha256_bytes(archive.read_bytes())}
            mapping_record = install_bytes(output / 'mapping.json', mapping.read_bytes())
            mapping_record['path'] = 'mapping.json'
            files.append(mapping_record)
        atomic_json(work / 'manifest.json', {'task': task, 'source': source, 'files': files})
        print(f'VERIFIED {task}: {len(files)} files -> {output}', flush=True)
    print(f'FINISHED SUCCESS: {len(args.tasks)} tasks', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
