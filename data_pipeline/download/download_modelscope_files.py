"""Download a seeded sample of files from ModelScope dataset directories.

For each --root directory (non-recursive), list files with the given suffix,
shuffle with --seed, and take --per-root files; the selection is written to
<output>/plan.json before downloading and reused unchanged on restart.
Each file is fetched with resumable HTTP Range requests, checked against
ModelScope's size and SHA256, validated (parquet, supported JSONL records, or
instruction CSV), then renamed into place.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import time

import requests

API = 'https://www.modelscope.cn/api/v1/datasets'
CHUNK = 16 * 1024 * 1024


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError('must be greater than zero')
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError('must be zero or greater')
    return parsed


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--repo', required=True, help='ModelScope dataset ID, for example OWNER/DATASET.')
    parser.add_argument(
        '--root', action='append', required=True,
        help='Dataset directory to list non-recursively; repeat for multiple roots.',
    )
    parser.add_argument('--suffix', default='.parquet', help='Filename suffix to select (.parquet or JSONL suffix).')
    parser.add_argument('--per-root', type=nonnegative_int, default=0, help='Files selected per root; 0 means all.')
    parser.add_argument('--seed', type=int, default=20261109, help='Seed for deterministic per-root shuffling.')
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory for plan.json and files; use a durable path outside the repository.',
    )
    parser.add_argument('--workers', type=positive_int, default=2, help='Maximum concurrent file downloads.')
    return parser


def safe_target(out, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe filename {name!r}')
    return out.joinpath(*relative.parts)


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(CHUNK), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path, payload):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(payload, indent=1), encoding='utf-8')
    temporary.replace(path)


def get_json(url, params):
    for attempt in range(6):
        try:
            response = requests.get(url, params=params, timeout=(20, 60))
            response.raise_for_status()
            body = response.json()
            if body.get('Code') != 200:
                raise RuntimeError(f'ModelScope error: {body.get("Message")}')
            return body['Data']
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def list_files(repo, root, suffix):
    files, page = [], 1
    while True:
        data = get_json(
            f'{API}/{repo}/repo/tree',
            {'Revision': 'master', 'Root': root, 'PageNumber': page, 'PageSize': 500},
        )
        files += [
            {'repo': repo, 'path': item['Path'], 'size': item['Size'], 'sha256': item['Sha256']}
            for item in data['Files']
            if item['Type'] == 'blob' and item['Path'].endswith(suffix)
        ]
        if len(data['Files']) < 500:
            return sorted(files, key=lambda item: item['path'])
        page += 1


def make_plan(out, args):
    """Create or reuse a plan without changing an existing selection."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'plan.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        old_suffix = saved.get('suffix', args.suffix)
        if (saved['repo'], saved['roots'], saved['seed'], saved['per_root'], old_suffix) != (
                args.repo, args.root, args.seed, args.per_root, args.suffix):
            raise ValueError('Existing plan differs from these arguments; refusing to mix samples')
        return saved
    queue = []
    for root in args.root:
        files = list_files(args.repo, root, args.suffix)
        if not files:
            raise ValueError(f'No {args.suffix} files under {root}')
        random.Random(f'{args.seed}-{root}').shuffle(files)
        queue += files[:args.per_root or None]
        selected = min(len(files), args.per_root or len(files))
        print(f'LISTED {root}: {len(files)} files, selected {selected}', flush=True)
    saved = {
        'source': API,
        'repo': args.repo,
        'revision': 'master',
        'roots': args.root,
        'suffix': args.suffix,
        'seed': args.seed,
        'per_root': args.per_root,
        'selection': 'per-root seeded shuffle, first per_root files',
        'queue': queue,
    }
    atomic_write_json(path, saved)
    return saved


def validate(path):
    """Validate formats consumed by the pretraining and SFT preparation tools."""
    if path.name.endswith('.parquet.partial'):
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        names = parquet.schema_arrow.names
        if not {'text', 'content'} & set(names):
            raise ValueError(f'No text/content column: {names}')
        return f'rows={parquet.metadata.num_rows} columns={names}'
    if path.name.endswith('.csv.partial'):
        import csv

        with path.open(encoding='utf-8-sig', newline='') as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not {'instruction', 'output'} <= set(reader.fieldnames):
                raise ValueError(f'Instruction CSV lacks instruction/output columns: {reader.fieldnames}')
            record = next(reader, None)
        if not record or not record.get('instruction', '').strip() or not record.get('output', '').strip():
            raise ValueError('Instruction CSV has no nonempty first example')
        return f'csv columns={reader.fieldnames}'
    with path.open('rb') as handle:
        record = json.loads(handle.readline())
    if not isinstance(record, dict):
        raise ValueError('First JSONL record is not an object')
    text = record.get('text')
    if isinstance(text, str) and text.strip():
        return f'jsonl keys={sorted(record)}'
    is_firefly = all(isinstance(record.get(key), str) and record[key].strip()
                     for key in ('kind', 'input', 'target'))
    answers = record.get('chatgpt_answers') or record.get('human_answers')
    is_hc3 = isinstance(record.get('question'), str) and record['question'].strip() and bool(answers)
    is_instruction = all(isinstance(record.get(key), str) and record[key].strip()
                         for key in ('instruction', 'output'))
    if not (is_firefly or is_hc3 or is_instruction):
        raise ValueError(f'First JSONL record has no supported schema: {sorted(record)}')
    return f'jsonl keys={sorted(record)}'


def fetch(record, out):
    target = safe_target(out, record['path'])
    if target.exists():
        return 'exists'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.partial')
    session = requests.Session()
    params = {'Revision': 'master', 'FilePath': record['path']}
    total = record['size']
    start = partial.stat().st_size if partial.exists() else 0
    if start > total:
        raise ValueError(f'Partial larger than remote: {partial}')
    with partial.open('ab') as handle:
        while start < total:
            end = min(start + CHUNK, total) - 1
            for attempt in range(8):
                try:
                    with session.get(
                        f'{API}/{record["repo"]}/repo',
                        params=params,
                        headers={'Range': f'bytes={start}-{end}'},
                        stream=True,
                        timeout=(20, 120),
                    ) as response:
                        if response.status_code != 206:
                            raise RuntimeError(f'HTTP {response.status_code}')
                        data = response.raw.read(end - start + 2)
                    if len(data) != end - start + 1:
                        raise RuntimeError(f'Short chunk {len(data)}')
                    break
                except Exception as exc:
                    print(
                        f'RETRY {record["path"]} offset={start} attempt={attempt + 1} '
                        f'{type(exc).__name__}',
                        flush=True,
                    )
                    if attempt == 7:
                        raise
                    time.sleep(min(2 ** attempt, 60))
            handle.write(data)
            start += len(data)
    if partial.stat().st_size != total:
        raise ValueError('File size mismatch')
    if sha256_file(partial) != record['sha256']:
        bad = partial.with_name(target.name + '.sha_mismatch')
        partial.rename(bad)
        raise ValueError(f'SHA256 mismatch; kept as {bad.name}')
    info = validate(partial)
    partial.rename(target)
    return f'verified {info}'


def with_network_retries(task, label):
    """Retry transient failures; validation ValueError is not retried."""
    wait = 30
    while True:
        try:
            return task()
        except ValueError:
            raise
        except Exception as exc:
            print(f'WAIT {label} {type(exc).__name__}; retrying in {wait}s', flush=True)
            time.sleep(wait)
            wait = min(wait * 2, 600)


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    queue = make_plan(out, args)['queue']
    print(f'QUEUE {len(queue)} files {sum(item["size"] for item in queue) / 1e9:.2f} GB', flush=True)
    failed = []

    def run(record):
        try:
            result = with_network_retries(lambda: fetch(record, out), record['path'])
            print(f'DONE {record["path"]} {result}', flush=True)
        except Exception as exc:
            failed.append(record['path'])
            print(f'FAILED {record["path"]} {type(exc).__name__}: {exc}', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, queue))
    print(f'FINISHED files={len(queue)} failed={len(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
