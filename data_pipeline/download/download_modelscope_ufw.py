"""Download a seeded sample of Ultra-FineWeb parquet shards from ModelScope.

Shard order is fixed before any download: per language, the shard list is
shuffled with --seed (shards already used by data/ufw_v1 are excluded), then
the two languages are interleaved so any prefix of the queue is balanced.
Each file is fetched with HTTP Range requests (resumable .partial files),
size- and SHA256-checked against ModelScope's metadata, and parquet-validated
before being renamed into place. Existing final files are skipped.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import random
import threading
import time

import requests

API = 'https://www.modelscope.cn/api/v1/datasets/OpenBMB/Ultra-FineWeb'
CHUNK = 16 * 1024 * 1024
USED_BY_UFW_V1 = {
    'data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet',
    'data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet',
}
lock = threading.Lock()


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
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory for plan.json and shards; use a durable path outside the repository.',
    )
    parser.add_argument('--seed', type=int, default=20261109, help='Seed for deterministic per-language shuffling.')
    parser.add_argument(
        '--per-language', type=positive_int, default=100,
        help='Number of shards to select from each language before applying --limit.',
    )
    parser.add_argument('--limit', type=nonnegative_int, default=0, help='Only first N queue entries; 0 means all.')
    parser.add_argument('--workers', type=positive_int, default=4, help='Maximum concurrent shard downloads.')
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


def get_json(session, url, params=None):
    for attempt in range(6):
        try:
            response = session.get(url, params=params, timeout=(20, 60))
            response.raise_for_status()
            body = response.json()
            if body.get('Code') != 200:
                raise RuntimeError(f'ModelScope error: {body.get("Message")}')
            return body['Data']
        except Exception:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)


def list_shards(session, lang):
    files, page = [], 1
    while True:
        data = get_json(
            session,
            f'{API}/repo/tree',
            {
                'Revision': 'master',
                'Root': f'data/ultrafineweb_{lang}',
                'PageNumber': page,
                'PageSize': 500,
            },
        )
        batch = [item for item in data['Files'] if item['Path'].endswith('.parquet')]
        files += [
            {'lang': lang, 'path': item['Path'], 'size': item['Size'], 'sha256': item['Sha256']}
            for item in batch
        ]
        if len(data['Files']) < 500:
            return sorted(files, key=lambda item: item['path'])
        page += 1


def plan(session, out, seed, count):
    """Create or reuse the deterministic selection plan."""
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'plan.json'
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['seed'] != seed:
            raise ValueError('Existing plan has a different seed; refusing to mix samples')
        return saved
    queues = {}
    for lang in ('zh', 'en'):
        shards = [item for item in list_shards(session, lang) if item['path'] not in USED_BY_UFW_V1]
        random.Random(f'{seed}-{lang}').shuffle(shards)
        queues[lang] = shards
        print(f'LISTED {lang} {len(shards)} shards', flush=True)
    order = [queue[i] for i in range(count) for queue in (queues['zh'], queues['en']) if i < len(queue)]
    saved = {
        'source': API,
        'revision': 'master',
        'seed': seed,
        'per_language': count,
        'selection': 'per-language seeded shuffle excluding ufw_v1 shards, interleaved zh/en',
        'queue': order,
    }
    atomic_write_json(path, saved)
    return saved


def apply_limit(queue, limit):
    if limit < 0:
        raise ValueError('limit must be zero or greater')
    return queue[:limit or None]


def fetch(record, out, status):
    import pyarrow.parquet as pq

    target = safe_target(out, record['path'])
    if target.exists():
        return 'exists'
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix('.parquet.partial')
    session = requests.Session()
    url = f'{API}/repo'
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
                        url,
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
            with lock:
                status['bytes'] += len(data)
    if partial.stat().st_size != total:
        raise ValueError(f'File size mismatch for {record["path"]}')
    if sha256_file(partial) != record['sha256']:
        bad = partial.with_suffix('.sha_mismatch')
        partial.rename(bad)
        raise ValueError(f'SHA256 mismatch for {record["path"]}; kept as {bad.name}')
    parquet = pq.ParquetFile(partial)
    if not {'content', 'score', 'source'} <= set(parquet.schema_arrow.names):
        raise ValueError(f'Unexpected schema {parquet.schema_arrow.names}')
    rows = parquet.metadata.num_rows
    partial.rename(target)
    return f'verified rows={rows}'


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
    saved = plan(requests.Session(), out, args.seed, args.per_language)
    queue = apply_limit(saved['queue'], args.limit)
    status = {'bytes': 0}
    started = time.monotonic()
    failed = []

    def run(record):
        try:
            result = with_network_retries(lambda: fetch(record, out, status), record['path'])
            with lock:
                rate = status['bytes'] / max(time.monotonic() - started, 1) / 1e6
            print(f'DONE {record["path"]} {result} session_rate={rate:.2f}MB/s', flush=True)
        except Exception as exc:
            failed.append(record['path'])
            print(f'FAILED {record["path"]} {type(exc).__name__}: {exc}', flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, queue))
    print(f'FINISHED files={len(queue)} failed={len(failed)}', flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
