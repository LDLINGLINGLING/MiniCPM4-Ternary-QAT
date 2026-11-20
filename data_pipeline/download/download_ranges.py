"""Resume an Ultra-FineWeb download using explicit HTTP Range requests.

Read the manifest produced by download_ultrafineweb.py, fetch each selected
parquet shard into <output>, validate its size/SHA256/schema/text sample, and
write <output>/range-download-manifest.json. Existing final files are never
overwritten; .partial files are resumed. HF_TOKEN may be set in the environment
when the repository requires authentication.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import time
from urllib.parse import quote

import requests

DEFAULT_ENDPOINT = 'https://hf-mirror.com'
CHUNK = 8 * 1024 * 1024


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
        help='Destination directory for shards and range-download-manifest.json.',
    )
    parser.add_argument(
        '--manifest', type=Path, default=None,
        help='Input Ultra-FineWeb manifest; default: <output>/download-manifest.json.',
    )
    parser.add_argument('--workers', type=positive_int, default=2, help='Maximum concurrent shard downloads.')
    parser.add_argument(
        '--limit', type=nonnegative_int, default=0,
        help='Only first N entries from the input manifest; 0 means all.',
    )
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


def apply_limit(records, limit):
    if limit < 0:
        raise ValueError('limit must be zero or greater')
    return records[:limit or None]


def save_manifest(path, manifest):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def fetch(record, out, repo, revision, endpoint):
    """Fetch one manifest record, resuming its partial file if present."""
    import pyarrow.parquet as pq

    session = requests.Session()
    token = os.environ.get('HF_TOKEN')
    if token:
        session.headers['Authorization'] = f'Bearer {token}'
    target = safe_target(out, record['path'])
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + '.partial')
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite existing target: {target}')
    total = record['expected_bytes']
    url = (
        f'{endpoint.rstrip("/")}/datasets/{repo}/resolve/'
        f'{quote(revision, safe="")}/{quote(record["path"], safe="/")}'
    )
    start = partial.stat().st_size if partial.exists() else 0
    if start > total:
        raise ValueError(f'Partial larger than remote: {partial}')
    language = record.get('language', record['path'])
    print(f'START {language} offset={start} bytes={total}', flush=True)
    started = time.monotonic()
    with partial.open('ab') as handle:
        while start < total:
            end = min(start + CHUNK, total) - 1
            for attempt in range(6):
                try:
                    # Distinct URLs avoid intermediary caches reusing a different range.
                    with session.get(
                        url,
                        params={'download': 'true', 'range_start': start},
                        headers={'Range': f'bytes={start}-{end}', 'Accept-Encoding': 'identity'},
                        stream=True,
                        timeout=(20, 120),
                    ) as response:
                        response.raise_for_status()
                        expected_range = f'bytes {start}-{end}/{total}'
                        actual_range = response.headers.get('Content-Range')
                        if response.status_code != 206 or actual_range != expected_range:
                            raise RuntimeError(f'Invalid range response {response.status_code}: {actual_range}')
                        data = response.raw.read(end - start + 2)
                        if len(data) != end - start + 1:
                            raise RuntimeError(f'Unexpected chunk length: {len(data)}')
                    break
                except Exception as exc:
                    print(
                        f'RETRY {language} offset={start} attempt={attempt + 1} {type(exc).__name__}',
                        flush=True,
                    )
                    if attempt == 5:
                        raise
                    time.sleep(min(2 ** attempt, 20))
            handle.write(data)
            handle.flush()
            start += len(data)
            if start % (CHUNK * 16) == 0 or start == total:
                print(
                    f'PROGRESS {language} {start}/{total} ({100 * start / total:.1f}%) '
                    f'elapsed={time.monotonic() - started:.0f}s',
                    flush=True,
                )
    if partial.stat().st_size != total:
        raise ValueError('File size mismatch')
    sha = sha256_file(partial)
    if record.get('expected_sha256') and sha != record['expected_sha256']:
        raise ValueError('SHA256 mismatch')
    parquet = pq.ParquetFile(partial)
    fields = parquet.schema_arrow.names
    if not {'content', 'score', 'source'}.issubset(fields):
        raise ValueError(f'Unexpected schema {fields}')
    try:
        batch = next(parquet.iter_batches(batch_size=8, columns=['content']))
    except StopIteration as exc:
        raise ValueError('Parquet shard has no rows') from exc
    sample = batch.column(0).to_pylist()
    if not all(isinstance(text, str) and text.strip() for text in sample):
        raise ValueError('Invalid text sample')
    result = {
        'language': language,
        'path': record['path'],
        'status': 'verified',
        'bytes': total,
        'sha256': sha,
        'sha256_matches_remote': bool(record.get('expected_sha256')),
        'rows': parquet.metadata.num_rows,
        'row_groups': parquet.num_row_groups,
        'fields': fields,
    }
    partial.rename(target)
    print('VERIFIED ' + json.dumps(result), flush=True)
    return result


def safe_fetch(record, out, repo, revision, endpoint):
    try:
        return fetch(record, out, repo, revision, endpoint)
    except Exception as exc:
        result = {
            'language': record.get('language'),
            'path': record.get('path'),
            'status': 'failed',
            'error': f'{type(exc).__name__}: {exc}',
        }
        print('FAILED ' + json.dumps(result), flush=True)
        return result


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest.expanduser().resolve() if args.manifest else out / 'download-manifest.json'
    original = json.loads(manifest_path.read_text(encoding='utf-8'))
    revision = original['revision']
    repo = original['repo']
    endpoint = original.get('endpoint', DEFAULT_ENDPOINT)
    records = apply_limit(original['files'], args.limit)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(
            lambda record: safe_fetch(record, out, repo, revision, endpoint),
            records,
        ))
    manifest = {
        'repo': repo,
        'revision': revision,
        'endpoint': endpoint,
        'method': f'Anonymous or HF_TOKEN-authenticated HTTP range GET via {endpoint}',
        'selection': original['selection'],
        'token_count': None,
        'files': results,
    }
    save_manifest(out / 'range-download-manifest.json', manifest)
    failed = any(record['status'] != 'verified' for record in results)
    print('FINISHED ' + ('WITH ERRORS' if failed else 'SUCCESS'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
