"""Download and verify the first complete Ultra-FineWeb shard per language.

The original Chinese and English shards are pinned to the repository's resolved
commit. Hugging Face Hub provides resumable downloads; this script additionally
checks remote size and SHA256, validates parquet content, and writes an atomic
manifest. Set HF_TOKEN in the environment only if authentication is required.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import requests

REPO = 'openbmb/Ultra-FineWeb'
ENDPOINT = 'https://hf-mirror.com'
SELECTED = {
    'zh': 'data/ultrafineweb_zh/ultrafineweb-zh-part-001-of-256.parquet',
    'en': 'data/ultrafineweb_en/ultrafineweb-en-part-0001-of-2048.parquet',
}


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory, created if needed; use a durable path outside the repository.',
    )
    return parser


def configure_huggingface():
    """Configure Hub behavior at execution time, never at module import."""
    os.environ['HF_ENDPOINT'] = ENDPOINT
    os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN'] = '1'
    os.environ['HF_HUB_DISABLE_XET'] = '1'
    os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
    os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '120'
    os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'


def get_json(session, url):
    for attempt in range(4):
        try:
            response = session.get(url, timeout=(20, 60))
            response.raise_for_status()
            return response.json()
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)


def normalize_lfs_sha256(value):
    """Convert a Hub LFS OID to a plain SHA256 hex string."""
    if value and value.startswith('sha256:'):
        return value.split(':', 1)[1]
    return value


def save_manifest(out, manifest):
    temporary = out / 'download-manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(out / 'download-manifest.json')


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_download(local, expected_bytes, expected_hash):
    """Check size, hash, schema, and an initial text sample."""
    import pyarrow.parquet as pq

    size = local.stat().st_size
    if size != expected_bytes:
        raise ValueError(f'File size mismatch: {size} != {expected_bytes}')
    sha = sha256_file(local)
    if expected_hash and sha != expected_hash:
        raise ValueError(f'SHA256 mismatch: {sha} != {expected_hash}')
    parquet = pq.ParquetFile(local)
    fields = parquet.schema_arrow.names
    if not {'content', 'score', 'source'}.issubset(fields):
        raise ValueError(f'Unexpected schema: {fields}')
    try:
        batch = next(parquet.iter_batches(batch_size=8, columns=['content']))
    except StopIteration as exc:
        raise ValueError('Parquet shard has no rows') from exc
    sample = batch.column(0).to_pylist()
    if not all(isinstance(text, str) and text.strip() for text in sample):
        raise ValueError('Empty or invalid text in initial sample')
    return {
        'bytes': size,
        'sha256': sha,
        'rows': parquet.metadata.num_rows,
        'row_groups': parquet.num_row_groups,
        'fields': fields,
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    configure_huggingface()
    token = os.environ.get('HF_TOKEN') or False
    session = requests.Session()
    if token:
        session.headers['Authorization'] = f'Bearer {token}'
    revision = get_json(session, f'{ENDPOINT}/api/datasets/{REPO}')['sha']
    manifest = {
        'repo': REPO,
        'revision': revision,
        'endpoint': ENDPOINT,
        'selection': 'First complete original shard of each language; not a random corpus sample.',
        'token_count': None,
        'files': [],
    }
    print(f'OUTPUT {out}\nREVISION {revision}', flush=True)
    save_manifest(out, manifest)
    failed = False
    for language, name in SELECTED.items():
        record = {'language': language, 'path': name, 'status': 'downloading'}
        manifest['files'].append(record)
        save_manifest(out, manifest)
        try:
            directory = name.rsplit('/', 1)[0]
            entries = get_json(session, f'{ENDPOINT}/api/datasets/{REPO}/tree/{revision}/{directory}?limit=10')
            metadata = next(item for item in entries if item['path'] == name)
            expected_hash = normalize_lfs_sha256(metadata.get('lfs', {}).get('oid'))
            record['expected_bytes'] = metadata['size']
            record['expected_sha256'] = expected_hash
            save_manifest(out, manifest)
            print(f'START {language} {name} bytes={metadata["size"]}', flush=True)

            from huggingface_hub import hf_hub_download

            for attempt in range(4):
                try:
                    local = Path(hf_hub_download(
                        repo_id=REPO,
                        filename=name,
                        repo_type='dataset',
                        revision=revision,
                        local_dir=str(out),
                        token=token,
                        endpoint=ENDPOINT,
                    ))
                    break
                except Exception as exc:
                    print(
                        f'RETRY {language} attempt={attempt + 1} {type(exc).__name__}: {exc}',
                        flush=True,
                    )
                    if attempt == 3:
                        raise
                    time.sleep(5 * (attempt + 1))
            verified = validate_download(local, metadata['size'], expected_hash)
            record.update(status='verified', sha256_matches_remote=bool(expected_hash), **verified)
            print('VERIFIED ' + json.dumps(record, ensure_ascii=False), flush=True)
        except Exception as exc:
            failed = True
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            print(f'FAILED {language}: {record["error"]}', flush=True)
        save_manifest(out, manifest)
    print('FINISHED ' + ('WITH ERRORS' if failed else 'SUCCESS'), flush=True)
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())
