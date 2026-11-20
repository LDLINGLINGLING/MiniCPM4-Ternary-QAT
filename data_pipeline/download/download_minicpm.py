"""Download and verify every file in a Hugging Face model repository.

Files are downloaded in deterministic size order with resumable ``.partial``
files. Size, SHA256 (when supplied by the Hub), JSON, and safetensors metadata
are validated before each file is renamed into place. The manifest is updated
atomically after every file. Existing final files are never overwritten.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import struct
import time
from urllib.parse import quote

import requests

ENDPOINT = 'https://hf-mirror.com'
CHUNK = 8 * 1024 * 1024


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output', type=Path, required=True,
        help='Output directory, created if needed; use a durable path outside the repository.',
    )
    parser.add_argument('--repo', default='openbmb/MiniCPM4-0.5B', help='Hugging Face model repository ID.')
    parser.add_argument(
        '--revision', default=None,
        help='Branch, tag, or commit to resolve; default: the repository default branch.',
    )
    return parser


def auth_headers():
    """Return optional Hub authentication without exposing the token."""
    token = os.environ.get('HF_TOKEN')
    return {'Authorization': f'Bearer {token}'} if token else {}


def safe_target(out, name):
    """Map a remote POSIX filename to a path below *out*."""
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe filename {name!r}')
    return out.joinpath(*relative.parts)


def sha256_file(path, chunk=CHUNK):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(chunk), b''):
            digest.update(block)
    return digest.hexdigest()


def save_manifest(out, manifest):
    """Atomically save the download manifest."""
    temporary = out / 'download-manifest.json.tmp'
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(out / 'download-manifest.json')


def get_metadata(session, repo, revision=None):
    suffix = f'/revision/{quote(revision, safe="")}' if revision else ''
    response = session.get(
        f'{ENDPOINT}/api/models/{repo}{suffix}',
        params={'blobs': 'true'},
        timeout=(20, 60),
    )
    response.raise_for_status()
    return response.json()


def validate_download(partial, name, total, expected, record):
    """Validate a completed partial file and return its computed SHA256."""
    if partial.stat().st_size != total:
        raise ValueError(f'File size mismatch: {partial.stat().st_size} != {total}')
    sha = sha256_file(partial)
    if expected and sha != expected:
        raise ValueError(f'SHA256 mismatch: {sha} != {expected}')
    if name.endswith('.json'):
        json.loads(partial.read_text(encoding='utf-8'))
    if name.endswith('.safetensors'):
        if total < 9:
            raise ValueError('Invalid safetensors file length')
        with partial.open('rb') as handle:
            raw_header_size = handle.read(8)
            if len(raw_header_size) != 8:
                raise ValueError('Truncated safetensors header')
            header_size = struct.unpack('<Q', raw_header_size)[0]
            if not 0 < header_size < min(total - 8, 64 * 1024 * 1024):
                raise ValueError('Invalid safetensors header length')
            header = json.loads(handle.read(header_size))
        tensors = {key: value for key, value in header.items() if key != '__metadata__'}
        for value in tensors.values():
            start, end = value['data_offsets']
            if not 0 <= start <= end <= total - 8 - header_size:
                raise ValueError('Invalid tensor offsets')
        record['tensor_count'] = len(tensors)
        record['tensor_dtypes'] = sorted({value['dtype'] for value in tensors.values()})
    return sha


def fetch_file(session, entry, out, repo, revision, record):
    """Fetch one metadata entry, resuming any compatible partial file."""
    name = entry['rfilename']
    target = safe_target(out, name)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    partial = target.with_name(target.name + '.partial')
    total = entry['size']
    expected = entry.get('lfs', {}).get('sha256')
    record.update(expected_bytes=total, expected_sha256=expected)
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > total:
        raise ValueError(f'Partial larger than remote: {partial}')
    url = f'{ENDPOINT}/{repo}/resolve/{quote(revision, safe="")}/{quote(name, safe="/")}'
    print(f'START {name} bytes={total}', flush=True)
    with partial.open('ab') as handle:
        while offset < total:
            end = min(offset + CHUNK, total) - 1
            for attempt in range(6):
                try:
                    # range_start distinguishes cached requests for different ranges.
                    with session.get(
                        url,
                        params={'download': 'true', 'range_start': offset},
                        headers={'Range': f'bytes={offset}-{end}', 'Accept-Encoding': 'identity'},
                        stream=True,
                        timeout=(20, 120),
                    ) as response:
                        response.raise_for_status()
                        if response.status_code == 206:
                            expected_range = f'bytes {offset}-{end}/{total}'
                            if response.headers.get('Content-Range') != expected_range:
                                raise ValueError('Unexpected Content-Range')
                        elif not (response.status_code == 200 and offset == 0 and end + 1 == total):
                            raise ValueError(f'Unexpected status {response.status_code}')
                        data = response.raw.read(end - offset + 2)
                        if len(data) != end - offset + 1:
                            raise ValueError(f'Unexpected response length {len(data)}')
                    break
                except Exception as exc:
                    print(
                        f'RETRY {name} offset={offset} attempt={attempt + 1} {type(exc).__name__}',
                        flush=True,
                    )
                    if attempt == 5:
                        raise
                    time.sleep(min(2 ** attempt, 20))
            handle.write(data)
            handle.flush()
            offset += len(data)
            if total > CHUNK and (offset % (16 * CHUNK) == 0 or offset == total):
                print(f'PROGRESS {name} {offset}/{total} ({offset / total:.1%})', flush=True)
    sha = validate_download(partial, name, total, expected, record)
    partial.rename(target)
    record.update(status='verified', bytes=total, sha256=sha, sha256_matches_remote=bool(expected))
    print('VERIFIED ' + json.dumps(record), flush=True)


def main(argv=None):
    args = build_parser().parse_args(argv)
    out = args.output.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers.update(auth_headers())
    metadata = get_metadata(session, args.repo, args.revision)
    revision = metadata['sha']
    manifest = {
        'repo': args.repo,
        'revision': revision,
        'endpoint': ENDPOINT,
        'remote_code_executed': False,
        'files': [],
    }
    print(f'REVISION {revision}\nOUTPUT {out}', flush=True)
    files = sorted(metadata['siblings'], key=lambda item: item.get('size', 0))
    for entry in files:
        name = entry['rfilename']
        record = {'path': name, 'status': 'downloading'}
        manifest['files'].append(record)
        save_manifest(out, manifest)
        try:
            fetch_file(session, entry, out, args.repo, revision, record)
        except Exception as exc:
            record.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            save_manifest(out, manifest)
            raise
        save_manifest(out, manifest)
    print('FINISHED SUCCESS', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
