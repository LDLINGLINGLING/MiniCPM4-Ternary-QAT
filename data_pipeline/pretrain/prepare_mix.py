"""Frozen, bounded, resumable pretraining corpus preparation (no model training).

Run outside raw download directories using isolated Python::

    python -I data_pipeline/pretrain/prepare_mix.py freeze --root /PROJECT
    python -I data_pipeline/pretrain/prepare_mix.py run --root /PROJECT --max-batches 2
    python -I data_pipeline/pretrain/prepare_mix.py run --root /PROJECT

Freeze hashes only completed files in the download plans. Later downloads are
never added. SQLite commits follow output fsync; recovery truncates uncommitted
output tails. READY is published only when every domain meets both token targets.
No timestamps, wall-clock durations or invocation-specific paths enter manifest.
"""
import argparse
from array import array
from contextlib import ExitStack, contextmanager
from concurrent.futures import ProcessPoolExecutor
import fcntl
import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import sys

# -I removes the script directory too. Add only our trusted source directory,
# never the working directory, raw data directory, or model directory.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from data_pipeline.sft.gsm8k_data import atomic_json, sha256_file
from data_pipeline.pretrain.decontam import ExclusionIndex, UNCOVERED, discover_benchmarks, normalize, questions

DOMAINS = ('web_zh', 'web_en', 'math', 'code')
CAPS = dict(zip(DOMAINS, (1_840_000_000, 1_920_000_000, 350_000_000, 350_000_000)))
DEFAULT_ROOT = ROOT
DEFAULT_PLANS = ('data/ufw_raw_v2/plan.json',
                 'data/math_raw_v1/ultradata_math/plan.json',
                 'data/math_raw_v1/ultradata_math_l3/plan.json',
                 'data/code_raw_v1/opc_annealing/plan.json')


def config():
    return {'seed': 20261110, 'train_caps': dict(CAPS), 'val_windows': 256,
            'window_tokens': 2049, 'val_modulus': 100, 'batch_docs': 512,
            'batch_bytes': 8 * 1024 * 1024, 'max_doc_bytes': 1024 * 1024,
            'parquet_batch_rows': 128, 'workers': 128, 'min_chars': 1}


def versions():
    result = {'python': sys.version.split()[0]}
    for package in ('transformers', 'tokenizers', 'numpy', 'pyarrow', 'pyahocorasick'):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = None
    return result


def fingerprint(path):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha256_file(path)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
            after.st_size, after.st_mtime_ns, after.st_ino):
        raise ValueError(f'File changed while hashing: {path}')
    return {'path': str(path), 'size': after.st_size, 'sha256': digest}


def verify(spec):
    path = Path(spec['path'])
    if not path.is_file() or path.stat().st_size != spec['size']:
        raise ValueError(f'Size/missing-file mismatch: {path}')
    if sha256_file(path) != spec['sha256']:
        raise ValueError(f'SHA256 mismatch: {path}')


def code_identity():
    names = ('data_pipeline/pretrain/prepare_mix.py',
             'data_pipeline/pretrain/decontam.py',
             'data_pipeline/pretrain/prepare_data.py',
             'data_pipeline/sft/gsm8k_data.py')
    return {name: sha256_file(ROOT / name) for name in names}


def tokenizer_identity(directory):
    directory = Path(directory).resolve()
    names = ('tokenizer*', 'special_tokens_map.json', 'added_tokens.json',
             'vocab.*', 'merges.txt', '*.model', 'config.json')
    paths = sorted({p for name in names for p in directory.glob(name) if p.is_file()})
    if not paths:
        raise ValueError(f'No tokenizer files in {directory}')
    return {'path': str(directory), 'files': [fingerprint(p) for p in paths]}


def freeze_sources(root, plan_paths):
    """Exact planned final paths only; missing, wrong-size and partial ignored.

    A same-sized completed file with wrong SHA is an error, not an exclusion.
    Enumeration is captured before hashing so downloads finishing during hashing
    cannot expand the snapshot. Repo paths must remain under their plan directory.
    """
    plans, candidates, excluded = [], [], []
    for plan_path in plan_paths:
        plan_path = Path(plan_path).resolve()
        plan = json.loads(plan_path.read_text())
        identity = fingerprint(plan_path)
        plans.append(identity)
        for item in plan['queue']:
            relative = Path(item['path'])
            path = (plan_path.parent / relative).resolve()
            if not path.is_relative_to(plan_path.parent) or relative.is_absolute():
                raise ValueError(f'Unsafe planned path: {relative}')
            reason = None
            if path.suffix not in ('.jsonl', '.parquet') or 'partial' in path.name:
                reason = 'partial_or_unsupported'
            elif not path.is_file():
                reason = 'not_completed'
            elif path.stat().st_size != item['size']:
                reason = 'incomplete_size'
            if reason:
                excluded.append({'plan': str(plan_path), 'path': str(relative), 'reason': reason})
                continue
            repo = item.get('repo', plan.get('repo', plan.get('source', 'unknown')))
            if item.get('lang') in ('zh', 'en'):
                domain = 'web_' + item['lang']
            elif 'math' in str(repo).lower():
                domain = 'math'
            elif 'opc' in str(repo).lower() or 'code' in str(repo).lower():
                domain = 'code'
            else:
                raise ValueError(f'Cannot infer domain: {plan_path}: {relative}')
            source = f'{domain}|{repo}|{relative.parent.as_posix()}'
            candidates.append({'path': str(path), 'size': int(item['size']),
                               'sha256': item['sha256'], 'repo': repo,
                               'repo_path': relative.as_posix(), 'domain': domain,
                               'source': source, 'plan': str(plan_path),
                               'revision': plan.get('revision'),
                               'field': 'content' if path.suffix == '.parquet' else 'text',
                               'format': path.suffix[1:],
                               'license': item.get('license', plan.get('license', 'unverified'))})
    seen = set()
    for spec in candidates:
        if spec['path'] in seen:
            raise ValueError(f'Duplicate planned file: {spec["path"]}')
        seen.add(spec['path'])
        verify(spec)
    return plans, candidates, excluded


def fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextmanager
def locked(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / '.lock').open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f'Another preparation owns {output}') from None
        yield output


def freeze(root, output, tokenizer=None, cfg=None, plan_paths=None, benchmark_files=None):
    root = Path(root).resolve()
    cfg = config() if cfg is None else cfg
    if not 1 <= cfg['workers'] <= 128 or cfg['max_doc_bytes'] > cfg['batch_bytes']:
        raise ValueError('Require 1..16 workers and max_doc_bytes <= batch_bytes')
    if set(cfg['train_caps']) != set(DOMAINS) or min(cfg['train_caps'].values()) <= 0:
        raise ValueError('Need positive train caps for all four domains')
    if min(cfg[k] for k in ('batch_docs', 'batch_bytes', 'max_doc_bytes',
                           'parquet_batch_rows', 'val_windows', 'window_tokens', 'val_modulus')) < 1:
        raise ValueError('Batch and validation settings must be positive')
    with locked(output) as out:
        if (out / 'freeze.json').exists() or (out / 'manifest.json').exists():
            raise ValueError('Snapshot already exists; use run to resume or a new output directory')
        if tokenizer is None:
            paths = sorted(root.glob('MiniCPM4-0.5B-*'))
            if len(paths) != 1:
                raise ValueError('Specify --tokenizer explicitly (no unique base model directory)')
            tokenizer = paths[0]
        plans, sources, excluded = freeze_sources(root, plan_paths or [root / p for p in DEFAULT_PLANS])
        missing = set(DOMAINS) - {s['domain'] for s in sources}
        if missing:
            raise ValueError(f'No completed files for domains: {sorted(missing)}')
        benchmarks = discover_benchmarks(root) if benchmark_files is None else benchmark_files
        benchmarks = [{**spec, **fingerprint(spec['path'])} for spec in benchmarks]
        # Read each question now, fail on malformed files before any token output.
        benchmark_counts = {}
        for _, info in questions(benchmarks):
            key = info['benchmark'] + '/' + info['split']
            benchmark_counts[key] = benchmark_counts.get(key, 0) + 1
        frozen = {'version': 1, 'config': cfg, 'plans': plans, 'sources': sources,
                  'excluded_at_freeze': excluded, 'benchmarks': benchmarks,
                  'benchmark_counts': benchmark_counts, 'uncovered_benchmarks': UNCOVERED,
                  'tokenizer': tokenizer_identity(tokenizer), 'code': code_identity(),
                  'software': versions(), 'normalization': 'NFKC/casefold/collapse whitespace',
                  'decontamination': 'normalized embedded whole question OR contiguous 13 units '
                      '(English words, numbers, individual CJK characters); question stems only',
                  'limitations': 'Boilerplate false positives; semantic paraphrases missed; '
                      'does not remove contamination already in base model; unverified licenses '
                      'do not authorize redistribution',
                  'split': 'SHA256(normalized full text) interpreted as integer modulo 100 == 0 '
                      'is validation; globally deduplicated; no split reseeding',
                  'interleaving': 'sorted subsource keys, one document per turn, files in plan order'}
        atomic_json(out / 'freeze.json', frozen)
        fsync_dir(out)
        return frozen


def document_hash(text):
    return hashlib.sha256(normalize(text).encode('utf-8')).digest()


def split_for(digest, modulus=100):
    return 'val' if int.from_bytes(digest, 'big') % modulus == 0 else 'train'


def iter_records(spec, position, cfg):
    """Yield (next cursor, text-or-None, status), bounded JSONL and Parquet reads.

    JSONL cursors are byte offsets; Parquet cursors are absolute row numbers.
    Parquet resume skips complete row groups via metadata, not decoded payloads.
    Arrow decodes at most parquet_batch_rows rows at once; an individual malicious
    oversized Parquet cell can still require memory to decode before rejection.
    """
    if spec['format'] == 'jsonl':
        with open(spec['path'], 'rb') as handle:
            handle.seek(position)
            limit = cfg['max_doc_bytes']
            while True:
                line = handle.readline(limit + 1)
                if not line:
                    return
                too_long = len(line) > limit
                while line and not line.endswith(b'\n'):
                    # Drain oversized lines without retaining them.
                    more = handle.readline(limit + 1)
                    if not more:
                        break
                    too_long = True
                    line = more
                if too_long:
                    yield handle.tell(), None, 'oversize'
                    continue
                try:
                    row = json.loads(line)
                    text = row.get(spec['field']) if isinstance(row, dict) else None
                except (ValueError, UnicodeError):
                    text = None
                yield handle.tell(), text, 'ok' if isinstance(text, str) else 'invalid'
    else:
        import pyarrow.parquet as pq
        with pq.ParquetFile(spec['path']) as parquet:
            if spec['field'] not in parquet.schema_arrow.names:
                raise ValueError(f'Missing field {spec["field"]}: {spec["path"]}')
            start = 0
            for group in range(parquet.num_row_groups):
                end = start + parquet.metadata.row_group(group).num_rows
                if end <= position:
                    start = end
                    continue
                row_number = start
                for batch in parquet.iter_batches(batch_size=cfg['parquet_batch_rows'],
                        row_groups=[group], columns=[spec['field']], use_threads=False):
                    for scalar in batch.column(0):
                        row_number += 1
                        if row_number <= position:
                            continue
                        text = scalar.as_py()
                        yield row_number, text, 'ok' if isinstance(text, str) else 'invalid'
                start = end


class SourceReader:
    def __init__(self, files, cursor, cfg):
        self.files, self.cursor, self.cfg = files, cursor, cfg
        self.reader = None

    def next(self):
        while self.cursor['file'] < len(self.files):
            spec = self.files[self.cursor['file']]
            if self.reader is None:
                verify(spec)
                self.reader = iter_records(spec, self.cursor['position'], self.cfg)
            try:
                position, text, status = next(self.reader)
            except StopIteration:
                self.reader = None
                self.cursor['file'] += 1
                self.cursor['position'] = 0
                continue
            old_position = self.cursor['position']
            self.cursor['position'] = position
            return spec, old_position, text, status
        self.cursor['exhausted'] = True
        return None

    def close(self):
        if self.reader is not None:
            self.reader.close()


_worker_tokenizer = None


def init_tokenizer(path):
    global _worker_tokenizer
    # Reuse the established AutoTokenizer initialization and Rayon restrictions.
    from data_pipeline.pretrain.prepare_data import _init
    from data_pipeline.pretrain import prepare_data
    _init(path)
    _worker_tokenizer = prepare_data._tokenizer
    if _worker_tokenizer.bos_token_id != 1:
        raise ValueError('This stream format requires tokenizer BOS=1')


def encode_texts(texts):
    # Explicit BOS=1, separator EOS=2, with no tokenizer-added special tokens.
    rows = _worker_tokenizer(texts, add_special_tokens=False, truncation=False)['input_ids']
    outputs = []
    for row in rows:
        ids = array('I', [1] + row + [2])
        if ids.itemsize != 4:
            raise RuntimeError('uint32 array unavailable')
        if sys.byteorder != 'little':
            ids.byteswap()
        outputs.append(ids.tobytes())
    return outputs


def initial_state(groups):
    return {'batches': 0, 'turn': 0,
            'cursors': {key: {'file': 0, 'position': 0, 'exhausted': False} for key in groups},
            'tokens': {d: {'train': 0, 'val': 0} for d in DOMAINS},
            'counts': {}, 'source_tokens': {key: {'train': 0, 'val': 0} for key in groups},
            'benchmark_hits': {}}


def increment(state, key, amount=1):
    state['counts'][key] = state['counts'].get(key, 0) + amount


def enough(state, domain, cfg):
    tokens = state['tokens'][domain]
    return tokens['train'] >= cfg['train_caps'][domain] and tokens['val'] >= (
        cfg['val_windows'] * cfg['window_tokens'])


def recover_outputs(out, stage, state, stack):
    handles = {}
    for domain in DOMAINS:
        for split in ('train', 'val'):
            name = f'{domain}_{split}.bin'
            path = stage / name
            # An interrupted final publication may already have renamed some files.
            if not path.exists() and (out / name).exists():
                os.replace(out / name, path)
            length = state['tokens'][domain][split] * 4
            if not path.exists() and length:
                raise ValueError(f'Missing committed output: {name}')
            handle = stack.enter_context(path.open('r+b' if path.exists() else 'w+b'))
            if handle.seek(0, os.SEEK_END) < length:
                raise ValueError(f'Truncated committed output: {name}')
            handle.truncate(length)
            handle.seek(length)
            handles[(domain, split)] = handle
    return handles


def finalize(out, stage, frozen, frozen_sha, state):
    streams = {}
    for domain in DOMAINS:
        entries = {}
        for split in ('train', 'val'):
            name = f'{domain}_{split}.bin'
            path = stage / name
            tokens = state['tokens'][domain][split]
            if path.stat().st_size != tokens * 4:
                raise ValueError(f'Output size mismatch: {name}')
            entries[split] = {'path': name, 'tokens': tokens, 'sha256': sha256_file(path)}
            os.replace(path, out / name)
        streams[domain] = {'train': [entries['train']], 'val': entries['val']}
    fsync_dir(stage)
    fsync_dir(out)
    source_shares = {}
    for source, counts in state['source_tokens'].items():
        domain = source.split('|', 1)[0]
        source_shares[source] = {**counts, 'train_fraction': counts['train'] / state['tokens'][domain]['train']}
    manifest = {'version': 1, 'streams': streams, 'freeze_sha256': frozen_sha,
                'dtype': '<u4', 'document_encoding': '[BOS=1] + original text tokens + [EOS=2]',
                'provenance': frozen, 'statistics': state, 'source_token_shares': source_shares,
                'hit_log': 'staging/state.sqlite table hits (first canonical hit per rejected document)',
                'validation': {'minimum_windows': frozen['config']['val_windows'],
                               'window_tokens': frozen['config']['window_tokens'],
                               'policy': 'hash-selected fixed stream; whole last document retained'}}
    # SQLite restores sorted JSON keys; canonicalize fresh and resumed manifests alike.
    manifest = json.loads(json.dumps(manifest, sort_keys=True))
    atomic_json(out / 'manifest.json', manifest)
    fsync_dir(out)
    atomic_json(out / 'READY', {'version': 1, 'manifest_sha256': sha256_file(out / 'manifest.json')})
    fsync_dir(out)
    return manifest


def run(output, max_batches=None, encoder=None, require_fast=True):
    """Resume a snapshot. encoder is injectable ONLY for local synthetic unit tests."""
    if max_batches is not None and max_batches < 1:
        raise ValueError('--max-batches must be positive')
    with locked(output) as out, ExitStack() as stack:
        frozen = json.loads((out / 'freeze.json').read_text())
        frozen_sha = sha256_file(out / 'freeze.json')
        if frozen['version'] != 1:
            raise ValueError('Unsupported snapshot version')
        if frozen['software'] != versions():
            raise ValueError('Preparation software changed since freeze; use original environment or new snapshot')
        if frozen['code'] != code_identity():
            raise ValueError('Preparation code changed since freeze; use original environment or new snapshot')
        if (out / 'READY').exists():
            ready = json.loads((out / 'READY').read_text())
            if sha256_file(out / 'manifest.json') != ready['manifest_sha256']:
                raise ValueError('READY manifest SHA mismatch')
            manifest = json.loads((out / 'manifest.json').read_text())
            if manifest['freeze_sha256'] != frozen_sha:
                raise ValueError('Frozen identity changed after publication')
            for stream in manifest['streams'].values():
                for entry in stream['train'] + [stream['val']]:
                    verify({'path': out / entry['path'], 'size': entry['tokens'] * 4,
                            'sha256': entry['sha256']})
            return manifest
        for spec in frozen['tokenizer']['files'] + frozen['benchmarks']:
            verify(spec)
        cfg = frozen['config']
        index = ExclusionIndex(questions(frozen['benchmarks']), require_fast=require_fast)
        groups = {}
        for source in frozen['sources']:
            groups.setdefault(source['source'], []).append(source)
        groups = dict(sorted(groups.items()))
        keys = list(groups)
        stage = out / 'staging'
        stage.mkdir(exist_ok=True)
        db = sqlite3.connect(stage / 'state.sqlite')
        stack.callback(db.close)
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=FULL')
        db.execute('PRAGMA cache_size=-65536')
        db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), identity TEXT, payload TEXT)')
        db.execute('CREATE TABLE IF NOT EXISTS seen (hash BLOB PRIMARY KEY, domain TEXT, split TEXT) WITHOUT ROWID')
        db.execute('CREATE TABLE IF NOT EXISTS hits (id INTEGER PRIMARY KEY, source TEXT, position INTEGER, details TEXT)')
        saved = db.execute('SELECT identity,payload FROM state WHERE id=1').fetchone()
        if saved:
            if saved[0] != frozen_sha:
                raise ValueError('Frozen identity differs from SQLite checkpoint')
            state = json.loads(saved[1])
        else:
            state = initial_state(groups)
            db.execute('INSERT INTO state VALUES(1,?,?)', (frozen_sha, json.dumps(state, sort_keys=True)))
            db.commit()
        handles = recover_outputs(out, stage, state, stack)
        fsync_dir(stage)
        readers = {key: SourceReader(files, state['cursors'][key], cfg) for key, files in groups.items()}
        for reader in readers.values():
            stack.callback(reader.close)
        if encoder is None:
            pool = stack.enter_context(ProcessPoolExecutor(max_workers=cfg['workers'],
                mp_context=mp.get_context('spawn'), initializer=init_tokenizer,
                initargs=(frozen['tokenizer']['path'],)))
            # A finite byte-bounded list; worker/init failure propagates, never respawns forever.
            map_encode = lambda chunks: pool.map(encode_texts, chunks, chunksize=1)
        else:
            map_encode = lambda chunks: [encoder(chunk) for chunk in chunks]
        invocation_batches = 0
        while not all(enough(state, d, cfg) for d in DOMAINS):
            active = [key for key in keys if not state['cursors'][key]['exhausted']
                      and not enough(state, groups[key][0]['domain'], cfg)]
            if not active:
                deficits = {d: {'train': max(0, cfg['train_caps'][d] - state['tokens'][d]['train']),
                                'val': max(0, cfg['val_windows'] * cfg['window_tokens'] - state['tokens'][d]['val'])}
                            for d in DOMAINS if not enough(state, d, cfg)}
                atomic_json(out / 'INSUFFICIENT.json', {'deficits_tokens': deficits, 'statistics': state})
                raise RuntimeError(f'Frozen sources exhausted; not READY. Missing tokens: {deficits}')
            db.execute('BEGIN IMMEDIATE')
            pending, raw_count, byte_count = [], 0, 0
            # Reserve worst-case one document before reading to bound retained text.
            while raw_count < cfg['batch_docs'] and byte_count + cfg['max_doc_bytes'] <= cfg['batch_bytes']:
                selected = None
                for _ in keys:
                    key = keys[state['turn'] % len(keys)]
                    state['turn'] += 1
                    if not state['cursors'][key]['exhausted'] and not enough(state, groups[key][0]['domain'], cfg):
                        selected = key
                        break
                if selected is None:
                    break
                record = readers[selected].next()
                if record is None:
                    continue
                spec, position, text, status = record
                raw_count += 1
                increment(state, 'raw')
                if status != 'ok':
                    increment(state, status)
                    continue
                size = len(text.encode('utf-8'))
                if size > cfg['max_doc_bytes']:
                    increment(state, 'oversize')
                    continue
                if len(text.strip()) < cfg['min_chars']:
                    increment(state, 'empty')
                    continue
                byte_count += size
                digest = document_hash(text)
                split = split_for(digest, cfg['val_modulus'])
                domain = spec['domain']
                if db.execute('SELECT 1 FROM seen WHERE hash=?', (digest,)).fetchone():
                    increment(state, 'duplicates')
                    continue
                hit = index.match(text)
                if hit:
                    increment(state, 'contaminated')
                    benchmark = hit['benchmark']
                    state['benchmark_hits'][benchmark] = state['benchmark_hits'].get(benchmark, 0) + 1
                    db.execute('INSERT INTO hits(source,position,details) VALUES(?,?,?)',
                               (spec['path'], position, json.dumps(hit, sort_keys=True)))
                    continue
                db.execute('INSERT INTO seen VALUES(?,?,?)', (digest, domain, split))
                pending.append((selected, domain, split, text))
            chunks = []
            chunk_size = max(1, (len(pending) + cfg['workers'] - 1) // cfg['workers'])
            for start in range(0, len(pending), chunk_size):
                chunks.append([p[3] for p in pending[start:start + chunk_size]])
            encoded = [value for batch in map_encode(chunks) for value in batch]
            if len(encoded) != len(pending):
                raise ValueError('Tokenizer returned incorrect document count')
            touched = set()
            for (source, domain, split, _), data in zip(pending, encoded):
                target = cfg['train_caps'][domain] if split == 'train' else cfg['val_windows'] * cfg['window_tokens']
                if state['tokens'][domain][split] >= target:
                    increment(state, 'past_cap')
                    continue
                if not isinstance(data, bytes) or len(data) % 4 or len(data) < 8:
                    raise ValueError('Tokenizer returned invalid uint32 document')
                tokens = len(data) // 4
                handles[(domain, split)].write(data)
                touched.add((domain, split))
                state['tokens'][domain][split] += tokens
                state['source_tokens'][source][split] += tokens
                increment(state, 'kept')
                increment(state, 'kept_' + split)
            for key in touched:
                handles[key].flush()
                os.fsync(handles[key].fileno())
            state['batches'] += 1
            db.execute('UPDATE state SET payload=? WHERE id=1', (json.dumps(state, sort_keys=True),))
            db.commit()
            invocation_batches += 1
            print(json.dumps({'batches': state['batches'], 'tokens': state['tokens'],
                              'counts': state['counts']}, sort_keys=True), flush=True)
            if max_batches is not None and invocation_batches >= max_batches:
                return {'paused': True, 'statistics': state}
        # Flush even empty/recovered handles before names become public.
        for handle in handles.values():
            handle.flush()
            os.fsync(handle.fileno())
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        return finalize(out, stage, frozen, frozen_sha, state)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('freeze', 'setup', 'run', 'resume'))
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--output', type=Path, help='Default: ROOT/data/mix_v1')
    parser.add_argument('--tokenizer', type=Path, help='Freeze only; defaults to unique original MiniCPM4 directory')
    parser.add_argument('--plan', type=Path, action='append',
                        help='Download plan JSON; repeatable. Defaults to the four production plan paths')
    parser.add_argument('--max-batches', type=int, help='Pause after N committed batches; run/resume only')
    parser.add_argument('--workers', type=int, default=16, help='Freeze only, 1..16 spawn workers')
    args = parser.parse_args()
    if not sys.flags.isolated:
        parser.error('Run raw-data preparation using python -I')
    output = args.output or args.root / 'data/mix_v1'
    if args.command in ('freeze', 'setup'):
        cfg = config()
        cfg['workers'] = args.workers
        frozen = freeze(args.root, output, tokenizer=args.tokenizer, cfg=cfg,
                        plan_paths=args.plan)
        print(json.dumps({'frozen_files': len(frozen['sources']),
                          'benchmark_counts': frozen['benchmark_counts'], 'output': str(output)}, sort_keys=True))
    else:
        result = run(output, max_batches=args.max_batches)
        print('PAUSED' if result.get('paused') else 'READY', flush=True)


if __name__ == '__main__':
    main()
