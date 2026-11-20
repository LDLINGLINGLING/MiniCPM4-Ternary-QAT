"""Deterministic, replayable window sampling over flat uint32 token streams."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

DOMAIN_ORDER = ('web_zh', 'web_en', 'math', 'code')


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path):
    """Validate the published frozen pool once; returned paths are explicit and absolute.

    Distributed callers run this on rank zero and broadcast the verified result.
    """
    path = Path(path)
    if not path.is_absolute():
        raise ValueError('data_manifest must be an absolute path')
    path = path.resolve()
    if not (path.parent / 'READY').is_file():
        raise ValueError(f'data manifest is not READY: {path}')
    manifest = json.loads(path.read_text())
    if manifest.get('version') != 1 or not manifest.get('streams'):
        raise ValueError('Expected version 1 data manifest with streams')
    streams, seen = {}, set()
    for domain, stream in manifest['streams'].items():
        if not isinstance(stream.get('train'), list) or not stream['train']:
            raise ValueError(f'Empty training domain: {domain}')
        streams[domain] = {'train': [], 'val': None}
        for split, entries in (('train', stream['train']), ('val', [stream['val']])):
            for entry in entries:
                relative = Path(entry['path'])
                target = (path.parent / relative).resolve()
                if relative.is_absolute() or not target.is_relative_to(path.parent):
                    raise ValueError(f'Manifest path must stay inside its directory: {relative}')
                if target in seen:
                    raise ValueError(f'Duplicate manifest path: {target}')
                seen.add(target)
                tokens = entry['tokens']
                if type(tokens) is not int or tokens <= 0 or target.stat().st_size != tokens * 4:
                    raise ValueError(f'Manifest token count/size mismatch: {target}')
                if sha256_file(target) != entry['sha256']:
                    raise ValueError(f'Manifest SHA256 mismatch: {target}')
                if split == 'train':
                    streams[domain]['train'].append(str(target))
                else:
                    streams[domain]['val'] = str(target)
    digest = sha256_file(path)
    ready = json.loads((path.parent / 'READY').read_text())
    if ready.get('manifest_sha256') != digest:
        raise ValueError('READY manifest SHA256 mismatch')
    return {'path': str(path), 'sha256': digest, 'streams': streams}


def as_dirs(data):
    return [data] if isinstance(data, (str, Path)) else list(data)


class TokenStream:
    def __init__(self, paths, seq_len):
        if seq_len < 1:
            raise ValueError('seq_len must be positive')
        paths = [Path(p).resolve() for p in paths]
        if not paths or len(set(paths)) != len(paths):
            raise ValueError('Token stream requires unique, nonempty paths')
        if any(p.stat().st_size == 0 or p.stat().st_size % 4 for p in paths):
            raise ValueError('Token files must contain nonempty uint32 streams')
        self.files = [np.memmap(p, dtype=np.uint32, mode='r') for p in paths]
        self.paths = [str(p) for p in paths]
        self.window = seq_len + 1
        counts = np.asarray([len(f) // self.window for f in self.files], dtype=np.int64)
        self.offsets = np.concatenate([[0], np.cumsum(counts)])
        self.num_windows = int(self.offsets[-1])
        self.num_tokens = int(sum(len(f) for f in self.files))
        if self.num_windows == 0:
            raise ValueError(f'Empty domain: no complete {self.window}-token window in {self.paths}')

    def get(self, idx):
        if not 0 <= idx < self.num_windows:
            raise IndexError(idx)
        f = int(np.searchsorted(self.offsets, idx, side='right') - 1)
        s = (idx - int(self.offsets[f])) * self.window
        return torch.from_numpy(self.files[f][s:s + self.window].astype(np.int64))


def train_files(data, lang):
    files = []
    for d in as_dirs(data):
        files += sorted(Path(d).glob(f'{lang}_train*.bin'))
    if not files:
        raise FileNotFoundError(f'no {lang}_train*.bin under {as_dirs(data)}')
    return files


def validation_path(data_dir, lang, manifest=None):
    return (Path(manifest['streams'][lang]['val']) if manifest else
            Path(data_dir) / f'{lang}_val.bin')


class MixedSampler:
    def __init__(self, data, langs, weights, seq_len, seed, manifest=None, canonical_order=False):
        langs = list(langs)
        weights = np.asarray(weights, dtype=np.float64)
        if (not langs or len(set(langs)) != len(langs) or weights.ndim != 1 or
                len(weights) != len(langs) or not np.isfinite(weights).all() or
                (weights < 0).any() or not np.isfinite(weights.sum()) or weights.sum() <= 0):
            raise ValueError('Require unique languages and finite nonnegative weights with positive sum')
        if canonical_order:
            order = sorted(range(len(langs)), key=lambda i: (
                DOMAIN_ORDER.index(langs[i]) if langs[i] in DOMAIN_ORDER else len(DOMAIN_ORDER), langs[i]))
            langs, weights = [langs[i] for i in order], weights[order]
        self.streams = [TokenStream(manifest['streams'][lang]['train'] if manifest else
                                    train_files(data, lang), seq_len) for lang in langs]
        paths = [p for stream in self.streams for p in stream.paths]
        if len(paths) != len(set(paths)):
            raise ValueError('Duplicate training path across domains')
        self.langs = langs
        self.weights = weights / weights.sum()
        self.seed = seed
        self.seq_len = seq_len
        self._perm_cache = {}

    def describe(self):
        return {lang: {'files': len(s.paths), 'tokens': s.num_tokens, 'windows': s.num_windows}
                for lang, s in zip(self.langs, self.streams)}

    def _perm(self, lang_i, epoch):
        key = (lang_i, epoch)
        if key not in self._perm_cache:
            rng = np.random.default_rng([self.seed, lang_i, epoch])
            self._perm_cache = {k: v for k, v in self._perm_cache.items() if k[0] != lang_i}
            self._perm_cache[key] = rng.permutation(self.streams[lang_i].num_windows)
        return self._perm_cache[key]

    def plan(self, num_samples, max_data_passes=None):
        """Generate the FULL schedule, even when a caller will pause after a few steps."""
        if num_samples < 0:
            raise ValueError('num_samples must be nonnegative')
        rng = np.random.default_rng([self.seed, 999])
        lang_ids = rng.choice(len(self.langs), size=num_samples, p=self.weights).astype(np.int32)
        positions = np.zeros(num_samples, dtype=np.int64)
        for i in range(len(self.langs)):
            mask = lang_ids == i
            positions[mask] = np.arange(mask.sum())
        self.lang_ids, self.positions = lang_ids, positions
        stats = self.usage(num_samples)
        if max_data_passes is not None:
            if not np.isfinite(max_data_passes) or max_data_passes <= 0:
                raise ValueError('max_data_passes must be finite and positive')
            shortages = {lang: {'required_windows': s['windows'],
                                 'available_windows': self.streams[i].num_windows,
                                 'missing_windows': s['windows'] - int(max_data_passes * self.streams[i].num_windows)}
                         for i, (lang, s) in enumerate(stats.items()) if s['data_passes'] > max_data_passes}
            if shortages:
                raise ValueError(f'Full schedule exceeds max_data_passes={max_data_passes}: {shortages}')
        return {lang: s['data_passes'] for lang, s in stats.items()}

    def usage(self, num_samples):
        if not 0 <= num_samples <= len(self.lang_ids):
            raise ValueError('Sampler cursor outside schedule')
        counts = np.bincount(self.lang_ids[:num_samples], minlength=len(self.langs))
        return {lang: {'windows': int(counts[i]), 'tokens': int(counts[i]) * self.seq_len,
                       'data_passes': float(counts[i] / stream.num_windows)}
                for i, (lang, stream) in enumerate(zip(self.langs, self.streams))}

    def sample(self, k):
        i = int(self.lang_ids[k])
        stream = self.streams[i]
        epoch, pos = divmod(int(self.positions[k]), stream.num_windows)
        return stream.get(int(self._perm(i, epoch)[pos]))

    def batch(self, start, size):
        return torch.stack([self.sample(k) for k in range(start, start + size)])


def val_batches(data_dir, lang, seq_len, num_windows, batch_size, manifest=None):
    """Fixed leading windows of a validation stream, for comparable eval across runs."""
    stream = TokenStream([validation_path(data_dir, lang, manifest)], seq_len)
    n = min(num_windows, stream.num_windows)
    for s in range(0, n, batch_size):
        yield torch.stack([stream.get(i) for i in range(s, min(s + batch_size, n))])


def load_meta(data_dir):
    return json.loads((Path(data_dir) / 'meta.json').read_text())
