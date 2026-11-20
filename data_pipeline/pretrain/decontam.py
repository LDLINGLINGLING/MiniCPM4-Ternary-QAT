"""Read-only benchmark exclusion; question stems only, never answers/options.

Offsets are into the NFKC/casefold/whitespace-normalized document, not raw text.
Exact substring matching and 13-unit overlap can over-filter boilerplate and miss
semantic paraphrases; neither removes contamination in the pretrained model.
The optional pyahocorasick package accelerates full-stem substring matching.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

NGRAM = 13
# English words (including contractions), number runs, CJK single characters.
UNIT_RE = re.compile(r"[a-z]+(?:'[a-z]+)*|[0-9]+|[㐀-䶿一-鿿豈-﫿\U00020000-\U000323af]")
UNCOVERED = ['MATH', 'HumanEval', 'MBPP']


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def units(text):
    return [m.group() for m in UNIT_RE.finditer(normalize(text))]


def gram_key(values):
    return hashlib.blake2b('\x1f'.join(values).encode('utf-8'), digest_size=16).digest()


def discover_benchmarks(root):
    """Freeze all installed question files; enforce each required split exists."""
    base = Path(root) / 'eval_data'
    files = []
    for split in ('train', 'test'):
        paths = sorted((base / 'gsm8k').glob(f'{split}*.jsonl'))
        if not paths:
            raise ValueError(f'Missing GSM8K {split}')
        files.extend({'path': str(p.resolve()), 'benchmark': 'gsm8k', 'split': split,
                      'format': 'jsonl'} for p in paths)
    variants = []
    for directory in sorted(base.glob('gsm8k_variants*')):
        # Audit logs contain answers and review notes, NOT benchmark questions.
        variants.extend(p for p in sorted(directory.glob('*.jsonl'))
                        if p.name != 'review_audit.jsonl')
    if not variants:
        raise ValueError('Missing existing GSM8K variants')
    files.extend({'path': str(p.resolve()), 'benchmark': 'gsm8k_variants',
                  'split': p.stem, 'format': 'jsonl'} for p in variants)
    for benchmark, splits in [('mmlu', ('dev', 'val', 'test')),
                               ('cmmlu', ('dev', 'test')),
                               ('ceval', ('dev', 'val', 'test'))]:
        directory = base / benchmark
        if benchmark == 'mmlu':
            directory /= 'data'
        for split in splits:
            paths = sorted((directory / split).glob('*.csv'))
            if not paths:
                raise ValueError(f'Missing {benchmark} {split}')
            files.extend({'path': str(p.resolve()), 'benchmark': benchmark,
                          'split': split, 'format': 'csv'} for p in paths)
    return files


def questions(files):
    for spec in files:
        path = Path(spec['path'])
        count = 0
        with path.open(encoding='utf-8-sig', newline='') as handle:
            if spec['format'] == 'jsonl':
                rows = (json.loads(line) for line in handle if line.strip())
                for index, row in enumerate(rows):
                    text = row.get('question')
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f'{path}:{index}: missing question')
                    yield text, {'benchmark': spec['benchmark'], 'split': spec['split'],
                                 'sample_id': f'{path.name}:{row.get("pair_id", row.get("id", index))}'}
                    count += 1
            else:
                if spec['benchmark'] == 'mmlu':
                    rows = ({'question': row[0]} for row in csv.reader(handle) if row)
                else:
                    rows = csv.DictReader(handle)
                for index, row in enumerate(rows):
                    # CMMLU uses Question; C-Eval uses question and test has no answer.
                    text = row.get('question', row.get('Question'))
                    if not isinstance(text, str) or not text.strip():
                        raise ValueError(f'{path}:{index}: missing question')
                    yield text, {'benchmark': spec['benchmark'], 'split': spec['split'],
                                 'sample_id': f'{path.name}:{row.get("id", index)}'}
                    count += 1
        if not count:
            raise ValueError(f'Empty benchmark file: {path}')


class ExclusionIndex:
    """One canonical hit per rejected document, with benchmark/sample/position.

    Full stems use Aho-Corasick when installed. The bounded-memory fallback is
    deliberately simple for synthetic tests; production should install the native
    accelerator. N-grams use 128-bit hashes, avoiding Python tuples of 13 strings
    per indexed gram. A hash collision can only conservatively reject a document.
    """
    def __init__(self, records, require_fast=False):
        self.full = {}
        self.grams = {}
        self.sample_count = 0
        try:
            import ahocorasick
        except ImportError:
            ahocorasick = None
            if require_fast:
                raise RuntimeError('Production preparation requires pyahocorasick; '
                                   'install with python -I -m pip install pyahocorasick') from None
        self.vocabulary = {}
        self.gram_automaton = ahocorasick.Automaton() if ahocorasick else None
        has_grams = False
        for text, info in records:
            text = normalize(text)
            if not text:
                continue
            # Never add options/answers. Even a short question is kept as a whole.
            canonical = self.full.setdefault(text, dict(info))
            values = units(text)
            # Each distinct unit gets ONE Unicode symbol, not text concatenation:
            # this preserves unit boundaries without quadratic 13-string joins.
            encoded = ''.join(chr(self.vocabulary.setdefault(v, len(self.vocabulary) + 1))
                              for v in values)
            for start in range(len(values) - NGRAM + 1):
                if self.gram_automaton is not None:
                    key = encoded[start:start + NGRAM]
                    if key not in self.gram_automaton:
                        self.gram_automaton.add_word(key, canonical)
                        has_grams = True
                else:
                    self.grams.setdefault(gram_key(values[start:start + NGRAM]), canonical)
            self.sample_count += 1
        if has_grams:
            self.gram_automaton.make_automaton()
        else:
            self.gram_automaton = None
        self.automaton = None
        if ahocorasick is not None and self.full:
            automaton = ahocorasick.Automaton()
            for text, info in self.full.items():
                automaton.add_word(text, (len(text), info))
            automaton.make_automaton()
            self.automaton = automaton

    def match(self, text):
        text = normalize(text)
        if self.automaton is not None:
            found = next(self.automaton.iter(text), None)
            if found is not None:
                end, (length, info) = found
                return {**info, 'rule': 'full_question', 'position': end - length + 1}
        else:
            # Same earliest-ending match as the native automaton, including ties.
            best = None
            for question, info in self.full.items():
                start = text.find(question)
                if start >= 0:
                    key = (start + len(question), -len(question))
                    if best is None or key < best[0]:
                        best = (key, start, info)
            if best is not None:
                return {**best[2], 'rule': 'full_question', 'position': best[1]}
        matches = list(UNIT_RE.finditer(text))
        values = [m.group() for m in matches]
        if self.gram_automaton is not None:
            encoded = ''.join(chr(self.vocabulary.get(v, 0)) for v in values)
            found = next(self.gram_automaton.iter(encoded), None)
            if found is not None:
                end, info = found
                return {**info, 'rule': '13_unit', 'position': matches[end - NGRAM + 1].start()}
            return None
        for start in range(len(values) - NGRAM + 1):
            info = self.grams.get(gram_key(values[start:start + NGRAM]))
            if info is not None:
                return {**info, 'rule': '13_unit', 'position': matches[start].start()}
        return None
