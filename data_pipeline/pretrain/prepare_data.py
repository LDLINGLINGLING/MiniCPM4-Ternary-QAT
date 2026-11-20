"""Tokenize Ultra-FineWeb parquet shards into flat uint32 token streams.

Each document becomes [BOS] + tokens + [</s>]. Documents are shuffled with a
fixed seed, the first `--val-docs` go to the validation stream, the rest to
the training stream. Training later samples fixed-length windows from these
streams (see data_pipeline/sampling.py).
"""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

_tokenizer = None


def _init(tokenizer_path):
    # One Rust thread per worker; the container caps total threads (pids.max).
    os.environ['RAYON_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    global _tokenizer
    from transformers import AutoTokenizer
    _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)


def _encode(texts):
    ids = _tokenizer(texts, add_special_tokens=False)['input_ids']
    bos, eos = _tokenizer.bos_token_id, 2  # </s> separates pre-training documents
    parts = []
    for x in ids:
        parts.append(np.asarray([bos] + x + [eos], dtype=np.uint32))
    return parts


def tokenize_split(texts, out_path, pool, batch_docs):
    batches = [texts[i:i + batch_docs] for i in range(0, len(texts), batch_docs)]
    total_tokens = 0
    doc_lens = []
    with open(out_path, 'wb') as f:
        for parts in pool.imap(_encode, batches, chunksize=1):
            for p in parts:
                f.write(p.tobytes())
                doc_lens.append(len(p))
            total_tokens += sum(len(p) for p in parts)
    return total_tokens, np.asarray(doc_lens, dtype=np.int64)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', required=True)
    parser.add_argument('--input', action='append', required=True,
                        help='lang=path.parquet, repeatable')
    parser.add_argument('--output', required=True)
    parser.add_argument('--val-docs', type=int, default=2000)
    parser.add_argument('--min-chars', type=int, default=50)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--workers', type=int, default=96)
    parser.add_argument('--batch-docs', type=int, default=500)
    args = parser.parse_args(argv)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    meta = {'tokenizer': str(Path(args.tokenizer).resolve()), 'seed': args.seed,
            'val_docs': args.val_docs, 'min_chars': args.min_chars,
            'format': 'flat uint32; each doc = [BOS=1] + tokens + [EOS=2]', 'splits': {}}

    with mp.get_context('spawn').Pool(args.workers, initializer=_init, initargs=(args.tokenizer,)) as pool:
        for spec in args.input:
            lang, path = spec.split('=', 1)
            t0 = time.time()
            texts = pq.read_table(path, columns=['content']).column('content').to_pylist()
            n_raw = len(texts)
            texts = [t for t in texts if isinstance(t, str) and len(t.strip()) >= args.min_chars]
            order = np.random.default_rng(args.seed).permutation(len(texts))
            texts = [texts[i] for i in order]
            print(f'{lang}: {n_raw} docs read, {len(texts)} kept, {time.time() - t0:.0f}s', flush=True)
            for split, subset in (('val', texts[:args.val_docs]), ('train', texts[args.val_docs:])):
                t1 = time.time()
                path_out = out / f'{lang}_{split}.bin'
                n_tok, lens = tokenize_split(subset, path_out, pool, args.batch_docs)
                np.save(out / f'{lang}_{split}_doclens.npy', lens)
                meta['splits'][f'{lang}_{split}'] = {
                    'source': str(Path(path).resolve()), 'docs': len(subset), 'tokens': int(n_tok),
                    'chars': int(sum(len(t) for t in subset)),
                    'mean_doc_tokens': float(lens.mean()) if len(lens) else 0.0}
                print(f'  {lang}_{split}: {len(subset)} docs, {n_tok:,} tokens, {time.time() - t1:.0f}s', flush=True)
            del texts

    (out / 'meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2))
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
