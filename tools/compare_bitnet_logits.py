#!/usr/bin/env python3
"""Compare HF QAT prefill/cached-decode logits with llama-bitnet-eval output."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--hf-dir', type=Path, required=True)
    p.add_argument('--qat-root', type=Path, default=ROOT)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--native-logits', type=Path, required=True)
    p.add_argument('--granularity', choices=('row', 'g32'), required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--autocast', choices=('bf16', 'none'), default='bf16')
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args()


def metrics(native, reference, targets):
    difference = np.abs(native - reference)
    native_argmax = native.argmax(axis=1)
    reference_argmax = reference.argmax(axis=1)
    native_t = torch.from_numpy(native)
    reference_t = torch.from_numpy(reference)
    target_t = torch.from_numpy(targets.astype(np.int64))
    return {
        'positions': int(native.shape[0]),
        'max_abs_error': float(difference.max()),
        'mean_abs_error': float(difference.mean()),
        'rmse': float(np.sqrt(np.mean(np.square(difference.astype(np.float64))))),
        'argmax_agreement': float(np.mean(native_argmax == reference_argmax)),
        'native_cross_entropy': float(F.cross_entropy(native_t, target_t)),
        'hf_cross_entropy': float(F.cross_entropy(reference_t, target_t)),
        'native_target_accuracy': float(np.mean(native_argmax == targets)),
        'hf_target_accuracy': float(np.mean(reference_argmax == targets)),
    }


def main():
    args = parse_args()
    with args.native_logits.open('rb') as handle:
        magic = handle.read(8)
        if magic != b'BNELG1\0\0':
            raise ValueError(f'bad native logits magic: {magic!r}')
        seq_len, decode_steps, vocab = np.frombuffer(handle.read(12), dtype='<u4')
    native = np.memmap(args.native_logits, dtype='<f4', mode='r', offset=20,
                       shape=(int(seq_len + decode_steps), int(vocab)))
    tokens = np.fromfile(args.tokens, dtype='<u4', count=int(seq_len + decode_steps + 1)).astype(np.int64)
    if tokens.size != seq_len + decode_steps + 1:
        raise ValueError('token stream too short')

    spec = importlib.util.spec_from_file_location('qat_quant_reference', args.qat_root / 'qat/quant.py')
    quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant)
    quant.set_quant_group(None if args.granularity == 'row' else 32)
    from transformers import AutoModelForCausalLM
    from transformers.cache_utils import DynamicCache
    model = AutoModelForCausalLM.from_pretrained(
        args.hf_dir, dtype=torch.float32, trust_remote_code=True,
    ).to(args.device)
    replaced = quant.convert_to_ternary(model)
    if len(replaced) != 168:
        raise ValueError(f'expected 168 ternary projections, got {len(replaced)}')
    quant.set_alpha(model, 1.0)
    model.eval()

    reference_parts = []
    past = DynamicCache()
    with torch.inference_mode():
        autocast = torch.autocast('cuda', dtype=torch.bfloat16,
                                  enabled=args.autocast == 'bf16')
        with autocast:
            ids = torch.from_numpy(tokens[:seq_len]).to(args.device).unsqueeze(0)
            mask = torch.ones_like(ids)
            output = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
            reference_parts.append(output.logits.float().cpu())
            past = output.past_key_values
            for step in range(int(decode_steps)):
                ids = torch.from_numpy(tokens[seq_len + step:seq_len + step + 1]).to(args.device).unsqueeze(0)
                mask = torch.ones((1, seq_len + step + 1), dtype=torch.int64, device=args.device)
                output = model(input_ids=ids, attention_mask=mask, past_key_values=past, use_cache=True)
                reference_parts.append(output.logits.float().cpu())
                past = output.past_key_values
    reference = torch.cat(reference_parts, dim=1).squeeze(0).numpy()
    if reference.shape != native.shape:
        raise ValueError(f'logit shape mismatch: HF {reference.shape}, native {native.shape}')
    native_array = np.asarray(native)
    targets = tokens[1:]
    result = {
        'hf_dir': str(args.hf_dir), 'native_logits': str(args.native_logits),
        'granularity': args.granularity, 'device': args.device, 'autocast': args.autocast,
        'sequence_length': int(seq_len), 'decode_steps': int(decode_steps), 'vocab_size': int(vocab),
        'overall': metrics(native_array, reference, targets),
        'prefill': metrics(native_array[:seq_len], reference[:seq_len], targets[:seq_len]),
    }
    if decode_steps:
        result['cached_decode'] = metrics(native_array[seq_len:], reference[seq_len:], targets[seq_len:])
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text + '\n')


if __name__ == '__main__':
    main()
