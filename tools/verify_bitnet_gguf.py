#!/usr/bin/env python3
"""Verify native I2_S_ROW/G32 GGUF weights against FP32 HF masters."""

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--gguf', type=Path, required=True)
    p.add_argument('--hf-dir', type=Path, required=True)
    p.add_argument('--granularity', choices=('row', 'g32'), required=True)
    p.add_argument('--llama-dir', type=Path, default=ROOT / 'bitnet.cpp/3rdparty/llama.cpp')
    p.add_argument('--qat-root', type=Path, default=ROOT)
    p.add_argument('--output', type=Path)
    return p.parse_args()


def hf_name(name):
    if name == 'token_embd.weight':
        return 'model.embed_tokens.weight'
    if name == 'output.weight':
        return 'lm_head.weight'
    if name == 'output_norm.weight':
        return 'model.norm.weight'
    if not name.startswith('blk.'):
        return None
    parts = name.split('.')
    layer = parts[1]
    suffix = {
        'attn_q.weight': 'self_attn.q_proj.weight',
        'attn_k.weight': 'self_attn.k_proj.weight',
        'attn_v.weight': 'self_attn.v_proj.weight',
        'attn_output.weight': 'self_attn.o_proj.weight',
        'attn_norm.weight': 'input_layernorm.weight',
        'ffn_down.weight': 'mlp.down_proj.weight',
        'ffn_gate.weight': 'mlp.gate_proj.weight',
        'ffn_up.weight': 'mlp.up_proj.weight',
        'ffn_norm.weight': 'post_attention_layernorm.weight',
    }['.'.join(parts[2:])]
    return f'model.layers.{layer}.{suffix}'


def permute_rows(x, heads):
    return x.reshape(heads, 2, x.shape[0] // heads // 2, *x.shape[1:]).swapaxes(1, 2).reshape(x.shape)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.llama_dir / 'gguf-py'))
    from gguf import GGUFReader, GGMLQuantizationType

    spec = importlib.util.spec_from_file_location('qat_quant_reference', args.qat_root / 'qat/quant.py')
    quant = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(quant)
    quant.set_quant_group(None if args.granularity == 'row' else 32)

    import torch
    config = json.loads((args.hf_dir / 'config.json').read_text())
    reader = GGUFReader(str(args.gguf))
    expected_type = GGMLQuantizationType.I2_S_ROW if args.granularity == 'row' else GGMLQuantizationType.I2_S_G32
    quantized = f32_exact = code_mismatches = nonboundary_mismatches = 0
    max_scale_error = max_scale_rel_error = max_weight_error = max_matching_weight_error = max_boundary_error = 0.0
    worst = []
    mismatch_details = []
    seen_hf = set()

    with safe_open(str(args.hf_dir / 'model.safetensors'), framework='pt', device='cpu') as hf:
        for tensor in reader.tensors:
            name = tensor.name
            source_name = hf_name(name)
            if name.startswith('rope_factors_'):
                key = 'long_factor' if name == 'rope_factors_long.weight' else 'short_factor'
                expected = np.asarray(config['rope_scaling'][key], dtype=np.float32)
                assert tensor.tensor_type == GGMLQuantizationType.F32
                assert np.array_equal(tensor.data, expected), name
                f32_exact += 1
                continue
            assert source_name is not None, name
            seen_hf.add(source_name)
            master = hf.get_tensor(source_name)
            if tensor.tensor_type == GGMLQuantizationType.F32:
                assert np.array_equal(tensor.data, master.numpy()), name
                f32_exact += 1
                continue

            assert tensor.tensor_type == expected_type, (name, tensor.tensor_type)
            k, rows = tensor.shape.tolist()
            groups = 1 if args.granularity == 'row' else k // 32
            row_bytes = k // 4 + 4 * groups
            assert tuple(master.shape) == (rows, k), (name, master.shape, tensor.shape)
            assert tensor.n_bytes == rows * row_bytes, name
            packed = tensor.data.reshape(rows, row_bytes)
            shifts = np.array([6, 4, 2, 0], dtype=np.uint8)
            codes = ((packed[:, :k // 4, None] >> shifts) & 3).astype(np.int8).reshape(rows, k) - 1
            scales = packed[:, k // 4:].copy().view('<f4').reshape(rows, groups)

            ref_codes, ref_scales = quant.ternary_quantize(master)
            ref_master = master
            ref_scales = ref_scales.reshape(rows, groups)
            if name.endswith(('attn_q.weight', 'attn_k.weight')):
                heads = config['num_attention_heads'] if name.endswith('attn_q.weight') else config['num_key_value_heads']
                ref_codes = permute_rows(ref_codes, heads)
                ref_scales = permute_rows(ref_scales, heads)
                ref_master = permute_rows(ref_master, heads)
            ref_codes = ref_codes.reshape(rows, k).numpy()
            ref_scales = ref_scales.numpy()
            mismatches = int(np.count_nonzero(codes != ref_codes))
            code_mismatches += mismatches
            if mismatches:
                mr, mi = np.nonzero(codes != ref_codes)
                master_np = ref_master.numpy()
                ms = ref_scales[mr, mi // 32 if groups > 1 else 0]
                boundary_errors = np.abs(np.abs(master_np[mr, mi]) / ms - 0.5)
                max_boundary_error = max(max_boundary_error, float(np.max(boundary_errors)))
                nonboundary_mismatches += int(np.count_nonzero(boundary_errors > 1e-5))
            scale_error = float(np.max(np.abs(scales - ref_scales)))
            actual = (codes.reshape(rows, groups, -1) * scales[:, :, None]).reshape(rows, k)
            expected = (ref_codes.reshape(rows, groups, -1) * ref_scales[:, :, None]).reshape(rows, k)
            weight_error = float(np.max(np.abs(actual - expected)))
            max_scale_error = max(max_scale_error, scale_error)
            max_scale_rel_error = max(max_scale_rel_error, float(np.max(np.abs(scales - ref_scales) / np.maximum(np.abs(ref_scales), 1e-30))))
            max_weight_error = max(max_weight_error, weight_error)
            if mismatches < codes.size:
                max_matching_weight_error = max(max_matching_weight_error, float(np.max(np.abs(actual - expected)[codes == ref_codes])))
            worst.append((scale_error, name, mismatches))
            for r, i in np.argwhere(codes != ref_codes)[:8]:
                mismatch_details.append({
                    'tensor': name, 'row': int(r), 'column': int(i),
                    'master': float(ref_master[r, i]), 'native_code': int(codes[r, i]),
                    'reference_code': int(ref_codes[r, i]),
                    'native_scale': float(scales[r, i // 32 if groups > 1 else 0]),
                    'reference_scale': float(ref_scales[r, i // 32 if groups > 1 else 0]),
                })
            assert np.allclose(scales, ref_scales, rtol=1e-5, atol=1e-8), f'{name}: scale error {scale_error}'
            quantized += 1

        assert seen_hf == set(hf.keys()), f'HF/GGUF tensor mismatch: {set(hf.keys()) ^ seen_hf}'

    assert quantized == 168, quantized
    assert f32_exact == 53, f32_exact
    result = {
        'gguf': str(args.gguf), 'granularity': args.granularity,
        'file_bytes': args.gguf.stat().st_size, 'tensor_count': len(reader.tensors),
        'quantized_tensors': quantized, 'exact_f32_tensors': f32_exact,
        'ternary_code_mismatches': code_mismatches,
        'half_threshold_code_differences': code_mismatches - nonboundary_mismatches,
        'nonboundary_code_mismatches': nonboundary_mismatches,
        'max_half_threshold_normalized_error': max_boundary_error,
        'max_scale_abs_error': max_scale_error, 'max_scale_relative_error': max_scale_rel_error,
        'max_weight_abs_error': max_weight_error,
        'max_matching_code_weight_abs_error': max_matching_weight_error,
        'code_mismatch_examples': mismatch_details[:20],
        'largest_scale_errors': [
            {'tensor': name, 'scale_abs_error': error, 'code_mismatches': mismatches}
            for error, name, mismatches in sorted(worst, reverse=True)[:5]
        ],
        'status': 'passed' if nonboundary_mismatches == 0 else 'failed',
    }
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    if nonboundary_mismatches:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
