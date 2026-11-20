"""Compare alpha=1 QAT training forward, fixed export, and cached decoding."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qat.train import load_model
from qat.quant import convert_to_ternary, set_alpha, set_quantize, ternary_modules, ternary_dequant


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    from transformers import AutoTokenizer, DynamicCache
    tok = AutoTokenizer.from_pretrained(args.model)
    records = [json.loads(l) for l in Path(args.data).read_text().splitlines()[:4]]
    texts = [f"Question: {r['question']}\nAnswer: {r['answer']}" for r in records]
    tok.pad_token_id = 0
    tok.padding_side = 'left'
    enc = tok(texts, return_tensors='pt', padding=True).to('cuda')
    # Keep short windows, but preserve the real left-padding masks.
    enc = {k: v[:, -384:] for k, v in enc.items() if k in ('input_ids', 'attention_mask')}
    pos = enc['attention_mask'].cumsum(-1) - 1
    pos.masked_fill_(enc['attention_mask'] == 0, 1)
    model = load_model(args.model, 'cuda', torch.float32).eval()
    convert_to_ternary(model)
    state = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'], strict=True)
    del state
    set_alpha(model, 1.0)
    fixed = copy.deepcopy(model)
    with torch.no_grad():
        for m in ternary_modules(fixed):
            m.weight.copy_(ternary_dequant(m.weight.float()))
    set_quantize(fixed, False)

    def forward(m, x, mask, positions, **kwargs):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            return m(input_ids=x, attention_mask=mask, position_ids=positions, **kwargs)

    a = forward(model, enc['input_ids'], enc['attention_mask'], pos, use_cache=False).logits
    fp32_fixed = forward(fixed, enc['input_ids'], enc['attention_mask'], pos, use_cache=False).logits
    export_check = {'max_abs': (a - fp32_fixed).abs().max().item(),
                    'mean_abs': (a - fp32_fixed).abs().mean().item()}
    del fp32_fixed
    # Isolate fp32 cancellation in W + (Q-W), especially BF16 rounding ties.
    with torch.no_grad():
        for src, dst in zip(ternary_modules(model), ternary_modules(fixed)):
            q = ternary_dequant(src.weight.float())
            dst.weight.copy_(src.weight + (q - src.weight))
    reconstructed = forward(fixed, enc['input_ids'], enc['attention_mask'], pos, use_cache=False).logits
    export_check['ste_reconstructed_max_abs'] = (a - reconstructed).abs().max().item()
    del reconstructed
    with torch.no_grad():
        for src, dst in zip(ternary_modules(model), ternary_modules(fixed)):
            dst.weight.copy_(ternary_dequant(src.weight.float()))
    fixed.to(dtype=torch.bfloat16)
    b = forward(fixed, enc['input_ids'], enc['attention_mask'], pos, use_cache=False).logits
    valid = enc['attention_mask'].bool()
    diff = (a[valid] - b[valid]).abs()
    shift_valid = valid[:, 1:] & valid[:, :-1]
    def ce(logits):
        return F.cross_entropy(logits[:, :-1][shift_valid], enc['input_ids'][:, 1:][shift_valid]).item()
    report = {'fp32_export': export_check, 'forward': {'max_abs': diff.max().item(), 'mean_abs': diff.mean().item(),
                          'argmax_agreement': (a[valid].argmax(-1) == b[valid].argmax(-1)).float().mean().item(),
                          'ce_fake': ce(a), 'ce_fixed': ce(b)}}
    del a, b, diff
    # Greedy continuation in the exact same batch, without cache for either model.
    xa = enc['input_ids'].clone()
    xb = xa.clone()
    mask = enc['attention_mask'].clone()
    seq_a, seq_b = [], []
    for _ in range(8):
        positions = mask.cumsum(-1) - 1
        positions.masked_fill_(mask == 0, 1)
        na = forward(model, xa, mask, positions, use_cache=False).logits[:, -1].argmax(-1)
        nb = forward(fixed, xb, mask, positions, use_cache=False).logits[:, -1].argmax(-1)
        seq_a.append(na.tolist()); seq_b.append(nb.tolist())
        xa = torch.cat([xa, na[:, None]], 1); xb = torch.cat([xb, nb[:, None]], 1)
        mask = torch.cat([mask, torch.ones_like(mask[:, :1])], 1)
    report['greedy'] = {'fake': seq_a, 'fixed': seq_b, 'equal': seq_a == seq_b}
    full = forward(fixed, enc['input_ids'], enc['attention_mask'], pos, use_cache=False).logits[:, -1]
    cache = DynamicCache()
    forward(fixed, enc['input_ids'][:, :-1], enc['attention_mask'][:, :-1], pos[:, :-1],
            use_cache=True, past_key_values=cache)
    cached = forward(fixed, enc['input_ids'][:, -1:], enc['attention_mask'], pos[:, -1:],
                     use_cache=True, past_key_values=cache).logits[:, -1]
    report['cache'] = {'max_abs': (full - cached).abs().max().item(),
                       'mean_abs': (full - cached).abs().mean().item(),
                       'argmax_equal': bool(torch.equal(full.argmax(-1), cached.argmax(-1)))}
    single_checks = []
    for i in range(len(texts)):
        ids = enc['input_ids'][i][valid[i]][None]
        one = forward(fixed, ids, torch.ones_like(ids), torch.arange(ids.shape[1], device='cuda')[None],
                      use_cache=False).logits[:, -1]
        single_checks.append({'max_abs': (one[0] - full[i]).abs().max().item(),
                              'argmax_equal': one.argmax(-1).item() == full[i].argmax(-1).item()})
    report['single_vs_batch'] = single_checks
    report['model'] = args.model
    report['ckpt'] = args.ckpt
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
