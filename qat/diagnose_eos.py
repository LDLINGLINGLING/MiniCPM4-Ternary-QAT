"""Diagnose whether a checkpoint puts probability on EOS at the answer boundary.

Pure inference, no training. For each development example the full prompt+answer+EOS
sequence is teacher-forced; at the position that predicts the final EOS we record the
EOS logit rank, probability and margin against the top token. A short greedy generation
then measures whether the model actually emits EOS on its own.

Usage:
  python3 -I qat/diagnose_eos.py --ckpt runs/.../model.pt --precision ternary \
      --out reports/.../eos_qat.json --limit 200 --gen-limit 64
"""
import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sft.gsm8k_data import atomic_json, encode_example
from qat.quant import convert_to_ternary, set_alpha, set_quantize, ternary_modules

ROOT = Path(__file__).resolve().parents[1]


def load(model_path, ckpt, precision, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32,
                                                 trust_remote_code=True)
    model.config.use_cache = True
    convert_to_ternary(model)
    if ckpt:
        state = torch.load(ckpt, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
    set_alpha(model, 1.0)
    set_quantize(model, precision == 'ternary')
    return tok, model.to(device=device, dtype=torch.bfloat16).eval()


def embedding_stats(model, stops):
    """Are the stop-token rows under-weighted relative to the embedding table?"""
    with torch.no_grad():
        emb = model.get_input_embeddings().weight.float()
        norms = emb.norm(dim=1)
        out = {'mean_row_norm': norms.mean().item(), 'p90_row_norm': torch.quantile(norms, 0.9).item(),
               'median_row_norm': norms.median().item()}
        for name, tid in stops.items():
            out[f'{name}_row_norm'] = norms[tid].item()
            out[f'{name}_ratio_to_median'] = (norms[tid] / norms.median()).item()
        return out


@torch.no_grad()
def boundary_stats(tok, model, rows, device, max_len, stops):
    """Rank/probability of each stop token where the answer ends.

    The official tokenizer has two stop tokens: </s> (id 2) and <|im_end|>
    (73440). Evaluation counts either as a stop, so both are measured.
    """
    records = []
    for row in rows:
        example = encode_example(tok, row, max_len)
        ids = torch.tensor([example['input_ids']], device=device)
        # Flash attention requires an explicit mask; use_cache must be off because
        # this model only accepts the new Cache interface.
        logits = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                       use_cache=False).logits[0].float()
        pos = len(example['input_ids']) - 2
        scores = logits[pos]
        probs = torch.softmax(scores, dim=-1)
        best = int(scores.argmax())
        per_stop = {}
        for name, tid in stops.items():
            per_stop[name] = {
                'rank': int((scores > scores[tid]).sum().item()) + 1,
                'prob': float(probs[tid]),
                'margin': float(scores[tid] - scores.max()),
            }
        records.append({**{f'{n}_{k}': v for n, d in per_stop.items() for k, v in d.items()},
                        'top1_is_stop': best in set(stops.values()),
                        'top_token': best, 'answer_tokens': example['answer_tokens']})

    summary = {'n': len(records),
               'top1_is_stop_rate': sum(r['top1_is_stop'] for r in records) / len(records)}
    for name in stops:
        ranks = sorted(r[f'{name}_rank'] for r in records)
        summary[name] = {
            'top1_rate': sum(r[f'{name}_rank'] == 1 for r in records) / len(records),
            'top5_rate': sum(r[f'{name}_rank'] <= 5 for r in records) / len(records),
            'median_rank': statistics.median(ranks),
            'p90_rank': ranks[int(len(ranks) * 0.9)],
            'median_prob': statistics.median(r[f'{name}_prob'] for r in records),
            'median_margin': statistics.median(r[f'{name}_margin'] for r in records),
        }
    summary['examples'] = records[:40]
    return summary


@torch.no_grad()
def generation_stats(tok, model, rows, device, max_len, max_new, stops):
    """Does the model emit either stop token by itself within a short budget?"""
    stop_ids = sorted(set(stops.values()))
    stopped = 0
    positions = []
    for row in rows:
        example = encode_example(tok, row, max_len)
        prompt = [i for i, label in zip(example['input_ids'], example['labels']) if label == -100]
        ids = torch.tensor([prompt], device=device)
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=stop_ids, pad_token_id=0)
        new = out[0, len(prompt):].tolist()
        hit = next((j for j, t in enumerate(new) if t in stop_ids), None)
        if hit is not None:
            stopped += 1
            positions.append(hit)
    return {'n': len(rows), 'stopped': stopped, 'stop_rate': stopped / len(rows),
            'stop_tokens': stop_ids,
            'median_stop_position': statistics.median(positions) if positions else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'))
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--precision', choices=['ternary', 'float'], required=True)
    parser.add_argument('--data', default=str(ROOT / 'data/gsm8k_sft_v1/dev.jsonl'))
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--gen-limit', type=int, default=64)
    parser.add_argument('--max-len', type=int, default=2048)
    parser.add_argument('--max-new', type=int, default=64)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.data).read_text(encoding='utf-8').splitlines()]
    tok, model = load(args.model, args.ckpt, args.precision, args.device)
    stops = {'eos': tok.eos_token_id, 'endoftext': 2} if tok.eos_token_id != 2 else {'eos': 2}
    result = {'ckpt': str(Path(args.ckpt).resolve()), 'precision': args.precision,
              'stop_tokens': stops,
              'embedding': embedding_stats(model, stops),
              'boundary': boundary_stats(tok, model, rows[:args.limit], args.device, args.max_len, stops),
              'generation': generation_stats(tok, model, rows[:args.gen_limit], args.device,
                                             args.max_len, args.max_new, stops)}
    atomic_json(Path(args.out), result)
    b = result['boundary']
    print(json.dumps({k: v for k, v in result.items() if k != 'boundary'}, indent=2))
    print(json.dumps({k: b[k] for k in b if k != 'examples'}, indent=2))


if __name__ == '__main__':
    main()
