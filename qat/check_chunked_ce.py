"""Check that train.py's chunked CE matches the full-logit CE (loss and gradients).

Single GPU, one fixed batch from data/ufw_v1, ternary QAT at alpha=1 under BF16 autocast.
"""
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_pipeline.sampling import MixedSampler  # noqa: E402
from qat.quant import convert_to_ternary, set_alpha  # noqa: E402
from qat.train import FinalHidden, chunked_ce, lm_logits, load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def grads(model):
    return {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}


def main():
    device = torch.device('cuda')
    torch.backends.cuda.matmul.allow_tf32 = True
    model = load_model(str(ROOT / 'MiniCPM4-0.5B-fjzcuwlf'), device, torch.float32)
    convert_to_ternary(model)
    set_alpha(model, 1.0)
    model.train()
    sampler = MixedSampler(str(ROOT / 'data/ufw_v1'), ['zh', 'en'], [0.5, 0.5], 2048, 1234)
    sampler.plan(4)
    batch = sampler.batch(0, 4).to(device)
    inputs, labels = batch[:, :-1], batch[:, 1:]
    result = {'tied_lm_head': model.lm_head.weight is model.model.embed_tokens.weight}

    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = lm_logits(model, inputs)
        full = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
    full.backward()
    g_full = grads(model)
    model.zero_grad(set_to_none=True)
    del logits
    # Repeat the full path: flash-attention backward is not bitwise deterministic, so this
    # sets the noise floor the chunked-vs-full difference should be compared against.
    with torch.autocast('cuda', dtype=torch.bfloat16):
        logits = lm_logits(model, inputs)
        F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1)).backward()
    g_repeat = grads(model)
    model.zero_grad(set_to_none=True)
    del logits
    noise = [((g_full[n] - g_repeat[n]).norm() / g_full[n].norm().clamp_min(1e-30)).item() for n in g_full]
    result['full_repeat_rel_err_max'] = max(noise)
    result['full_repeat_rel_err_median'] = sorted(noise)[len(noise) // 2]

    torch.cuda.reset_peak_memory_stats()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        chunk = chunked_ce(FinalHidden(model)(inputs), model.lm_head.weight, labels, 8192)
    chunk.backward()
    g_chunk = grads(model)

    rel = {n: ((g_full[n] - g_chunk[n]).norm() / g_full[n].norm().clamp_min(1e-30)).item() for n in g_full}
    worst = sorted(rel.items(), key=lambda kv: -kv[1])[:5]
    result.update({'loss_full': full.item(), 'loss_chunked': chunk.item(),
                   'loss_abs_diff': abs(full.item() - chunk.item()),
                   'grad_params_full': len(g_full), 'grad_params_chunked': len(g_chunk),
                   'grad_rel_err_max': worst[0][1], 'grad_rel_err_worst5': worst,
                   'grad_rel_err_median': sorted(rel.values())[len(rel) // 2]})
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
