"""Ternary weight fake-quantization for QAT (BitCPM4 / BitNet b1.58 style).

Quantizer:
    s     = mean_j |W_j|                 over the elements of one group
    Q(W)  = s * clip(round(W_j / s), -1, 1)
where a "group" is either one whole output row (legacy `per-row`) or a block of
GROUP_SIZE consecutive input-dimension weights inside one row (BitCPM4).

Forward uses W + alpha * (Q(W) - W).detach(), i.e. a straight-through estimator;
alpha ramps 0 -> 1 for a gradual float->ternary transition and stays 1 after.

Granularity: BitCPM4-0.5B ships fake-quantized ternary weights in which every
block of 32 consecutive input-dimension elements shares a single magnitude
(verified against BitCPM4-0.5B-verified-Gf7CzOLq: 0.02% of groups violate the
one-magnitude rule at group 32, but 98.8% violate it at group 64). Its zero
fraction is 34.95%, which matches absmean (31.45% on the original MiniCPM4
weights) and rules out absmax (76.51%). So GROUP_SIZE = 32 is the default here.

Legacy runs trained with per-row absmean; call set_quant_group(None) before
loading one of those checkpoints so evaluation uses the same rule it was
trained with. Trainers record the granularity in `resolved_config.json`.
"""
import torch
import torch.nn.functional as F
from torch import nn

TARGET_SUFFIXES = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
GROUP_SIZE = 32
EPS = 1e-8

# None selects the legacy per-row quantizer used by runs created before the
# BitCPM4 port; an int selects per-group with that group size.
_quant_group = GROUP_SIZE


def set_quant_group(group_size):
    """Select quantization granularity: int for per-group, None for legacy per-row."""
    global _quant_group
    if group_size is not None and int(group_size) < 1:
        raise ValueError('group_size must be a positive int of None')
    _quant_group = None if group_size is None else int(group_size)


def get_quant_group():
    return _quant_group


def _grouped(w: torch.Tensor, group_size: int):
    """Reshape [rows, cols] into [rows, groups, group_size], padding the tail.

    Returns (grouped, mask) where mask is None when no padding was needed. The
    mask excludes zero padding from the mean so a short final group averages
    over real weights only.
    """
    rows, cols = w.shape
    pad = (-cols) % group_size
    x = F.pad(w, (0, pad)) if pad else w
    x = x.reshape(rows, -1, group_size)
    if not pad:
        return x, None
    mask = torch.ones(rows, x.size(1), group_size, dtype=x.dtype, device=x.device)
    mask[:, -1, group_size - pad:] = 0
    return x, mask


def ternary_quantize(w: torch.Tensor, group_size=None):
    """Return (codes, scale) for a 2-D weight.

    Per-group mode returns codes [rows, groups, group_size] and scale
    [rows, groups, 1]; legacy per-row mode returns codes [rows, cols] and scale
    [rows, 1]. Use ternary_dequant to get a [rows, cols] float tensor.
    """
    if w.ndim != 2:
        raise ValueError(f'ternary_quantize expects a 2-D weight, got shape {tuple(w.shape)}')
    g = _quant_group if group_size is None else group_size
    if g is None:
        scale = w.abs().mean(dim=1, keepdim=True).clamp(min=EPS)
        return (w / scale).round().clamp(-1, 1), scale
    g = int(g)
    x, mask = _grouped(w, g)
    if mask is None:
        scale = x.abs().mean(dim=2, keepdim=True)
    else:
        scale = (x.abs() * mask).sum(dim=2, keepdim=True) / mask.sum(dim=2, keepdim=True).clamp(min=1)
    scale = scale.clamp(min=EPS)
    return (x / scale).round().clamp(-1, 1), scale


def ternary_dequant(w: torch.Tensor, group_size=None) -> torch.Tensor:
    rows, cols = w.shape
    codes, scale = ternary_quantize(w, group_size)
    return (codes * scale).reshape(rows, -1)[:, :cols]


class TernaryLinear(nn.Linear):
    """nn.Linear whose forward weight is fake-quantized to ternary values."""

    def __init__(self, in_features, out_features, bias=False, device=None, dtype=None):
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.register_buffer('alpha', torch.ones((), dtype=torch.float32), persistent=False)
        self.quantize = True

    def forward(self, x):
        w = self.weight
        if self.quantize:
            w_q = ternary_dequant(w.float()).to(w.dtype)
            w = w + self.alpha.to(w.dtype) * (w_q - w).detach()
        return F.linear(x, w, self.bias)


def convert_to_ternary(model: nn.Module, suffixes=None):
    """Replace target nn.Linear modules inside decoder layers with TernaryLinear in place.

    If suffixes is given (a subset of TARGET_SUFFIXES), only those projection
    types are quantized; the rest stay as plain nn.Linear with float weights.
    """
    targets = set(suffixes) if suffixes is not None else set(TARGET_SUFFIXES)
    replaced = []
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f'{name}.{child_name}' if name else child_name
            if (type(child) is nn.Linear and child_name in targets and '.layers.' in f'.{full}'):
                new = TernaryLinear(child.in_features, child.out_features, bias=child.bias is not None,
                                    device=child.weight.device, dtype=child.weight.dtype)
                new.weight = child.weight
                if child.bias is not None:
                    new.bias = child.bias
                setattr(module, child_name, new)
                replaced.append(full)
    return replaced


def ternary_modules(model: nn.Module):
    return [m for m in model.modules() if isinstance(m, TernaryLinear)]


def set_alpha(model: nn.Module, alpha: float):
    for m in ternary_modules(model):
        m.alpha.fill_(alpha)


def set_quantize(model: nn.Module, enabled: bool):
    for m in ternary_modules(model):
        m.quantize = enabled


@torch.no_grad()
def quant_stats(model: nn.Module):
    """Fraction of zero codes and mean relative reconstruction error across ternary layers."""
    zeros = total = 0
    rel_err = []
    for m in ternary_modules(model):
        w = m.weight.float()
        codes, scale = ternary_quantize(w)
        zeros += (codes == 0).sum().item()
        total += codes.numel()
        rel_err.append(((codes * scale).reshape(w.shape) - w).norm().div(w.norm().clamp(min=EPS)).item())
    return {'zero_frac': zeros / max(total, 1),
            'rel_err': sum(rel_err) / max(len(rel_err), 1)}
