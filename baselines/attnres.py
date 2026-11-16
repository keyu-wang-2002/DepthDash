"""Full Attention Residuals (Full AttnRes): every sublayer reads a softmax-weighted mixture of the embedding
and all preceding sublayer outputs, instead of their plain sum.

    v_0 = x0,  v_i = F_i(LN_i(h_i)),  h_i = sum_{j<i} a_ij v_j,  a_i. = softmax_j(q_i . RMSNorm_i(v_j))

with one learned pseudo-query q_i (zero-init, so every read starts as a uniform average) and one key-norm gain
per read, plus one more read for the output. As in DepthBench.

On CUDA, each read and the pre-norm LN_i behind it run as one fused Triton kernel (fused/attnres.py, from
DepthBench). Elsewhere, or with FUSED = False, the plain PyTorch path below runs; it recomputes the mixing in
the backward pass, so memory stays O(L) instead of O(L^2). fused/check_fused.py compares the two.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import RMSNorm

try:
    from fused.attnres import fused_attnres
except ImportError:  # no Triton: plain PyTorch only
    fused_attnres = None

FUSED = fused_attnres is not None  # set False to run the plain PyTorch reference


def aggregate(query, key_gains, *values):
    stacked = torch.stack(values, dim=-2)  # (B, T, n, D)
    keys = F.rms_norm(stacked, (stacked.size(-1),), weight=key_gains.type_as(stacked))
    weights = (keys @ query.type_as(stacked)).float().softmax(dim=-1).type_as(stacked)
    return (weights.unsqueeze(-1) * stacked).sum(dim=-2)


def read(query, key_gains, values, norm=None):
    """The softmax mixture of values, followed by norm (the next sublayer's pre-norm) if given."""
    if FUSED and values[0].is_cuda:
        return fused_attnres(query, values, key_gains, None if norm is None else norm.gains,
                             rms_eps=torch.finfo(values[0].dtype).eps)  # F.rms_norm's default eps
    if torch.is_grad_enabled():
        h = checkpoint(aggregate, query, key_gains, *values, use_reentrant=False)
    else:
        h = aggregate(query, key_gains, *values)
    return h if norm is None else norm(h)


class Residual(nn.Module):
    # the fused kernel runs eagerly between the compiled sublayers
    compile_mode = "sublayer" if FUSED else "model"

    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        # one row per sublayer read; the last row reads out to the head
        self.queries = nn.Parameter(torch.zeros(num_sublayers + 1, dim))
        self.key_gains = nn.Parameter(torch.ones(num_sublayers + 1, dim))

    def forward(self, x, sublayers):
        values = [x]
        for i, (norm, f) in enumerate(zip(self.norms, sublayers)):
            values.append(f(read(self.queries[i], self.key_gains[i], values, norm)))
        return read(self.queries[-1], self.key_gains[-1], values)
