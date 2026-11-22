"""Full Attention Residuals (Full AttnRes): every sublayer reads a softmax-weighted mixture of the embedding
and all preceding sublayer outputs, instead of their plain sum.

    v_0 = x0,  v_i = F_i(LN_i(h_i)),  h_i = sum_{j<i} a_ij v_j,  a_i. = softmax_j(q_i . RMSNorm_i(v_j))

with one learned pseudo-query q_i (zero-init, so every read starts as a uniform average) and one key-norm gain
per read, plus one more read for the output. As in DepthBench.

This is the plain PyTorch reference; it recomputes the mixing in the backward pass, so memory stays O(L)
instead of O(L^2). baselines/fast/attnres.py is the fast model, with DepthBench's fused Triton kernel.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import RMSNorm


def aggregate(query, key_gains, *values):
    stacked = torch.stack(values, dim=-2)  # (B, T, n, D)
    keys = F.rms_norm(stacked, (stacked.size(-1),), weight=key_gains.type_as(stacked))
    weights = (keys @ query.type_as(stacked)).float().softmax(dim=-1).type_as(stacked)
    return (weights.unsqueeze(-1) * stacked).sum(dim=-2)


def read(query, key_gains, values):
    if torch.is_grad_enabled():
        return checkpoint(aggregate, query, key_gains, *values, use_reentrant=False)
    return aggregate(query, key_gains, *values)


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        # one row per sublayer read; the last row reads out to the head
        self.queries = nn.Parameter(torch.zeros(num_sublayers + 1, dim))
        self.key_gains = nn.Parameter(torch.ones(num_sublayers + 1, dim))

    def forward(self, x, sublayers):
        values = [x]
        for i, (norm, f) in enumerate(zip(self.norms, sublayers)):
            values.append(f(norm(read(self.queries[i], self.key_gains[i], values))))
        return read(self.queries[-1], self.key_gains[-1], values)
