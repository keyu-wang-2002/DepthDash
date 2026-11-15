"""Full Attention Residuals (Full AttnRes): every sublayer reads a softmax-weighted mixture of the embedding
and all preceding sublayer outputs, instead of their plain sum.

    v_0 = x0,  v_i = F_i(LN_i(h_i)),  h_i = sum_{j<i} a_ij v_j,  a_i. = softmax_j(w_i . RMSNorm(v_j))

with one learned pseudo-query w_i per sublayer (zero-init, so every read starts as a uniform average) and one
more for the output. The mixing is recomputed in the backward pass, so memory stays O(L) instead of O(L^2).
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import RMSNorm


def aggregate(query, *values):
    stacked = torch.stack(values, dim=-2)  # (B, T, n, D)
    logits = F.rms_norm(stacked, (stacked.size(-1),)) @ query.type_as(stacked)
    weights = logits.float().softmax(dim=-1).type_as(stacked)
    return (weights.unsqueeze(-1) * stacked).sum(dim=-2)


def attend(query, values):
    if torch.is_grad_enabled():
        return checkpoint(aggregate, query, *values, use_reentrant=False)
    return aggregate(query, *values)


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        self.queries = nn.Parameter(torch.zeros(num_sublayers + 1, dim))  # last row reads out to the head

    def forward(self, x, sublayers):
        values = [x]
        for i, (norm, f) in enumerate(zip(self.norms, sublayers)):
            values.append(f(norm(attend(self.queries[i], values))))
        return attend(self.queries[-1], values)
