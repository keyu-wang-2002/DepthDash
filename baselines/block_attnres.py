"""Block Attention Residuals (Block AttnRes): AttnRes (baselines/attnres.py) over block summaries.

The 2L sublayers are cut into ~8 blocks of b = round(2L / 8). Inside a block, sublayer outputs are summed
into a partial block sum; once the block ends, the sum is frozen. Each sublayer reads a softmax mixture over
[embedding, frozen block sums..., current partial sum], so the mixing costs O(blocks), not O(L).

This is the plain PyTorch reference. baselines/fast/block_attnres.py is the fast model, with DepthBench's fused
Triton kernel.
"""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from model import RMSNorm

NUM_BLOCKS = 8


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
        self.block_size = max(1, round(num_sublayers / NUM_BLOCKS))
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        # one row per sublayer read; the last row reads out to the head
        self.queries = nn.Parameter(torch.zeros(num_sublayers + 1, dim))
        self.key_gains = nn.Parameter(torch.ones(num_sublayers + 1, dim))

    def forward(self, x, sublayers):
        blocks, partial = [x], None
        for i, (norm, f) in enumerate(zip(self.norms, sublayers)):
            sources = blocks if partial is None else blocks + [partial]
            out = f(norm(read(self.queries[i], self.key_gains[i], sources)))
            partial = out if partial is None else partial + out
            if (i + 1) % self.block_size == 0:
                blocks, partial = blocks + [partial], None
        sources = blocks if partial is None else blocks + [partial]
        return read(self.queries[-1], self.key_gains[-1], sources)
