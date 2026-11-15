"""Pre-LN (the baseline): h <- h + F(LN(h)).

This file is also the reference for the residual interface. A residual's entry file defines

    class Residual(nn.Module):
        def __init__(self, dim: int, num_sublayers: int): ...
        def forward(self, x0, sublayers) -> Tensor: ...

x0 is the normalized token embedding, (B, T, D). sublayers is the list [attn_0, mlp_0, attn_1, mlp_1, ...],
each a pure F: (B, T, D) -> (B, T, D) with no norm and no residual add. forward returns a (B, T, D) tensor,
which goes through the frozen final RMSNorm and lm_head. The residual owns every norm in between.

Optional hooks (defaults shown in train_gpt.py):
    optim_groups() -> list[dict]   AdamW groups for this module's parameters (default: one group, lr 0.015, wd 0.001)
    init_(sublayers)               runs under no_grad after the frozen backbone init, e.g. a depth-dependent rescale
    compile_mode: str = "model"    "model" compiles the whole GPT; "sublayer" compiles each sublayer + the head

The step count and the Muon lr of the sublayers' matrices are command-line flags (--steps, --lr), not hooks.
"""
from torch import nn

from model import RMSNorm


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))

    def forward(self, x, sublayers):
        for norm, f in zip(self.norms, sublayers):
            x = x + f(norm(x))
        return x
