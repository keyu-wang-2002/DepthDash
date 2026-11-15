"""Sandwich-LN: h <- h + LN_out(F(LN_in(h)))."""
from torch import nn

from model import RMSNorm


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms_in = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        self.norms_out = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))

    def forward(self, x, sublayers):
        for norm_in, norm_out, f in zip(self.norms_in, self.norms_out, sublayers):
            x = x + norm_out(f(norm_in(x)))
        return x
