"""LayerNorm Scaling (LNS): h <- h + F(LN(h) / sqrt(l)), with l = 1..L the layer index (shared by its
attention and MLP sublayers)."""
from torch import nn

from model import RMSNorm


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        self.scales = [(i // 2 + 1) ** -0.5 for i in range(num_sublayers)]

    def forward(self, x, sublayers):
        for norm, scale, f in zip(self.norms, self.scales, sublayers):
            x = x + f(norm(x) * scale)
        return x
