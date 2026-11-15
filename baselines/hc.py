"""Hyper-Connections (HC), n = 4 residual streams with dynamic read / write / mixing coefficients.

For each sublayer, with streams x: (B, T, 4, D) and A[source, destination] the residual map:
    read:   u = sum_s pre[s] * x[s]
    write:  x[d] <- sum_s A[s, d] * x[s] + post[d] * F(LN(u))
pre, A = static + alpha_scale * tanh(RMSNorm(x) @ dynamic_alpha); post likewise. The embedding is copied into
all 4 streams and the streams are summed at the end. Ported from nanogpt-lab (records/20261011_mhc_hc_depth).
"""
import torch
import torch.nn.functional as F
from torch import nn

from model import RMSNorm

STREAMS = 4
ROUTING_LR = 0.001
ROUTING_DYNAMIC_WD = 0.1


class HyperConnection(nn.Module):
    def __init__(self, dim: int, sublayer_index: int):
        super().__init__()
        self.dynamic_alpha = nn.Parameter(torch.zeros(dim, STREAMS + 1))
        self.dynamic_beta = nn.Parameter(torch.zeros(dim))
        self.alpha_scale = nn.Parameter(torch.tensor(0.01))
        self.beta_scale = nn.Parameter(torch.tensor(0.01))
        read = torch.zeros(STREAMS)
        read[sublayer_index % STREAMS] = 1
        self.static_alpha = nn.Parameter(torch.cat((read[:, None], torch.eye(STREAMS)), dim=1))
        self.static_beta = nn.Parameter(torch.ones(STREAMS))

    def forward(self, x, f, norm):
        normalized = F.rms_norm(x.float(), (x.size(-1),), eps=1e-6)
        alpha = self.static_alpha + self.alpha_scale * (normalized @ self.dynamic_alpha).tanh()
        post = self.static_beta + self.beta_scale * (normalized @ self.dynamic_beta).tanh()
        pre, hres = alpha[..., 0], alpha[..., 1:]
        update = f(norm((pre.unsqueeze(-1) * x.float()).sum(dim=-2).to(x.dtype)))
        return (hres.transpose(-1, -2) @ x.float() + post.unsqueeze(-1) * update.float().unsqueeze(-2)).to(x.dtype)


class Residual(nn.Module):
    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        self.connections = nn.ModuleList(HyperConnection(dim, i) for i in range(num_sublayers))

    def optim_groups(self):
        dynamic = [p for n, p in self.connections.named_parameters() if n.split(".")[-1].startswith("dynamic")]
        static = [p for n, p in self.connections.named_parameters() if not n.split(".")[-1].startswith("dynamic")]
        return [dict(params=self.norms.parameters()),
                dict(params=dynamic, lr=ROUTING_LR, weight_decay=ROUTING_DYNAMIC_WD),
                dict(params=static, lr=ROUTING_LR, weight_decay=0.0)]

    def forward(self, x, sublayers):
        x = x.unsqueeze(-2).expand(-1, -1, STREAMS, -1)
        for norm, connection, f in zip(self.norms, self.connections, sublayers):
            x = connection(x, f, norm)
        return x.sum(dim=-2)
