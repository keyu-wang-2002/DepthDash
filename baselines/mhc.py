"""Manifold-constrained Hyper-Connections (mHC), n = 4 streams.

Like HC (baselines/hc.py), but pre = sigmoid(.), post = 2 * sigmoid(.) are non-negative, and the residual map A
is projected onto doubly-stochastic matrices by 20 Sinkhorn-Knopp iterations (in log space). The coefficients
come from RMSNorm of the flattened 4 streams. The streams are averaged at the end.
Ported from nanogpt-lab (records/20261011_mhc_hc_depth).

On CUDA the connection runs as Liger-Kernel's fused Triton mHC kernels (coefficients + Sinkhorn, read,
write), as in DepthBench. Elsewhere, or with FUSED = False, the plain PyTorch path below runs.
fused/check_fused.py compares the two.

compile_mode = "sublayer": a whole-model graph with 20 Sinkhorn iterations per sublayer is too large to compile.
The plain connection math is two pure functions with the same shapes at every sublayer, so each compiles once.
"""
import torch
import torch.nn.functional as F
from torch import nn

from model import RMSNorm

try:
    from liger_kernel.transformers.functional import liger_mhc_coeffs, liger_mhc_post_res, liger_mhc_pre
except ImportError:  # no liger-kernel: plain PyTorch only
    liger_mhc_coeffs = None

FUSED = liger_mhc_coeffs is not None  # set False to run the plain PyTorch reference

STREAMS = 4
SINKHORN_ITERATIONS = 20
ROUTING_LR = 0.001
ROUTING_DYNAMIC_WD = 0.1


def sinkhorn_log(logits, iterations):
    """Row then column normalization in log space; A[source, destination]. Staying in log space until the
    end keeps tiny entries from overflowing the backward pass."""
    log_matrix = logits - logits.amax(dim=(-2, -1), keepdim=True)
    for _ in range(iterations):
        log_matrix = F.log_softmax(log_matrix, dim=-1)
        log_matrix = F.log_softmax(log_matrix, dim=-2)
    return log_matrix.exp()


@torch.compile(dynamic=False)
def mhc_read(x, dynamic, scales, bias):
    flat = x.float().flatten(-2)
    coefficients = F.rms_norm(flat, (flat.size(-1),), eps=1e-6) @ dynamic
    pre = (scales[0] * coefficients[..., :STREAMS] + bias[:STREAMS]).sigmoid()
    post = 2 * (scales[1] * coefficients[..., STREAMS:2 * STREAMS] + bias[STREAMS:2 * STREAMS]).sigmoid()
    logits = (scales[2] * coefficients[..., 2 * STREAMS:] + bias[2 * STREAMS:]).unflatten(-1, (STREAMS, STREAMS))
    hres = sinkhorn_log(logits, SINKHORN_ITERATIONS)
    branch_input = (pre.unsqueeze(-1) * x.float()).sum(dim=-2).to(x.dtype)
    return branch_input, post, hres


@torch.compile(dynamic=False)
def mhc_write(x, update, post, hres):
    return (hres.transpose(-1, -2) @ x.float() + post.unsqueeze(-1) * update.float().unsqueeze(-2)).to(x.dtype)


class ManifoldHyperConnection(nn.Module):
    def __init__(self, dim: int, sublayer_index: int):
        super().__init__()
        self.dynamic = nn.Parameter(torch.zeros(STREAMS * dim, 2 * STREAMS + STREAMS ** 2))
        self.scales = nn.Parameter(torch.full((3,), 0.01))
        read = torch.full((STREAMS,), -8.0)
        read[sublayer_index % STREAMS] = 8.0
        residual = torch.full((STREAMS, STREAMS), -8.0)
        residual.fill_diagonal_(0)
        self.bias = nn.Parameter(torch.cat((read, torch.zeros(STREAMS), residual.flatten())))

    def forward(self, x, f, norm):
        if FUSED and x.is_cuda:
            # Liger's same Sinkhorn (row-softmax, then 20 row/column normalizations) yields our A[source, dest];
            # its write takes h_res[dest, source], hence the transpose.
            pre, post, hres = liger_mhc_coeffs(x, self.dynamic, self.bias, *self.scales.unbind(),
                                               tmax=SINKHORN_ITERATIONS, rms_eps=1e-6, post_mult=2.0)
            return liger_mhc_post_res(x, f(norm(liger_mhc_pre(x, pre))), post, hres.mT)
        branch_input, post, hres = mhc_read(x, self.dynamic, self.scales, self.bias)
        return mhc_write(x, f(norm(branch_input)), post, hres)


class Residual(nn.Module):
    compile_mode = "sublayer"

    def __init__(self, dim: int, num_sublayers: int):
        super().__init__()
        self.norms = nn.ModuleList(RMSNorm(dim) for _ in range(num_sublayers))
        self.connections = nn.ModuleList(ManifoldHyperConnection(dim, i) for i in range(num_sublayers))

    def optim_groups(self):
        dynamic = [c.dynamic for c in self.connections]
        static = [p for c in self.connections for p in (c.scales, c.bias)]
        return [dict(params=self.norms.parameters()),
                dict(params=dynamic, lr=ROUTING_LR, weight_decay=ROUTING_DYNAMIC_WD),
                dict(params=static, lr=ROUTING_LR, weight_decay=0.0)]

    def forward(self, x, sublayers):
        x = x.unsqueeze(-2).expand(-1, -1, STREAMS, -1)
        for norm, connection, f in zip(self.norms, self.connections, sublayers):
            x = connection(x, f, norm)
        return x.mean(dim=-2)
