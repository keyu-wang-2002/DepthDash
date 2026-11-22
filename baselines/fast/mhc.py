"""Fast model for mHC (baselines/mhc.py): each connection runs as Liger-Kernel's fused Triton mHC kernels
(coefficients + Sinkhorn, read, write), as in DepthBench. CUDA only; needs liger-kernel (requirements.txt).

    python check_equivalence.py --track deep --residual baselines/mhc.py --model baselines/fast/mhc.py
"""
from liger_kernel.transformers.functional import liger_mhc_coeffs, liger_mhc_post_res, liger_mhc_pre

import model

STREAMS = 4
SINKHORN_ITERATIONS = 20


class GPT(model.GPT):
    compile_mode = "sublayer"  # the kernels run eagerly between the compiled sublayers

    def forward(self, inputs, targets):
        R = self.residual
        x = self.norm1(self.embed(inputs)).unsqueeze(-2).expand(-1, -1, STREAMS, -1).contiguous()
        for norm, c, f in zip(R.norms, R.connections, self.sublayer_fns()):
            # Liger's Sinkhorn (a row softmax, then row/column normalizations) yields the reference's
            # A[source, destination]; its write takes h_res[destination, source], hence the transpose.
            # sinkhorn_eps=0: the reference's exact Sinkhorn (entries are exp(.) > 0, so no division by zero).
            pre, post, hres = liger_mhc_coeffs(x, c.dynamic, c.bias, *c.scales.unbind(), allow_fp32=True,
                                               tmax=SINKHORN_ITERATIONS, rms_eps=1e-6, sinkhorn_eps=0.0,
                                               post_mult=2.0)
            x = liger_mhc_post_res(x, f(norm(liger_mhc_pre(x, pre))), post, hres.mT)
        return self.head_loss(x.mean(dim=-2), targets)
