"""Fast model for Block AttnRes (baselines/block_attnres.py): each read, and the pre-norm of the sublayer behind
it, run as one fused Triton kernel (attnres_kernel.py, from DepthBench). CUDA only.

    python check_equivalence.py --track deep --residual baselines/block_attnres.py --model baselines/fast/block_attnres.py
"""
import torch

import model
from attnres_kernel import fused_attnres


def read(query, key_gains, values, norm=None):
    """The softmax mixture of values, followed by norm (the next sublayer's pre-norm) if given."""
    return fused_attnres(query, values, key_gains, None if norm is None else norm.gains,
                         rms_eps=torch.finfo(values[0].dtype).eps)  # F.rms_norm's default eps, as in the reference


class GPT(model.GPT):
    compile_mode = "sublayer"  # the kernel runs eagerly between the compiled sublayers

    def forward(self, inputs, targets):
        R = self.residual
        blocks, partial = [self.norm1(self.embed(inputs))], None
        for i, (norm, f) in enumerate(zip(R.norms, self.sublayer_fns())):
            sources = blocks if partial is None else blocks + [partial]
            out = f(read(R.queries[i], R.key_gains[i], sources, norm))
            partial = out if partial is None else partial + out
            if (i + 1) % R.block_size == 0:
                blocks, partial = blocks + [partial], None
        sources = blocks if partial is None else blocks + [partial]
        return self.head_loss(read(R.queries[-1], R.key_gains[-1], sources), targets)
