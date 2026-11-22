"""A fast model for Pre-LN (baselines/preln.py), and a template for --model.

    python check_equivalence.py --track deep --residual baselines/preln.py --model baselines/fast/preln.py
    ./run.sh --track deep --residual baselines/preln.py --model baselines/fast/preln.py --steps N --lr LR

The Residual is the reference: it defines the math. A fast model is a `class GPT(model.GPT)` that computes the
same function any way it likes, including across the sublayer boundary, as long as it keeps the reference's
parameters (names and shapes) and passes check_equivalence.py.

Here each pre-norm gain g is folded into the input projections of the sublayer behind it:
rms_norm(x) * g @ W.T == rms_norm(x) @ (W * g).T, so the norm's multiply moves into the matmul. The saving is
tiny; the point is the pattern. The other fast models here use real kernels (e.g. attnres.py, mhc.py).
"""
import torch.nn.functional as F

import model


def linear(layer, x, in_gain=None):
    weight = layer.weight if in_gain is None else layer.weight * in_gain
    return F.linear(x, weight.type_as(x), layer.bias.type_as(x))


def attention(attn: model.CausalSelfAttention, x, in_gain):
    B, T = x.size(0), x.size(1)
    q = linear(attn.q, x, in_gain).view(B, T, attn.num_heads, attn.head_dim)
    k = linear(attn.k, x, in_gain).view(B, T, attn.num_heads, attn.head_dim)
    v = linear(attn.v, x, in_gain).view(B, T, attn.num_heads, attn.head_dim)
    q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
    q, k = attn.rotary(q), attn.rotary(k)
    y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
    return linear(attn.proj, y.contiguous().view(B, T, attn.num_heads * attn.head_dim))


def mlp(m: model.MLP, x, in_gain):
    return linear(m.proj, linear(m.fc, x, in_gain).relu().square())


class GPT(model.GPT):
    def forward(self, inputs, targets):
        x = self.norm1(self.embed(inputs))
        for i, (norm, sublayer) in enumerate(zip(self.residual.norms, self.sublayers)):
            h = F.rms_norm(x, (x.size(-1),))
            x = x + (attention if i % 2 == 0 else mlp)(sublayer, h, norm.gains)
        return self.head_loss(x, targets)
