"""The frozen backbone: everything in the GPT except the residual connections.

    GPT(inputs) = lm_head(RMSNorm(residual(RMSNorm(embed(inputs)), sublayers)))

`sublayers` alternates [attn_0, mlp_0, attn_1, mlp_1, ...]. Each one is a pure map F: (B, T, D) -> (B, T, D)
with no norm in front and no residual add behind: where norms go, how sublayer outputs are combined,
and what reaches the head are the residual's job (see baselines/preln.py for the interface).

Descends from modded-nanogpt's track 3 trainer (records/track_3_optimization/train_gpt_simple.py upstream).
"""
from functools import partial

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x: Tensor):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        # half-truncated RoPE (w/ base freq tuning)
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim: int):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.q = Linear(dim, dim)
        self.k = Linear(dim, dim)
        self.v = Linear(dim, dim)
        self.proj = Linear(dim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), scale=0.12, is_causal=True).transpose(1, 2)
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = Linear(dim, 4 * dim)
        self.proj = Linear(4 * dim, dim)

    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu().square())


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int, head_dim: int, residual_cls):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.sublayers = nn.ModuleList()
        for _ in range(num_layers):
            self.sublayers.append(CausalSelfAttention(model_dim, head_dim))
            self.sublayers.append(MLP(model_dim))
        self.residual = residual_cls(model_dim, 2 * num_layers)
        self.proj = Linear(model_dim, vocab_size)
        self.norm1 = RMSNorm(model_dim)
        self.norm2 = RMSNorm(model_dim)
        self.activation_checkpointing = False  # memory only; does not change the math

    def sublayer_fns(self):
        if self.activation_checkpointing and torch.is_grad_enabled():
            return [partial(checkpoint, f, use_reentrant=False) for f in self.sublayers]
        return list(self.sublayers)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        x = self.residual(x, self.sublayer_fns())
        return self.head_loss(x, targets)

    def head_loss(self, x: Tensor, targets: Tensor):
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15**2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")
