"""Small decoder trained from random initialization: RoPE, GQA, RMSNorm, SwiGLU."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from ashayllm.execution import attention_context, custom_swiglu


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.dim = c.hidden_size // c.attention_heads
        self.q = nn.Linear(c.hidden_size, c.hidden_size, bias=c.bias)
        self.k = nn.Linear(c.hidden_size, self.dim * c.kv_heads, bias=c.bias)
        self.v = nn.Linear(c.hidden_size, self.dim * c.kv_heads, bias=c.bias)
        self.out = nn.Linear(c.hidden_size, c.hidden_size, bias=c.bias)
        self.register_buffer("frequencies", 1 / c.rope_theta ** (
            torch.arange(0, self.dim, 2).float() / self.dim), persistent=False)

    def rope(self, x):
        angles = torch.outer(torch.arange(x.shape[-2], device=x.device).float(), self.frequencies)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        b, s, _ = x.shape
        q = self.rope(self.q(x).view(b, s, self.c.attention_heads, self.dim).transpose(1, 2))
        k = self.rope(self.k(x).view(b, s, self.c.kv_heads, self.dim).transpose(1, 2))
        v = self.v(x).view(b, s, self.c.kv_heads, self.dim).transpose(1, 2)
        repeat = self.c.attention_heads // self.c.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        with attention_context(self.c.attention_kernel):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.c.attention_dropout if self.training else 0)
        return self.out(y.transpose(1, 2).contiguous().view(b, s, -1))


class Block(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.config = c
        self.attention_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.ffn_norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.attention = Attention(c)
        self.gate = nn.Linear(c.hidden_size, c.intermediate_size, bias=c.bias)
        self.up = nn.Linear(c.hidden_size, c.intermediate_size, bias=c.bias)
        self.down = nn.Linear(c.intermediate_size, c.hidden_size, bias=c.bias)
        self.dropout = nn.Dropout(c.residual_dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        y = self.ffn_norm(x)
        gate, up = self.gate(y), self.up(y)
        activation = (custom_swiglu(self.config.custom_swiglu_path, gate, up)
                      if self.config.custom_swiglu_path else F.silu(gate) * up)
        return x + self.dropout(self.down(activation))


class Decoder(nn.Module):
    def __init__(self, c, vocab_size):
        super().__init__()
        self.config = c
        self.embedding = nn.Embedding(vocab_size, c.hidden_size)
        self.blocks = nn.ModuleList([Block(c) for _ in range(c.layers)])
        self.norm = RMSNorm(c.hidden_size, c.norm_eps)
        self.head = nn.Linear(c.hidden_size, vocab_size, bias=False)
        self.apply(self.initialize)
        if c.tie_embeddings:
            self.head.weight = self.embedding.weight

    def initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.config.context_length:
            raise ValueError("Input exceeds configured context_length")
        x = self.embedding(ids)
        for block in self.blocks:
            x = checkpoint(block, x, use_reentrant=False) if (
                self.training and self.config.gradient_checkpointing) else block(x)
        return self.head(self.norm(x))
