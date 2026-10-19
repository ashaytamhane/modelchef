"""Small decoder trained from random initialization: RoPE, GQA, RMSNorm, SwiGLU."""
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint

from modelchef.execution import attention_context, custom_swiglu


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return normalized.to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.attention_config = model_config
        self.dim = model_config.hidden_size // model_config.attention_heads
        self.q = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=model_config.bias)
        self.k = nn.Linear(model_config.hidden_size, self.dim * model_config.kv_heads, bias=model_config.bias)
        self.v = nn.Linear(model_config.hidden_size, self.dim * model_config.kv_heads, bias=model_config.bias)
        self.out = nn.Linear(model_config.hidden_size, model_config.hidden_size, bias=model_config.bias)
        self.register_buffer("frequencies", 1 / model_config.rope_theta ** (
            torch.arange(0, self.dim, 2).float() / self.dim), persistent=False)

    def rope(self, x):
        angles = torch.outer(torch.arange(x.shape[-2], device=x.device).float(), self.frequencies)
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        even, odd = x[..., ::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), -1).flatten(-2)

    def forward(self, x):
        batch_size, sequence_length, _ = x.shape
        q = self.rope(self.q(x).view(batch_size, sequence_length, self.attention_config.attention_heads, self.dim).transpose(1, 2))
        k = self.rope(self.k(x).view(batch_size, sequence_length, self.attention_config.kv_heads, self.dim).transpose(1, 2))
        v = self.v(x).view(batch_size, sequence_length, self.attention_config.kv_heads, self.dim).transpose(1, 2)
        repeat = self.attention_config.attention_heads // self.attention_config.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        with attention_context(self.attention_config.attention_kernel):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True,
                dropout_p=self.attention_config.attention_dropout if self.training else 0)
        return self.out(y.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1))


class Block(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.config = model_config
        self.attention_norm = RMSNorm(model_config.hidden_size, model_config.norm_eps)
        self.ffn_norm = RMSNorm(model_config.hidden_size, model_config.norm_eps)
        self.attention = Attention(model_config)
        self.gate = nn.Linear(model_config.hidden_size, model_config.intermediate_size, bias=model_config.bias)
        self.up = nn.Linear(model_config.hidden_size, model_config.intermediate_size, bias=model_config.bias)
        self.down = nn.Linear(model_config.intermediate_size, model_config.hidden_size, bias=model_config.bias)
        self.dropout = nn.Dropout(model_config.residual_dropout)
        if model_config.type == 'decoder_moe':
            from modelchef.moe import SparseExperts
            del self.gate, self.up, self.down
            self.moe = SparseExperts(model_config)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.attention_norm(x)))
        y = self.ffn_norm(x)
        if self.config.type == 'decoder_moe':
            return x + self.dropout(self.moe(y))
        gate, up = self.gate(y), self.up(y)
        activation = (custom_swiglu(self.config.custom_swiglu_path, gate, up)
                      if self.config.custom_swiglu_path else F.silu(gate) * up)
        return x + self.dropout(self.down(activation))


class Decoder(nn.Module):
    def __init__(self, model_config, vocab_size):
        super().__init__()
        self.config = model_config
        self.embedding = nn.Embedding(vocab_size, model_config.hidden_size)
        self.blocks = nn.ModuleList([Block(model_config) for _ in range(model_config.layers)])
        self.norm = RMSNorm(model_config.hidden_size, model_config.norm_eps)
        self.head = nn.Linear(model_config.hidden_size, vocab_size, bias=False)
        self.apply(self.initialize)
        if model_config.tie_embeddings:
            self.head.weight = self.embedding.weight

    def initialize(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=self.config.init_std)
            if getattr(module, "bias", None) is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.config.context_length:
            raise ValueError("Input exceeds configured context_length")
        hidden_states = self.embedding(ids)
        for block in self.blocks:
            hidden_states = checkpoint(block, hidden_states, use_reentrant=False) if (
                self.training and self.config.gradient_checkpointing) else block(hidden_states)
        return self.head(self.norm(hidden_states))

    @property
    def auxiliary_loss(self):
        losses = [block.moe.auxiliary_loss for block in self.blocks
                  if hasattr(block, 'moe') and block.moe.auxiliary_loss is not None]
        return sum(losses) if losses else None
