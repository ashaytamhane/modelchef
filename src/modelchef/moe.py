"""Portable top-k sparse SwiGLU experts; correctness-first, without fused dispatch."""
import torch
import torch.nn.functional as F
from torch import nn


class Expert(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.bias)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.bias)

    def forward(self, inputs):
        return self.down(F.silu(self.gate(inputs)) * self.up(inputs))


class SparseExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
        self.auxiliary_loss = None

    def forward(self, inputs):
        flat = inputs.reshape(-1, inputs.shape[-1])
        probabilities = self.router(flat).float().softmax(-1)
        weights, indices = probabilities.topk(self.config.experts_per_token, dim=-1)
        weights = weights / weights.sum(-1, keepdim=True)
        output = torch.zeros_like(flat)
        for index, expert in enumerate(self.experts):
            tokens, slots = (indices == index).nonzero(as_tuple=True)
            if tokens.numel():
                contributions = expert(flat[tokens]) * weights[tokens, slots, None].to(flat.dtype)
                output = output.index_add(0, tokens, contributions.to(output.dtype))
        assignments = F.one_hot(indices, self.config.num_experts).float().mean((0, 1))
        self.auxiliary_loss = self.config.num_experts * (
            assignments * probabilities.mean(0)).sum() * self.config.router_aux_loss_coef
        return output.reshape_as(inputs)
