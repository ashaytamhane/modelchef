"""Experimental software NVFP4-style fake quantization with straight-through gradients.

This simulates block-scaled E2M1 values; it does not pack weights, accelerate GEMMs,
or claim bitwise parity with NVIDIA/Transformer Engine hardware recipes.
"""
import torch
import torch.nn.functional as F


def fake_nvfp4(value, block_size=16):
    if value.shape[-1] % block_size:
        raise ValueError('FP4 fake quantization requires last dimension divisible by 16')
    original = value.float()
    blocks = original.reshape(*original.shape[:-1], -1, block_size)
    local = blocks.detach().abs().amax(-1, keepdim=True) / 6
    global_scale = (local.amax() / 448).clamp_min(1e-12)
    normalized = local / global_scale
    # Software E4M3-style positive scale rounding, including subnormal spacing.
    spacing = torch.pow(2.0, normalized.clamp_min(2 ** -9).log2().floor() - 3).clamp_min(2 ** -9)
    scale = ((normalized / spacing).round() * spacing).clamp(max=448) * global_scale
    scale = scale.clamp_min(1e-12)
    ladder = original.new_tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])
    magnitude = (blocks / scale).abs()
    closest = (magnitude.unsqueeze(-1) - ladder).abs().argmin(-1)
    quantized = (ladder[closest] * blocks.sign() * scale).reshape_as(original).to(value.dtype)
    return value + (quantized - value).detach()


class FakeFP4Linear(torch.nn.Linear):
    def forward(self, inputs):
        return F.linear(fake_nvfp4(inputs), fake_nvfp4(self.weight), self.bias)


def apply_fake_fp4(model):
    """Preserve parameter names/sharing and exclude the output head."""
    count = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, torch.nn.Linear) and not name.endswith(('head', 'lm_head')):
            if module.in_features % 16:
                raise ValueError(f'{name}: FP4 requires in_features divisible by 16')
            module.__class__ = FakeFP4Linear
            count += 1
    if not count:
        raise ValueError('No compatible linear layers found for FP4 simulation')
    return count
