"""Explicit optional Blackwell NVFP4 projection integration."""
import platform
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def prepare_bnb_libraries():
    """Make CUDA 13's wheel-provided JIT linker visible before loading bitsandbytes."""
    if platform.system() == 'Linux' and str(torch.version.cuda).startswith('13.'):
        import ctypes
        library = Path(torch.__file__).resolve().parents[1] / 'nvidia/cu13/lib/libnvJitLink.so.13'
        if library.exists():
            return ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
    return None


def require_nvfp4():
    if platform.system() != 'Linux' or not torch.cuda.is_available() or torch.version.hip:
        raise ValueError('NVFP4 requires Linux and an NVIDIA Blackwell CUDA GPU')
    if torch.cuda.get_device_capability()[0] < 10:
        raise ValueError('NVFP4 requires NVIDIA Blackwell (compute capability 10+)')
    try:
        import transformer_engine.pytorch as te
        from transformer_engine.common.recipe import NVFP4BlockScaling
    except ImportError as error:
        raise ValueError('Install the optional nvfp4 extra and a compatible Transformer Engine build') from error
    return te, NVFP4BlockScaling()


def apply_nvfp4(model):
    te, recipe = require_nvfp4()
    for name, layer in list(model.named_modules()):
        if isinstance(layer, torch.nn.Linear) and name != 'head':
            if layer.in_features % 16 or layer.out_features % 16:
                raise ValueError(f'{name}: NVFP4 projection dimensions must be divisible by 16')
            replacement = te.Linear(layer.in_features, layer.out_features,
                                    bias=layer.bias is not None, params_dtype=torch.float32,
                                    device='cuda')
            with torch.no_grad():
                replacement.weight.copy_(layer.weight)
                if layer.bias is not None:
                    replacement.bias.copy_(layer.bias)
            parent_name, child = name.rsplit('.', 1)
            setattr(model.get_submodule(parent_name), child, replacement)
    original = model.forward

    def forward(ids):
        with te.autocast(enabled=True, recipe=recipe):
            return original(ids)

    model.forward = forward
