"""Trusted local CUDA SwiGLU example, with a differentiable PyTorch wrapper."""
import hashlib
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import CUDA_HOME, load
    source = Path(__file__).parent
    sources = [source / 'swiglu.cpp', source / 'swiglu.cu']
    fingerprint = hashlib.sha256(b''.join(p.read_bytes() for p in sources)).hexdigest()[:12]
    name = 'modelchef_swiglu_' + fingerprint
    build = Path.cwd() / 'artifacts/cuda-extensions' / name
    build.mkdir(parents=True, exist_ok=True)
    # NVIDIA runtime wheels may provide only the versioned shared library.
    # Keep the linker alias in the ignored build directory, not the installed toolkit.
    linker_flags = []
    if CUDA_HOME:
        library_dir = Path(CUDA_HOME) / 'lib'
        candidates = sorted(library_dir.glob('libcudart.so.*'))
        if not (library_dir / 'libcudart.so').exists() and candidates:
            alias = build / 'libcudart.so'
            if not alias.exists():
                alias.symlink_to(candidates[0].resolve())
            linker_flags = [f'-L{build}']
    return load(name=name, sources=[str(p) for p in sources],
                build_directory=str(build), extra_cuda_cflags=['-O3'],
                extra_ldflags=linker_flags, verbose=False)


class SwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate, up):
        ctx.save_for_backward(gate, up)
        return extension().forward(gate.contiguous(), up.contiguous())

    @staticmethod
    def backward(ctx, gradient):
        gate, up = ctx.saved_tensors
        return tuple(extension().backward(gradient.contiguous(), gate.contiguous(), up.contiguous()))


def swiglu(gate, up):
    return SwiGLU.apply(gate, up)
