"""Explicit kernel/compiler choices and trusted local CUDA extension hooks."""
import hashlib
import importlib.util
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import torch


def attention_context(kernel):
    if kernel == 'auto':
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    names = {'math': 'MATH', 'flash': 'FLASH_ATTENTION', 'efficient': 'EFFICIENT_ATTENTION',
             'cudnn': 'CUDNN_ATTENTION'}
    return sdpa_kernel(getattr(SDPBackend, names[kernel]))


def configure_execution(model, training, device):
    if training.optimizer_kernel == 'fused' and device != 'cuda':
        raise ValueError('Fused optimizer selection currently requires CUDA/ROCm')
    if model.config.attention_kernel not in {'auto', 'math'} and device != 'cuda':
        raise ValueError('Selected optimized attention kernel requires CUDA/ROCm')
    if training.compile:
        options = {'backend': training.compile_backend}
        if training.compile_backend == 'inductor':
            options['mode'] = training.compile_mode
        # Compile the callable, preserving normal checkpoint parameter names.
        model.forward = torch.compile(model.forward, **options)
    return {'attention_kernel': model.config.attention_kernel, 'compile': training.compile,
            'compile_backend': training.compile_backend, 'compile_mode': training.compile_mode,
            'optimizer_kernel': training.optimizer_kernel, 'allow_tf32': training.allow_tf32,
            'note': 'Requested execution policy; profile/benchmark to establish actual kernels and speed.'}


def plugin_fingerprint(path):
    directory = Path(path).resolve()
    files = sorted(p for p in directory.iterdir() if p.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'})
    if not (directory / 'kernel.py').exists():
        raise ValueError('Custom CUDA directory requires kernel.py')
    return hashlib.sha256(b''.join(p.name.encode() + b'\0' + p.read_bytes() for p in files)).hexdigest()


@lru_cache(maxsize=8)
def _load_plugin(path, fingerprint):
    spec = importlib.util.spec_from_file_location('modelchef_cuda_' + fingerprint[:12],
                                                 Path(path) / 'kernel.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not callable(getattr(module, 'swiglu', None)):
        raise TypeError('Custom CUDA plugin must export swiglu(gate, up) with autograd support')
    return module


def custom_swiglu(path, gate, up):
    if gate.device.type != 'cuda':
        raise ValueError('Custom CUDA SwiGLU requires CUDA tensors; disable it for CPU/MPS')
    path = str(Path(path).resolve())
    return _load_plugin(path, plugin_fingerprint(path)).swiglu(gate, up)


def check_custom_kernel(path):
    # A correctness probe must not change the training run's configured RNG state.
    with torch.random.fork_rng():
        return _check_custom_kernel(path)


def _check_custom_kernel(path):
    if not torch.cuda.is_available():
        raise ValueError('Custom CUDA validation requires an available CUDA device')
    results = []
    for dtype in [torch.float32, torch.float16, torch.bfloat16]:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue
        torch.manual_seed(42)
        gate = torch.randn(2, 8, 64, device='cuda', dtype=dtype, requires_grad=True)
        up = torch.randn_like(gate, requires_grad=True)
        expected = torch.nn.functional.silu(gate) * up
        actual = custom_swiglu(path, gate, up)
        gradient = torch.randn_like(gate)
        reference_grad = torch.autograd.grad(expected, (gate, up), gradient)
        actual_grad = torch.autograd.grad(actual, (gate, up), gradient)
        tolerance = 0.03 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 1e-5
        if not torch.allclose(actual, expected, atol=tolerance, rtol=tolerance) or not all(
                torch.allclose(a, b, atol=tolerance, rtol=tolerance)
                for a, b in zip(actual_grad, reference_grad)):
            raise RuntimeError(f'Custom kernel forward/backward differs for {dtype}')
        results.append({'dtype': str(dtype), 'forward_backward': 'passed'})
    return {'source_sha256': plugin_fingerprint(path), 'checks': results,
            'note': 'Correctness probe only; no performance claim.'}
