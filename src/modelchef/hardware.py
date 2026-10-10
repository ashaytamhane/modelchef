"""Hardware discovery and explicit, reproducible runtime resolution."""

import platform
import shutil
import subprocess
import sys


def detect_hardware(progress=None):
    def announce(message):
        if progress:
            progress(message)

    announce('Inspecting Python, operating system and architecture.')
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "os": platform.system(), "architecture": platform.machine(),
            "torch_installed": False, "cuda_available": False,
            "mps_available": False, "backend": "cpu", "bf16_supported": False}
    try:
        announce('Importing PyTorch; the first import can take several seconds.')
        import torch
    except ImportError:
        announce('PyTorch is not installed; reporting the available environment.')
        return info
    info.update(torch_installed=True, torch_version=torch.__version__,
                cuda_build=torch.version.cuda, hip_build=torch.version.hip)
    announce('Checking CUDA/ROCm and Apple MPS availability.')
    info["cuda_available"] = torch.cuda.is_available()
    info["mps_available"] = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available())
    if info["cuda_available"]:
        announce('Querying GPU devices, precision support and available memory.')
        info["backend"] = "rocm" if torch.version.hip else "cuda"
        info["bf16_supported"] = torch.cuda.is_bf16_supported()
        info["gpu_count"] = torch.cuda.device_count()
        info['compute_capability'] = list(torch.cuda.get_device_capability())
        info["gpus"] = [{"name": torch.cuda.get_device_name(i),
                         "memory_bytes": torch.cuda.get_device_properties(i).total_memory}
                        for i in range(torch.cuda.device_count())]
        free, total = torch.cuda.mem_get_info()
        info.update(free_memory_bytes=free, total_memory_bytes=total)
    elif info["mps_available"]:
        info["backend"] = "mps"
    announce(f"Detected backend: {info['backend']}.")
    return info


def resolve_runtime(runtime, hardware):
    if not isinstance(runtime, dict):
        raise TypeError("runtime must be a mapping")
    accelerator = runtime.get("accelerator", "auto")
    precision = runtime.get("precision", "auto")
    if accelerator not in {"auto", "cpu", "cuda", "rocm", "mps"}:
        raise ValueError("accelerator must be auto, cpu, cuda, rocm, or mps")
    if precision not in {"auto", "fp32", "bf16", "fp16"}:
        raise ValueError("precision must be auto, fp32, bf16, or fp16")
    selected = hardware["backend"] if accelerator == "auto" else accelerator
    if selected != "cpu" and selected != hardware["backend"]:
        raise ValueError(f"Requested {selected} is unavailable; detected {hardware['backend']}")
    if precision == "auto":
        precision = "bf16" if selected in {"cuda", "rocm"} and hardware[
            "bf16_supported"] else "fp32"
    if precision == "bf16" and not (
        selected in {"cuda", "rocm"} and hardware["bf16_supported"]
    ):
        raise ValueError("bf16 is not verified on the selected backend; use fp32")
    if precision == "fp16" and selected not in {"cuda", "rocm"}:
        raise ValueError("fp16 training is currently supported on CUDA/ROCm only; use fp32")
    return {**runtime, "accelerator": selected,
            "device": "cuda" if selected == "rocm" else selected,
            "precision": precision, "requires_grad_scaler": precision == "fp16",
            "micro_batch_size": runtime.get("micro_batch_size", 1),
            "hardware": hardware}


def setup_plan(backend="auto", index_url=None, torch_version=None):
    if backend not in {"auto", "cpu", "cuda", "rocm", "mps"}:
        raise ValueError("Unknown installation backend")
    system = platform.system()
    driver = None
    if shutil.which("nvidia-smi"):
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=False)
        if result.returncode == 0 and result.stdout.strip():
            driver = result.stdout.strip().splitlines()[0]
    if backend == "auto":
        if driver:
            backend = "cuda"
        elif system == "Darwin":
            backend = "mps"
        elif shutil.which("rocminfo") or shutil.which("rocm-smi"):
            backend = "rocm"
        else:
            backend = "cpu"
    if backend == "mps" and system != "Darwin":
        raise ValueError("MPS installation requires macOS")
    if backend == "rocm" and system != "Linux":
        raise ValueError("This setup supports ROCm installation on Linux only")
    if backend == "rocm" and not index_url:
        raise ValueError("ROCm requires --index-url matching the installed ROCm/GPU support matrix")
    if backend == "cuda" and not index_url:
        if not driver or int(driver.split('.')[0]) < 580:
            raise ValueError("Default CUDA 13.0 setup requires NVIDIA driver 580+; "
                             "select a compatible official --index-url")
        index_url = "https://download.pytorch.org/whl/cu130"
    if backend == "cpu" and system != "Darwin" and not index_url:
        index_url = "https://download.pytorch.org/whl/cpu"
    command = ["uv", "pip", "install", "--python", sys.executable,
               f"torch=={torch_version or '2.13.0'}", "numpy==2.3.5"]
    # NumPy is not consistently available on the PyTorch index; add PyPI as a named extra.
    if index_url:
        command += ["--index", index_url, "--index-strategy", "first-index"]
    return {"backend": backend, "driver": driver, "command": command,
            "python_supported": (3, 11) <= sys.version_info[:2] < (3, 13),
            "note": "Installs packages only; drivers and OS prerequisites must already exist. "
                    "Verify the resulting backend with doctor and smoke."}
