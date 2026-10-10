"""Optional vLLM layer sharing the project's verified PyTorch environment."""

import os
import platform
import shutil
from pathlib import Path


def vllm_setup_plan(root, hardware, version="0.31.0"):
    root = Path(root).resolve()
    if platform.system() != "Linux" or hardware["backend"] != "cuda":
        raise ValueError("Automatic vLLM setup currently requires Linux/WSL NVIDIA CUDA. "
                         "CPU, ROCm, and Metal need their backend-specific vLLM installation.")
    env = root / ".venv"
    return {"environment": str(env), "version": version, "commands": [
        ["uv", "pip", "install", "--python", str(env / "bin/python"),
         f"vllm=={version}", "torch==2.13.0", "--torch-backend=cu130"],
    ], "c_compiler_available": bool(shutil.which("cc") or shutil.which("gcc")),
        "note": "Shared training/inference environment; CUDA 13.0 driver support "
        "and a C compiler (Ubuntu build-essential) required. "
        "Model export and supported architecture are required before serving."}


def serve_plan(config, root):
    if not isinstance(config, dict):
        raise TypeError("Inference config must be a mapping")
    root = Path(root).resolve()
    settings = config.get("inference", {})
    if not isinstance(settings, dict):
        raise TypeError("inference must be a mapping")
    raw_model = settings.get("model_path")
    if not isinstance(raw_model, str) or not raw_model:
        raise ValueError("inference.model_path must name a local Hugging Face model export")
    model = Path(raw_model)
    model = (root / model).resolve() if not model.is_absolute() else model.resolve()
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError("Local model export must exist and contain config.json")
    if not any(model.glob("*.safetensors")) and not (model / "pytorch_model.bin").is_file():
        raise ValueError("Local model export needs safetensors or pytorch_model.bin weights")
    if not (model / "tokenizer.json").is_file():
        raise ValueError("Local model export needs tokenizer.json")
    memory = settings.get("gpu_memory_utilization", 0.5)
    if isinstance(memory, bool) or not isinstance(memory, (int, float)) or not 0 < memory < 1:
        raise ValueError("gpu_memory_utilization must be between 0 and 1")
    length = settings.get("max_model_len", 2048)
    sequences = settings.get("max_num_seqs", 4)
    port = settings.get("port", 8000)
    for name, value in [("max_model_len", length), ("max_num_seqs", sequences), ("port", port)]:
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if port > 65535:
        raise ValueError("port must be at most 65535")
    dtype = settings.get("dtype", "auto")
    if dtype not in {"auto", "half", "float16", "bfloat16", "float", "float32"}:
        raise ValueError("Unsupported vLLM dtype")
    executable = root / ".venv/bin/vllm"
    command = [str(executable), "serve", str(model), "--host", "127.0.0.1",
               "--port", str(port), "--dtype", dtype,
               "--gpu-memory-utilization", str(memory), "--max-model-len", str(length),
               "--max-num-seqs", str(sequences)]
    if settings.get("enforce_eager", True):
        command.append("--enforce-eager")
    return {"command": command, "environment": {"HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1"},
            "endpoint": f"http://127.0.0.1:{port}/v1",
            "note": "Local weights only; architecture compatibility and memory fit "
                    "are verified by vLLM at startup. No trust-remote-code enabled."}


def launch_server(plan):
    import subprocess

    if not Path(plan["command"][0]).is_file():
        raise ValueError("vLLM environment is missing; run setup-vllm --apply first")
    subprocess.run(plan["command"], env={**os.environ, **plan["environment"]}, check=True)
