"""Optional vLLM layer sharing the project's verified PyTorch environment."""

import ipaddress
import json
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


def serve_plan(config, root, *, check_hardware=True):
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
    host = settings.get('host', '127.0.0.1')
    try:
        external = not ipaddress.ip_address(host).is_loopback
    except ValueError as error:
        raise ValueError('host must be a literal IP address') from error
    security = settings.get('security', {})
    if not isinstance(security, dict):
        raise TypeError('security must be a mapping')
    key_env = security.get('api_key_env', 'MODELCHEF_API_KEY')
    if not isinstance(key_env, str) or not key_env.isidentifier():
        raise ValueError('api_key_env must be an environment variable name')
    limits = {}
    for name, default in [('max_body_bytes', 1048576), ('requests_per_minute', 30),
                          ('max_concurrent_requests', 2), ('request_timeout_seconds', 120),
                          ('max_output_tokens', min(256, length)), ('max_batch_size', 1),
                          ('max_choices', 1)]:
        value = security.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
        limits[name] = value
    if limits['max_output_tokens'] > length:
        raise ValueError('max_output_tokens cannot exceed max_model_len')
    limits['api_key_env'] = key_env
    cert, key = security.get('tls_certfile'), security.get('tls_keyfile')
    if bool(cert) != bool(key) or (external and not cert):
        raise ValueError('Non-loopback serving requires both TLS certificate and key files')
    command = [str(executable), "serve", str(model), "--host", host,
               "--port", str(port), "--dtype", dtype,
               "--gpu-memory-utilization", str(memory), "--max-model-len", str(length),
               "--max-num-seqs", str(sequences)]
    command += ['--middleware', 'modelchef.api_security.SecurityMiddleware',
                '--disable-fastapi-docs', '--no-enable-log-requests', '--api-server-count', '1']
    if cert:
        for flag, filename in [('--ssl-certfile', cert), ('--ssl-keyfile', key)]:
            path = (root / filename).resolve()
            if not path.is_file():
                raise ValueError(f'{flag} file does not exist')
            command += [flag, str(path)]
    if settings.get("enforce_eager", True):
        command.append("--enforce-eager")
    quantization = settings.get('quantization', 'none')
    if quantization not in {'none', 'nvfp4_per_token'}:
        raise ValueError('quantization must be none or nvfp4_per_token')
    if quantization != 'none' and check_hardware:
        from modelchef.hardware import detect_hardware
        hardware = detect_hardware()
        if hardware['backend'] != 'cuda' or hardware.get('compute_capability', [0])[0] < 10:
            raise ValueError('Native NVFP4 inference requires a supported Blackwell CUDA GPU')
    if quantization != 'none':
        command += ['--quantization', quantization]
    return {"command": command, "environment": {"HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "VLLM_NO_USAGE_STATS": "1",
            "MODELCHEF_API_SECURITY": json.dumps(limits),
            "PYTHONPATH": os.pathsep.join([str(root / 'src'), os.environ.get('PYTHONPATH', '')])},
            "endpoint": f"{'https' if cert else 'http'}://{'[' + host + ']' if ':' in host else host}:{port}/v1",
            "api_key_env": key_env,
            "note": "Local weights only; architecture compatibility and memory fit "
                    "are verified by vLLM at startup. No trust-remote-code enabled."}


def launch_server(plan):
    import subprocess

    if not Path(plan["command"][0]).is_file():
        raise ValueError("vLLM environment is missing; run setup-vllm --apply first")
    secret = os.environ.get(plan['api_key_env'], '')
    if len(secret) < 32 or any(char.isspace() for char in secret):
        raise ValueError('Set the configured API key environment variable to a random token of at least 32 characters')
    subprocess.run(plan["command"], env={**os.environ, **plan["environment"]}, check=True)
