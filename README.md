# AshayLLM

Local-first foundation-model training project. **Currently includes scaffolding and diagnostic checks only**; dataset discovery, tokenizer, and training are upcoming steps.

## Local environment

### Hardware-aware commands

Run these inside the project's Python 3.11/3.12 virtual environment, with uv on PATH:

```bash
ashayllm setup                          # Detect hardware and print an install plan
ashayllm setup --apply                  # Install PyTorch/NumPy into this virtual environment
ashayllm setup --backend cpu            # Preview an explicit CPU installation
ashayllm doctor                        # Report the installed backend and GPU memory
ashayllm resolve-runtime configs/local.yaml --output artifacts/environment/resolved-local.json
ashayllm smoke                         # Verify training and checkpoint recovery on auto hardware
ashayllm smoke --accelerator cpu --precision fp32
```

`accelerator: auto` selects an available CUDA/ROCm device, then MPS, then CPU. ROCm is reported distinctly but uses PyTorch's `cuda` device API. `precision: auto` selects BF16 on supported CUDA/ROCm hardware and FP32 otherwise; explicit FP16 on CUDA/ROCm enables gradient scaling. Unsupported explicit settings fail instead of silently falling back.

Setup pins PyTorch 2.13.0 and NumPy 2.3.5. Its default CUDA 13.0 build requires an NVIDIA driver major version of at least 580. Older hardware/drivers need a compatible version via `--torch-version` and an official wheel index via `--index-url`. ROCm setup requires an explicit index matching the GPU and installed ROCm stack; it does not assume every AMD GPU is supported. macOS uses PyPI wheels and detects actual MPS availability at runtime. Drivers, OS prerequisites, and uv must already be installed. See the [official PyTorch installation selector](https://pytorch.org/get-started/locally/).

### Optional vLLM inference

Training and inference can share `.venv`, with PyTorch 2.13.0, CUDA 13.0, and vLLM 0.31.0. vLLM is optional: the base CPU/MPS/ROCm training pipeline does not import it. Our automatic vLLM installer supports Linux/WSL NVIDIA CUDA and requires a C compiler (`sudo apt install build-essential` on Ubuntu); other inference backends need their backend-specific installation recipe from the [vLLM documentation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

```bash
ashayllm setup-vllm                    # Preview shared-environment installation
ashayllm setup-vllm --apply            # Install the pinned training/inference stack
python scripts/vllm_smoke.py          # Generate with tiny random local weights
ashayllm serve-vllm configs/inference.yaml       # Preview serving command
ashayllm serve-vllm configs/inference.yaml --run # Start foreground localhost server
```

Run from the project directory. Serving requires a local Hugging Face export with config, tokenizer, and safetensors/bin weights using a vLLM-supported architecture. A `.pt` training checkpoint alone cannot be served; export integration will follow the transformer implementation. The default configuration reserves 50% GPU memory, limits context to 2048 and concurrent sequences to 4, and uses eager execution. These are conservative starting settings, not a guarantee that every model fits 8 GB. Training and serving should run sequentially on the laptop.

The launcher disables Hugging Face network downloads, binds to localhost, and does not enable remote model code. `setup-vllm --apply` downloads software packages; no model or dataset download is performed. Package snapshots preserve the resolved environment; choosing a different vLLM version may also require updating the PyTorch pin.

Precision settings: training accepts `fp32`, `bf16`, and `fp16` (CUDA/ROCm FP16 uses gradient scaling). Inference accepts vLLM dtype values such as `float16`, `bfloat16`, and `float32`. FP4 training and inference quantization are not implemented; they require a separate supported quantization recipe, model export format, and compatible kernels/hardware.

Setup never runs during training. The resolved JSON records hardware, package versions, precision, and a conservative microbatch default of 1. Automatic model-specific batch-size calibration will be added when the transformer training step exists; this default is not a memory-fit guarantee. ROCm/MPS selection has simulated test coverage and requires real-device validation.

Use a Python 3.11/3.12 virtual environment. Keep generated environment reports and installation snapshots outside Git.

From Ubuntu:

```bash
cd ~/projects/ashayllm
source .venv/bin/activate
python -m ashayllm.cli doctor
python scripts/cuda_smoke.py
python -m pytest -q
```

For a new Linux environment with its development virtual environment already created, `bash scripts/setup-cuda.sh` installs the verified PyTorch/NumPy versions and runs checks. Its dependency snapshot records the installed environment; it is not a portable hashed lockfile.

## Windows 11 setup (PowerShell administrator)

```powershell
nvidia-smi
wsl --status
wsl --list --verbose
wsl --install -d Ubuntu-24.04  # ONLY if Ubuntu is not already installed
wsl --update
```

Reboot if prompted. Launch Ubuntu, create your Linux username/password. Docker Desktop should be set to use **WSL2 backend** with Ubuntu integration enabled.

**Never install a Linux NVIDIA display driver inside WSL2.** Use your NVIDIA Windows driver. See https://docs.nvidia.com/cuda/wsl-user-guide/index.html

## Within Ubuntu (WSL2)

Save/extract this folder under the Linux filesystem, e.g. `~/projects/ashayllm`, not `/mnt/c/...`.

```bash
sudo apt update && sudo apt install -y git curl build-essential python3-venv unzip
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~/projects/ashayllm
uv venv --python 3.11
uv pip install -e '.[dev]'
uv run --no-sync python -m ashayllm.cli check-config configs/local.yaml
uv run --no-sync python -m ashayllm.cli doctor
uv run --no-sync python -m pytest -q
```

For the pinned CUDA stack, first check your Windows NVIDIA driver version and compatible install options at https://pytorch.org/get-started/locally/ :

```bash
bash scripts/setup-cuda.sh
uv run --no-sync python -m ashayllm.cli doctor
```

The driver and GPU must support the selected CUDA runtime and PyTorch build. For older hardware, use the hardware-aware setup command with explicit version and wheel-index overrides.

## Docker GPU diagnostic (optional)

Run from Ubuntu with Docker Desktop and WSL integration enabled:

```bash
docker run --rm --gpus all ubuntu nvidia-smi
```

No AWS account required for Step 0.1.

## Reproducible AWS Spot smoke test

Available CLI commands (`python -m ashayllm.cli --help`):

| Command | Purpose |
| --- | --- |
| `doctor` | Inspect hardware, PyTorch/CUDA and GPU matrix multiplication |
| `setup` | Plan hardware-aware PyTorch installation; `--apply` installs |
| `resolve-runtime CONFIG` | Resolve accelerator/precision and optionally save a report |
| `check-config CONFIG` | Validate initial training configuration |
| `smoke` | Synthetic training and checkpoint recovery |
| `setup-vllm` | Plan optional vLLM installation; `--apply` installs |
| `serve-vllm CONFIG` | Plan local inference serving; `--run` starts it |
| `aws-test [CONFIG]` | Plan Spot test; `--run` runs local preflight, provisions and tests |
| `aws-reconcile RUN_ID` | Recover saved cloud results and clean temporary resources after confirmed termination |

Configure `configs/aws-test.yaml`: AWS profile/region, instance type, subnet, AMI,
runtime, hourly ceiling, and per-run/overall budget. Credentials stay in the AWS
CLI profile outside this repository; never paste keys into YAML or commit them.
The template uses the `ashayllm` profile (`aws configure --profile ashayllm`).
Machine-specific overrides belong in ignored `configs/*.local.yaml` files;
pass that file as the optional `aws-test` configuration argument.
Install the controller with `python -m pip install -e '.[dev,aws]'`.

```powershell
# Read-only plan; run these from the project directory.
python -m ashayllm.cli aws-test
# Launch and monitor the configured test.
python -m ashayllm.cli aws-test --run
# After a controller interruption or termination waiter timeout:
python -m ashayllm.cli aws-reconcile YOUR_SAVED_RUN_ID
```

The runner copies allowlisted Python source/tests through EC2 user data, uses
an Amazon Deep Learning AMI with PyTorch installed, detects hardware, runs tests,
and checks synthetic training/checkpoint recovery with automatic precision and FP16.
vLLM installation is optional and disabled for the initial inexpensive test.
Before each launch, the runner automatically runs local pytest and Ruff checks;
any failure prevents provisioning. Results are saved to `artifacts/aws-preflight.json`
and included in the cloud run report. Install the `dev` extra on the controller.
These tests do not require training corpora or a local GPU.
Guest shutdown terminates the instance; a guest timer, AWS Scheduler deadline,
and controller cleanup provide additional termination safeguards. Root EBS is
encrypted and deleted on termination; no inbound security-group rules are added.
The scoped scheduler IAM role and schedule group remain for reuse.

Reports under `artifacts/aws-runs/` record the resolved AMI, source hash, instance,
console output, and termination status. Pin `ami_id` to reproduce a resolved image.
The local SQLite ledger under `.aws-runs/` atomically reserves the full per-run
amount, including failed attempts, against the configured overall limit. Preserve
this ledger across runs. Its scope is this ledger/account/region, not all AWS
account spending or runners using other ledger copies. Limits use conservative
configured rates and runtime estimates; they are not a real-time AWS billing cap.
AWS/API failures can delay termination, so review the saved termination result.

Future dataset preflight must use deterministic bounded samples (fixed seed and
manifest identity, with record/byte/token limits), rather than downloading or
processing the complete corpus locally. Sample checks will verify schema,
filtering, tokenizer compatibility, and a small training step; full-corpus quality
and throughput checks remain separate. This sampling stage is not implemented yet.
