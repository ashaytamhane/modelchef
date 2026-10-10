# AshayLLM

AshayLLM is a Python/PyTorch project for building a small language model from random initialization, with one codebase for local development and AWS GPU experiments. Develop and debug cheaply on your own computer; explicitly launch a budget-controlled AWS Spot test when you need to check a cloud GPU.

**Current scope:** hardware-aware installation plans, runtime/precision selection, synthetic training and checkpoint checks, optional vLLM inference setup, and an automated EC2 Spot smoke-test runner. Corpus processing, tokenizer training, the actual transformer, pretraining, midtraining, SFT, DPO/RLVR and model benchmarks are planned. There is no `train` command yet.

## Choose where to run

| Goal | Where execution happens | What to run | AWS required? |
| --- | --- | --- | --- |
| Check your laptop/desktop GPU | Current machine | `doctor`, `resolve-runtime`, `smoke` | No |
| Develop code and run unit tests | Current machine | `python -m pytest -q`, `python -m ruff check .` | No |
| Install PyTorch or vLLM | Current machine | `setup --apply`, `setup-vllm --apply` | No |
| Serve an existing local model export | Current Linux/WSL CUDA machine | `serve-vllm ... --run` | No |
| Preview an EC2 test | Local controller makes read-only AWS requests | `aws-test` | Credentials, but no instance launched |
| Provision and test a cloud GPU | Local controller launches a temporary EC2 instance; tests run there | `aws-test ... --run` | Yes; incurs AWS charges |
| Recover cloud results/cleanup | Local controller queries AWS and removes temporary resources after termination | `aws-reconcile` | Yes; no new instance launched |

Commands run **where you invoke them**. If you SSH into an existing EC2 instance and run `doctor`, it inspects that EC2 instance. It does not provision another machine. Only `aws-test --run` launches compute in this project.

`configs/local.yaml` and `configs/aws.yaml` describe runtime/model planning settings. Selecting `aws.yaml` does **not** deploy anything. `configs/aws-test.yaml` is the separate EC2 provisioning/budget configuration.

## Tutorial roadmap

1. Clone the repository and create a local development environment.
2. Select and verify local hardware/precision.
3. Optionally install and test inference.
4. Set up AWS CLI credentials and IAM permissions.
5. Preview, launch and verify a budget-controlled Spot test.
6. Inspect results and recover interrupted runs.

Run all project commands from the repository root. After installing the package, `ashayllm COMMAND` and `python -m ashayllm.cli COMMAND` are equivalent; this guide uses the second form to make the Python environment explicit.

## 1. Install locally

### Windows with an NVIDIA GPU: prepare WSL2

Use the Windows NVIDIA driver. Do not install a Linux NVIDIA display driver inside WSL. See the [NVIDIA WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html).

In Windows PowerShell:

```powershell
nvidia-smi
wsl --status
wsl --list --verbose
# Only if Ubuntu is not already installed; installation may require administrator rights.
wsl --install -d Ubuntu-24.04
```

Restart if requested, open Ubuntu, and create your normal Linux user. Verify Ubuntu internet access before installing packages:

```bash
curl --connect-timeout 10 --max-time 15 -I https://pypi.org
nvidia-smi
```

Docker is optional; it is not required for the CLI or Spot runner.

### Linux/WSL: clone and create the development environment

Install Git, a C compiler and curl using your distribution's package manager. On Ubuntu:

```bash
sudo apt update
sudo apt install -y git curl build-essential
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/ashaytamhane/ashayllm.git
cd ashayllm
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev]'
python -m ashayllm.cli --help
python -m pytest -q
python -m ruff check .
```

The repository is private, so GitHub authentication is needed to clone it. Keep WSL working files on the Linux filesystem rather than under `/mnt/c` for development performance. Installation downloads software packages, not training corpora or pretrained models. See [uv installation](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

### Windows-only controller or CPU development

A Windows controller does not need PyTorch or a GPU to launch AWS tests. With Python 3.12 and Git installed:

```powershell
git clone https://github.com/ashaytamhane/ashayllm.git
cd ashayllm
py -3.12 -m venv .venv-aws
.\.venv-aws\Scripts\Activate.ps1
python -m pip install -e '.[dev,aws]'
python -m ashayllm.cli --help
python -m pytest -q
```

If PowerShell activation is unavailable, invoke `.\.venv-aws\Scripts\python.exe` directly instead of `python`. Local GPU inference through the automatic vLLM setup requires Linux/WSL NVIDIA CUDA.

## 2. Select and verify local hardware

Creating the base environment does not install PyTorch. First preview the installation:

```bash
python -m ashayllm.cli setup
# Or deliberately choose a CPU-only environment:
python -m ashayllm.cli setup --backend cpu
```

Review the detected backend, driver and proposed command. Install into the active environment only when ready:

```bash
python -m ashayllm.cli setup --apply
python -m ashayllm.cli doctor
python -m ashayllm.cli check-config configs/local.yaml
python -m ashayllm.cli resolve-runtime configs/local.yaml --output artifacts/environment/resolved-local.json
python -m ashayllm.cli smoke
```

Expected results: `doctor` describes the available backend; on CUDA it also checks matrix multiplication. `smoke` reports `training_step: passed` and `checkpoint_recovery: passed`. These are synthetic checks, not language-model training or throughput measurements.

The default stack pins PyTorch 2.13.0 and NumPy 2.3.5. The default CUDA 13.0 installation plan requires NVIDIA driver major version 580 or newer. Setup installs packages, not drivers or OS prerequisites. For another GPU/driver combination, select an appropriate official build with `--torch-version` and `--index-url`; check the [PyTorch installation selector](https://pytorch.org/get-started/locally/). ROCm setup needs an explicit compatible wheel index; MPS requires macOS. Those backends still need real-device validation.

### Precision and memory

| Setting | Behavior |
| --- | --- |
| `accelerator: auto` | Available CUDA/ROCm, then MPS, otherwise CPU |
| `precision: auto` | BF16 when PyTorch reports support on CUDA/ROCm; otherwise FP32 |
| `fp16` | Explicit CUDA/ROCm training with gradient scaling |
| `fp32` | Default fallback; useful for CPU checks |
| FP4 | Not implemented |

BF16 reporting can include emulated support; it does not establish native acceleration or performance. Explicit unavailable backends/precision fail rather than silently switching. Microbatch defaults to 1; automatic model-specific memory calibration is pending.

Examples:

```bash
# Debug checkpoint behavior without using the GPU.
python -m ashayllm.cli smoke --accelerator cpu --precision fp32
# Check mixed-precision training on your current CUDA machine.
python -m ashayllm.cli smoke --accelerator cuda --precision fp16
```

## 3. Optional local inference with vLLM

Use the Linux/WSL `.venv` from step 1, a compatible NVIDIA GPU/driver, and a C compiler:

```bash
python -m ashayllm.cli setup-vllm
python -m ashayllm.cli setup-vllm --apply
python scripts/vllm_smoke.py
```

The installer targets `.venv` in the project root and pins vLLM 0.31.0 with the shared PyTorch/CUDA stack. The smoke script generates from tiny random local weights; it does not test a useful trained language model. Other inference backends need their own [vLLM installation recipe](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

To serve an existing compatible Hugging Face export, set `inference.model_path` in `configs/inference.yaml` to a local directory containing `config.json`, `tokenizer.json` and safetensors or `pytorch_model.bin` weights:

```bash
python -m ashayllm.cli serve-vllm configs/inference.yaml
python -m ashayllm.cli serve-vllm configs/inference.yaml --run
```

The first command validates files and prints a plan. The second starts foreground serving at `http://127.0.0.1:8000/v1`. Stop it with Ctrl+C. Downloads and remote model code are disabled. A `.pt` checkpoint alone is insufficient; export from the future AshayLLM transformer is not implemented yet. Default memory utilization is 50%, context 2048 and concurrency 4; adjust for your model, and avoid simultaneous training/serving on a small GPU.

## 4. Prepare AWS access

Skip this entire section for local-only work. AWS commands run from your **controller** environment: Windows, WSL or Linux. Use the same environment for credentials and the runner; Windows and WSL normally have separate AWS profile files.

### 4.1 Install AWS CLI and the controller dependencies

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), reopen your terminal, and check:

```bash
aws --version
python -m pip install -e '.[dev,aws]'
```

The controller needs pytest and Ruff for automatic local preflight; it does not need local CUDA. Keep your controller terminal open while the run proceeds.

### 4.2 Set up the caller identity and permissions

There are two identities:

| Identity | Responsibility |
| --- | --- |
| Your configured CLI profile | Discover images/prices/subnets, provision EC2, monitor results, manage temporary schedules and clean up |
| `ashayllm-spot-deadline` scheduler role | Permit AWS Scheduler to terminate project-tagged instances at the deadline |

Use an IAM administrator for one-time permission setup. Do not use root access keys or put `AdministratorAccess` on the everyday controller just to avoid configuring permissions.

For an IAM user workflow:

1. In AWS Console, open **IAM > Users > Create user** and create a dedicated controller user, such as `ashayllm-controller`. Console access is unnecessary for CLI-only use.
2. Generate a baseline permission policy from the current AWS runner rather than relying on a stale copied policy. With uv installed, use the command below. It analyzes code; it does not launch EC2 or attach policies.
3. Have the administrator review the generated policies, replace wildcard resources where feasible with your account/region/project resources, and validate them in IAM. Static analysis is a starting point: dynamically constructed service clients, paginator/waiter operations and dependent permissions need review.
4. In **IAM > Policies > Create policy > JSON**, create the reviewed customer-managed policy or policies. Attach them to the controller user (or its group/role). Regenerate/review when AWS runner behavior changes.
5. Decide whether the controller may create the scheduler role on first use, or whether the administrator will pre-create it with the exact name and trust/termination policy described below.

From the repository root on Linux/WSL, replace `YOUR_ACCOUNT_ID` with your account ID:

```bash
uvx iam-policy-autopilot@latest generate-policies \
  "$PWD/src/ashayllm/aws_runner.py" \
  --region us-east-1 --account YOUR_ACCOUNT_ID \
  --service-hints ec2 iam scheduler ssm sts --pretty
```

See [AWS IAM Policy Autopilot](https://github.com/awslabs/iam-policy-autopilot) for supported inputs and policy review guidance. PowerShell uses `(Resolve-Path src/ashayllm/aws_runner.py).Path` for the source path and backticks for line continuation, or place the command on one line.

The administrator must cover EC2 discovery, Spot launch/tagging, security-group creation/deletion, console retrieval and termination, including operations used by waiters. Scheduler access covers the `ashayllm` group and its run schedules. Passing the role needs `iam:PassRole` restricted to `arn:aws:iam::YOUR_ACCOUNT_ID:role/ashayllm-spot-deadline` and `iam:PassedToService = scheduler.amazonaws.com`; see [AWS PassRole guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_passrole.html). First-time Spot accounts may also need the EC2 Spot service-linked role created by an administrator.

The runner creates the scheduler group/role if missing. Role creation requires the corresponding IAM role creation, tagging, inline-policy and read permissions. Alternatively, an administrator can pre-create `ashayllm-spot-deadline` using the trust and inline-policy definitions in `SpotRunner.prepare_deadline_role` in [aws_runner.py](src/ashayllm/aws_runner.py). Its trust permits `scheduler.amazonaws.com`, restricted to the caller account and the exact regional `schedule-group/ashayllm` ARN. Its policy permits only `ec2:TerminateInstances` on regional account instances tagged `Project=AshayLLM`. The EC2 guest receives no controller keys or instance profile in this recipe. See [Scheduler execution role setup](https://docs.aws.amazon.com/scheduler/latest/UserGuide/setting-up.html).

An existing scheduler role is reused without repairing its policy. Have the administrator verify it, particularly if switching regions: the current role name is shared but its trust/policy are region-specific.

### 4.3 Configure credentials securely

If you use IAM user keys, open the user's **Security credentials > Access keys > Create access key** and choose CLI use. Enter the key pair only in your own terminal:

```bash
aws configure --profile ashayllm
```

Supply the access key ID, secret access key, default region (for example `us-east-1`) and output format `json`. The CLI stores credentials under your user profile, outside this repository. They are stored as plaintext there, so protect that directory. Never paste keys into chat, YAML, scripts or Git. AWS recommends temporary credentials where available; an IAM Identity Center profile can instead be configured using `aws configure sso --profile ashayllm` and authenticated with `aws sso login --profile ashayllm`. See [AWS credential guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).

Check identity without launching compute:

```bash
aws sts get-caller-identity --profile ashayllm
```

Already have a differently named profile? Set `aws_test.profile` in your ignored local configuration; no new keys are required.

### 4.4 Check regional prerequisites

Use a region offering your chosen x86 NVIDIA G-series type. The runner currently accepts `g4dn`, `g5`, `g6` and `g6e` families. Check the EC2 Spot GPU vCPU quota in **Service Quotas**; quota and capacity are separate issues. A public subnet needs an internet-gateway route and outbound internet access for package installation. If no subnet is specified, the runner selects an available default-VPC subnet by Spot price; it does not build a VPC or repair routing.

The AMI must be an Amazon-owned x86 GPU PyTorch DLAMI with `/opt/pytorch/bin/activate`, and its root volume must fit the configured storage size. The runner first tries the public SSM AMI parameter, then falls back to the configured Amazon-owned image name on access denial/missing parameter. Arbitrary custom AMIs and ARM instances are not supported by this recipe.

## 5. Configure and preview an AWS test

Copy the checked-in template to an ignored local override:

```bash
cp configs/aws-test.yaml configs/aws-test.local.yaml
```

In PowerShell use `Copy-Item configs/aws-test.yaml configs/aws-test.local.yaml`. Edit the **complete** copied configuration; local overrides are complete files, not merged patches.

| Field | Template value | What it controls |
| --- | --- | --- |
| `profile`, `region` | `ashayllm`, `us-east-1` | Controller credentials and deployment region |
| `instance_type` | `g4dn.xlarge` | Requested Spot GPU machine |
| `total_budget_usd` | `10.00` | Total reservation limit for this ledger/account/region |
| `per_run_budget_usd` | `0.50` | Full amount reserved for every attempt |
| `max_runtime_minutes` | `12` | Runtime/deadline limit |
| `max_spot_hourly_usd` | `0.80` | Spot hourly price ceiling, not an hourly purchase |
| `safety_reserve_usd` | `0.15` | Estimate allowance for additional costs/delays |
| `storage_gb` | `100` | Encrypted gp3 root volume size, deleted on termination |
| `ebs_gb_month_usd`, `ipv4_hourly_usd` | Configurable estimates | Conservative regional rate inputs to budget validation |
| `ledger_path` | `.aws-runs/ledger.sqlite` | Durable local budget journal |
| `subnet_id`, `ami_id` | `null` | Automatic selection; explicit values pin a choice |
| `install_vllm` | `false` | Optional cloud package installation; not a cloud inference benchmark |

Preview first:

```bash
python -m ashayllm.cli aws-test configs/aws-test.local.yaml
# Temporary instance-type override without editing YAML:
python -m ashayllm.cli aws-test configs/aws-test.local.yaml --instance-type g6.xlarge
```

This prints the selected AMI, subnet, zone, current Spot price and reservation. It uses read-only AWS requests and does not provision resources or run the local test suite. Check rates and budget before launching. Pin `ami_id` from the plan when you need the same image on subsequent runs.

The budget validator includes the requested duration, three extra minutes, configured compute/storage/IPv4 rates and the safety allowance. It rejects configurations that do not fit. The ledger reserves the full per-run amount, including failed attempts, and retains reservations after termination. `reserve()` refers to this bookkeeping, **not EC2 Reserved Instances**. Preserve the ledger; deleting it or using another copy resets its accounting scope. It is not an account-wide billing cap or live AWS invoice monitor; taxes, data transfer and prolonged AWS/API failures are not exactly reconciled.

## 6. Launch, monitor and verify

**This command creates billable AWS resources:**

```bash
python -m ashayllm.cli aws-test configs/aws-test.local.yaml --run
```

The runner performs this sequence automatically:

1. Run pytest and Ruff locally. Failure prevents provisioning; save evidence to `artifacts/aws-preflight.json`.
2. Resolve the cloud plan, validate estimated budget and atomically reserve the per-run amount.
3. Create/reuse the scheduler role/group and create a temporary outbound-only security group.
4. Launch one one-time Spot instance with IMDSv2 and an encrypted root volume. There is no on-demand fallback.
5. Copy allowlisted Python source/tests through EC2 user data. Keys, datasets, Git history and local artifacts are not copied. This is a smoke-test payload, not a full repository clone or general deployment system.
6. Arm a guest timer and an AWS Scheduler deadline. Use the image's preinstalled PyTorch, install small test dependencies, detect cloud hardware, run remote tests and synthetic auto-precision/FP16 training/checkpoint checks.
7. Power off on success or failure; EC2 shutdown behavior is `terminate`. The controller monitors termination and its cleanup can escalate to force termination after a stalled shutdown.
8. Save the report and remove the temporary schedule/security group after confirmed termination. The reusable scheduler role/group remain.

The guest timeout starts during bootstrap; the cloud deadline is computed before launch. The controller's elapsed timeout includes provisioning setup. These safeguards reduce runaway costs but cannot guarantee instantaneous shutdown during AWS/API failures. No datasets are downloaded. The current cloud recipe runs source via PYTHONPATH in the DLAMI's preinstalled Python; it does not validate editable package installation there. DLAMI Python may differ from the project's local Python 3.11/3.12 target.

## 7. Read results and recover cleanup

A run report under `artifacts/aws-runs/ASHAYLLM_RUN_ID/report.json` records the resolved image, source hash, instance ID, local preflight, console output and status. Look for **both** `test_passed: true` and `terminated: true`. Cloud tests exclude AWS controller tests because the controller is not installed on the guest. Tests may pass while cleanup is still pending.

If the terminal was interrupted or the termination waiter timed out:

```bash
python -m ashayllm.cli aws-reconcile ASHAYLLM_RUN_ID --path configs/aws-test.local.yaml
```

Use the run ID from the saved ledger/report/error context. Reconciliation queries the journaled instance, saves console/results, and removes its temporary schedule/security group only when already terminated. **It does not terminate a still-running instance or launch anything.** Inspect the `terminated` field; command exit alone does not prove cleanup.

If needed, explicitly terminate the instance using its saved ID, then reconcile again:

```bash
aws ec2 terminate-instances --profile ashayllm --region us-east-1 --instance-ids INSTANCE_ID
aws ec2 describe-instances --profile ashayllm --region us-east-1 --instance-ids INSTANCE_ID
```

A successful termination request can still report `shutting-down`. Confirm `terminated` and root-volume removal in EC2. The ledger is local and run-ID recovery after an early crash is limited; it is not a distributed job service.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| `aws` not found | Install AWS CLI v2; reopen the terminal |
| Profile/credentials unavailable | Configure/login in the same OS environment as the controller |
| `AccessDenied` | Review the denied action with your IAM administrator, including PassRole/bootstrap permissions |
| Spot capacity unavailable | Preview another supported type or public subnet/AZ; retries currently require another explicit run |
| GPU Spot quota exceeded | Request the appropriate regional quota in Service Quotas |
| Local preflight fails | Read `artifacts/aws-preflight.json`; install `dev` dependencies and fix tests/lint |
| Budget exhausted | Review retained ledger reservations and intentionally adjust budget; do not delete the journal to bypass it |
| CUDA unavailable | Check driver, installed wheel and `doctor`; choose an appropriate official PyTorch build |
| WSL HTTPS timeout | Test networking inside Ubuntu before setup; this project does not repair host networking |
| vLLM export missing | Provide local config/tokenizer/weights; a checkpoint alone cannot be served |
| Termination timeout | Check EC2 state; use reconciliation after termination and retain the AWS deadline |

## Use cases and next stages

**Today: laptop development.** Install the base development environment, preview/install the appropriate PyTorch build, run unit tests and synthetic smoke checks without AWS. Use CPU FP32 when a GPU is unavailable.

**Today: cloud readiness check.** Keep a lightweight local controller, preview a small G-series Spot machine, then use `aws-test --run` to validate the same core code on a cloud GPU with a declared budget and cleanup safeguards.

**Today: offline inference plumbing.** On Linux CUDA, install optional vLLM and serve an existing supported local model export. This does not produce a trained AshayLLM model.

**Planned: foundation-model training.** Acquire approved public corpora using SmolLM2/OLMo recipes as references, build versioned manifests, quality/deduplication/PII/decontamination stages, train BPE and a decoder-only transformer from random initialization, then add midtraining, SFT and optional DPO/RLVR. Evaluate against GPT-2, Pythia, SmolLM2 and standard suites. FineWeb-Edu, DCLM, Stack-Edu, FineMath, Cosmopedia, Dolma and public reference corpora are candidate inputs, not automatic downloads.

**Planned: inexpensive data preflight.** Use deterministic samples identified by manifest/version and seed, with document/byte/token limits; test format, filtering, tokenization and a tiny training step locally before a separately approved cloud job. Sampling does not replace full-corpus quality/decontamination checks. `max_sample_documents` and model targets in the current YAML are planning fields, not implemented download/training controls.

## CLI reference and repository layout

| Command | Purpose |
| --- | --- |
| `doctor` | Inspect current hardware and PyTorch/CUDA |
| `setup [--apply]` | Plan/install hardware-aware PyTorch packages |
| `check-config CONFIG` | Validate required top-level project/runtime sections; not full schema validation |
| `resolve-runtime CONFIG [--output FILE]` | Resolve device/precision and optionally persist the result |
| `smoke [--accelerator ...] [--precision ...]` | Synthetic training and checkpoint recovery |
| `setup-vllm [--apply]` | Plan/install optional shared vLLM stack |
| `serve-vllm CONFIG [--run]` | Validate/plan or start local serving |
| `aws-test [CONFIG] [--instance-type TYPE] [--run]` | Preview or execute a budget-reserved Spot test |
| `aws-reconcile RUN_ID [--path CONFIG]` | Recover cloud results and finish cleanup after termination |

Use `python -m ashayllm.cli COMMAND --help` for flags.

- `src/ashayllm/`: CLI, hardware policy, smoke checks, inference and AWS runner.
- `configs/`: versioned portable defaults; `*.local.yaml` stays local.
- `tests/`: unit tests and small integration checks.
- `scripts/`: optional setup and diagnostic helpers.
- `artifacts/`, `.aws-runs/`: ignored reports/checkpoints and the durable budget ledger.

Credentials, local configurations, historical setup notes, dependency snapshots and run artifacts are excluded from Git. Avoid account-specific IDs and past-run details in committed documentation. As the pipeline evolves, keep each new command's execution location, downloads, costs, reproducibility inputs and validation scope explicit in this tutorial.
