# AWS execution and operations

Run these commands from the repository root in your controller environment. For the same recipe on local hardware or Runpod, see the [execution tutorial](EXECUTION.md).

### AWS execution

For scarce GPU Spot capacity, the AWS runner can optionally use an **instant EC2
Fleet** with `price-capacity-optimized` selection across explicit G5/G6/G6e
instance types and subnets. It requests one four-vCPU instance and keeps the
existing budget, checkpoint and shutdown lifecycle. See [Fleet setup](AWS_FLEET.md).

Choose the lifecycle before launching. A training job and a hosted API use different runners:

| Workflow | Purchase model | Lifetime | Cost policy |
| --- | --- | --- | --- |
| `pipeline`/`cook --target aws`, `aws-test`, `aws-train` | Temporary Spot | Terminates after completion or deadline | Conservative reservations and runtime safeguards |
| `aws-serve` | On-demand | Remains running until explicitly stopped or deleted | Status warnings and separately configured AWS Budgets notifications |

Do not use a temporary training/test instance for a persistent API: its termination safeguards remain active. Export the trained model and deploy it independently. Existing manually provisioned hosts can still run `serve-vllm` directly.

#### Prepare AWS access

Skip this entire section for local-only work. AWS commands run from your **controller** environment: Windows, WSL or Linux. Use the same environment for credentials and the runner; Windows and WSL normally have separate AWS profile files.

##### Install AWS CLI and the controller dependencies

Install [AWS CLI v2](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html), reopen your terminal, and check:

```bash
aws --version
python -m pip install -e '.[dev,aws]'
```

The controller needs pytest and Ruff for automatic local preflight; it does not need local CUDA. Keep your controller terminal open while the run proceeds.

##### Set up the caller identity and permissions

There are two identities:

| Identity | Responsibility |
| --- | --- |
| Your configured CLI profile | Discover images/prices/subnets, provision EC2, monitor results, manage temporary schedules and clean up |
| `modelchef-spot-deadline` scheduler role | Permit AWS Scheduler to terminate project-tagged instances at the deadline |

Use an IAM administrator for one-time permission setup. Do not use root access keys or put `AdministratorAccess` on the everyday controller just to avoid configuring permissions.

For an IAM user workflow:

1. In AWS Console, open **IAM > Users > Create user** and create a dedicated controller user, such as `modelchef-controller`. Console access is unnecessary for CLI-only use.
2. Generate a baseline permission policy from the current AWS runner rather than relying on a stale copied policy. With uv installed, use the command below. It analyzes code; it does not launch EC2 or attach policies.
3. Have the administrator review the generated policies, replace wildcard resources where feasible with your account/region/project resources, and validate them in IAM. Static analysis is a starting point: dynamically constructed service clients, paginator/waiter operations and dependent permissions need review.
4. In **IAM > Policies > Create policy > JSON**, create the reviewed customer-managed policy or policies. Attach them to the controller user (or its group/role). Regenerate/review when AWS runner behavior changes.
5. Decide whether the controller may create the scheduler role on first use, or whether the administrator will pre-create it with the exact name and trust/termination policy described below.

From the repository root on Linux/WSL, replace `YOUR_ACCOUNT_ID` with your account ID:

```bash
uvx iam-policy-autopilot@latest generate-policies \
  "$PWD/src/modelchef/aws_runner.py" \
  --region us-east-1 --account YOUR_ACCOUNT_ID \
  --service-hints ec2 iam scheduler ssm sts --pretty
```

See [AWS IAM Policy Autopilot](https://github.com/awslabs/iam-policy-autopilot) for supported inputs and policy review guidance. PowerShell uses `(Resolve-Path src/modelchef/aws_runner.py).Path` for the source path and backticks for line continuation, or place the command on one line.

The administrator must cover EC2 discovery, Spot launch/tagging, security-group creation/deletion, console retrieval and termination, including operations used by waiters. Scheduler access covers the `modelchef` group and its run schedules. Passing the role needs `iam:PassRole` restricted to `arn:aws:iam::YOUR_ACCOUNT_ID:role/modelchef-spot-deadline` and `iam:PassedToService = scheduler.amazonaws.com`; see [AWS PassRole guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_roles_use_passrole.html). First-time Spot accounts may also need the EC2 Spot service-linked role created by an administrator.

The runner creates the scheduler group/role if missing. Role creation requires the corresponding IAM role creation, tagging, inline-policy and read permissions. Alternatively, an administrator can pre-create `modelchef-spot-deadline` using the trust and inline-policy definitions in `SpotRunner.prepare_deadline_role` in [aws_runner.py](../src/modelchef/aws_runner.py). Its trust permits `scheduler.amazonaws.com`, restricted to the caller account and the exact regional `schedule-group/modelchef` ARN. Its policy permits only `ec2:TerminateInstances` on regional account instances tagged `Project=ModelChef`. The EC2 guest receives no controller keys or instance profile in this recipe. See [Scheduler execution role setup](https://docs.aws.amazon.com/scheduler/latest/UserGuide/setting-up.html).

An existing scheduler role is reused without repairing its policy. Have the administrator verify it, particularly if switching regions: the current role name is shared but its trust/policy are region-specific.

##### Configure credentials securely

If you use IAM user keys, open the user's **Security credentials > Access keys > Create access key** and choose CLI use. Enter the key pair only in your own terminal:

```bash
aws configure --profile modelchef
```

Supply the access key ID, secret access key, default region (for example `us-east-1`) and output format `json`. The CLI stores credentials under your user profile, outside this repository. They are stored as plaintext there, so protect that directory. Never paste keys into chat, YAML, scripts or Git. AWS recommends temporary credentials where available; an IAM Identity Center profile can instead be configured using `aws configure sso --profile modelchef` and authenticated with `aws sso login --profile modelchef`. See [AWS credential guidance](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html).

Check identity without launching compute:

```bash
aws sts get-caller-identity --profile modelchef
```

Already have a differently named profile? Set `aws_test.profile` in your ignored local configuration; no new keys are required.

##### Check regional prerequisites

Use a region offering your chosen x86 NVIDIA G-series type. The runner currently accepts `g4dn`, `g5`, `g6` and `g6e` families. Check the EC2 Spot GPU vCPU quota in **Service Quotas**; quota and capacity are separate issues. A public subnet needs an internet-gateway route and outbound internet access for package installation. If no subnet is specified, the runner selects an available default-VPC subnet by Spot price; it does not build a VPC or repair routing.

The AMI must be an Amazon-owned x86 GPU PyTorch DLAMI with `/opt/pytorch/bin/activate`, and its root volume must fit the configured storage size. The runner first tries the public SSM AMI parameter, then falls back to the configured Amazon-owned image name on access denial/missing parameter. Arbitrary custom AMIs and ARM instances are not supported by this recipe.

#### Configure and preview a Spot readiness test

Copy the checked-in template to an ignored local override:

```bash
cp configs/aws-test.yaml configs/aws-test.local.yaml
```

In PowerShell use `Copy-Item configs/aws-test.yaml configs/aws-test.local.yaml`. Edit the **complete** copied configuration; local overrides are complete files, not merged patches.

| Field | Template value | What it controls |
| --- | --- | --- |
| `profile`, `region` | `modelchef`, `us-east-1` | Controller credentials and deployment region |
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
python -m modelchef.cli aws-test configs/aws-test.local.yaml
# Temporary instance-type override without editing YAML:
python -m modelchef.cli aws-test configs/aws-test.local.yaml --instance-type g6.xlarge
```

This prints the selected AMI, subnet, zone, current Spot price and reservation. It uses read-only AWS requests and does not provision resources or run the local test suite. Check rates and budget before launching. Pin `ami_id` from the plan when you need the same image on subsequent runs.

The budget validator includes the requested duration, three extra minutes, configured compute/storage/IPv4 rates and the safety allowance. It rejects configurations that do not fit. The ledger reserves the full per-run amount, including failed attempts, and retains reservations after termination. `reserve()` refers to this bookkeeping, **not EC2 Reserved Instances**. Preserve the ledger; deleting it or using another copy resets its accounting scope. It is not an account-wide billing cap or live AWS invoice monitor; taxes, data transfer and prolonged AWS/API failures are not exactly reconciled.

#### Launch, monitor and verify termination

**This command creates billable AWS resources:**

```bash
python -m modelchef.cli aws-test configs/aws-test.local.yaml --run
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

The guest timeout starts during bootstrap; the cloud deadline is computed before launch. The controller's elapsed timeout includes provisioning setup. These safeguards reduce runaway costs but cannot guarantee instantaneous shutdown during AWS/API failures. No datasets are downloaded. The readiness test intentionally sends a minimal bundle: `guest_cli.py` provides the readiness commands while `diagnostics.py`, `progress.py` and smoke components share production checks. This fits EC2's user-data size limit; it is not a full training installation. Managed training (`pipeline`/`cook --target aws` or `aws-train`) and `aws-serve` package the full project. The current cloud recipe runs source via PYTHONPATH in the DLAMI's preinstalled Python; it does not validate editable package installation there. DLAMI Python may differ from the project's local Python 3.11–3.13 target.

#### Recover cloud results and cleanup

A run report under `artifacts/aws-runs/MODELCHEF_RUN_ID/report.json` records the resolved image, source hash, instance ID, local preflight, console output and status. Look for **both** `test_passed: true` and `terminated: true`. Cloud tests exclude AWS controller tests because the controller is not installed on the guest. Tests may pass while cleanup is still pending.

If the terminal was interrupted or the termination waiter timed out:

```bash
python -m modelchef.cli aws-reconcile MODELCHEF_RUN_ID --path configs/aws-test.local.yaml
```

Use the run ID from the saved ledger/report/error context. Reconciliation queries the journaled instance, saves console/results, and removes its temporary schedule/security group only when already terminated. **It does not terminate a still-running instance or launch anything.** Inspect the `terminated` field; command exit alone does not prove cleanup.

If needed, explicitly terminate the instance using its saved ID, then reconcile again:

```bash
aws ec2 terminate-instances --profile modelchef --region us-east-1 --instance-ids INSTANCE_ID
aws ec2 describe-instances --profile modelchef --region us-east-1 --instance-ids INSTANCE_ID
```

A successful termination request can still report `shutting-down`. Confirm `terminated` and root-volume removal in EC2. The ledger is local and run-ID recovery after an early crash is limited; it is not a distributed job service.

#### Run a managed training stage

Use your own validated stage/workflow recipe for cloud training. For a small software demonstration, the included `workflow-demo.yaml` checks stage linking and export; it does not produce a useful assistant. Keep model-development experiments in a separate project.

```bash
cp configs/aws-training.yaml configs/aws-training.local.yaml
# Configure credentials, instance profile, artifact bucket, runtime and budget first.
python -m modelchef.cli cook configs/workflow-demo.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --target aws --aws-config configs/aws-training.local.yaml --run
```

The last command requires interactive `yes` after showing the current Spot quote and
budget reservation. It performs another bounded CPU preflight, runs both stages on
one Spot GPU, uploads the results and terminates the training instance. If the AWS
run reports both `test_passed: true` and `terminated: true`, recover its export:

```bash
python -m modelchef.cli aws-export MODELCHEF_RUN_ID
```

`aws-export` verifies the downloaded archive hash and extracts only the model export
to `artifacts/exports/MODELCHEF_RUN_ID/`. It also writes an ignored inference recipe
under `artifacts/aws-serving/`.

For GPU training, select `training.accelerator: auto` or `cuda` in your own stage recipes. Use `precision: auto` or a precision supported by the selected GPU. Changing the instance type does not override explicitly selected stage settings.

Spot is the default purchase option. If AWS reports insufficient capacity, select a
different subnet/availability zone or instance type. A confirmed capacity or vCPU-quota
rejection that creates no instance releases its reservation while retaining the audit record.
Ambiguous launch failures retain their reservation until investigated.

For a temporary On-Demand training run, set `market_type: on_demand` and add
`on_demand_hourly_usd` to the AWS settings. This configured rate drives the runtime
budget calculation; verify it against current AWS pricing for your instance and region.
The same completion/deadline termination applies. This does not purchase a Reserved
Instance. On-Demand G-series launches require a sufficient regional GPU vCPU quota;
a quota of zero prevents launch. Keep the artifact bucket until results have been
downloaded and verified. If result retrieval fails after termination, recover the
existing run with `aws-reconcile` rather than repeating training.

Put the generated inference recipe's path into a copy of
[`aws-serving.yaml`](../configs/aws-serving.yaml), then fill in its existing AMI,
network, instance profile, S3 bucket and Secrets Manager fields as described in
[persistent API hosting](#persistent-api-hosting-on-ec2). Serving is a separate
**on-demand, persistent** EC2 instance; it uses its own serving budget. Preview and deploy with `aws-serve`, verify readiness over HTTPS, and
chat from your local machine:

```bash
python -m modelchef.cli aws-serve configs/aws-serving.local.yaml
python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --run
python -m modelchef.cli chat-api --endpoint https://YOUR_HOSTNAME:8443/v1 \
  --prompt "Hello" --max-tokens 16
python -m modelchef.cli chat-api --endpoint https://YOUR_HOSTNAME:8443/v1 \
  --max-tokens 16
```

Set `MODELCHEF_API_KEY` in the local terminal from your secret store before using
`chat-api`; do not put the key on the command line or in Git. Use `--ca-cert PATH`
when your HTTPS certificate chains to a private CA. The CLI verifies TLS and sends
the bearer key only over HTTPS. It chooses the sole served model automatically; use
`--model ID` if the endpoint lists several. It uses the completions endpoint because this tiny
fixture has no chat template. Stop or delete the host explicitly with `aws-serve`
when finished; training shutdown does not stop a serving instance.

The two-stage CUDA training, checkpoint reload, result upload/download, export
recovery and EC2 termination have been exercised on a T4 with FP16. Live vLLM
hosting and remote chat remain separate validation steps; a recovered export does
not prove that the API is ready. Before deploying, check the regional On-Demand
GPU vCPU quota and permissions to provision/read the serving secret.

AWS training execution shows the current Spot price, maximum runtime, conservative reservation and overall budget, then requires you to type `yes` before launch. To automate a reviewed plan, explicitly add `--yes`; configured limits still apply. A non-interactive run without that flag fails instead of launching. The budget review stays visible with `--quiet`. The same confirmation applies to `aws-test` and the lower-level `aws-train` command.

Use the same `pipeline` or `cook` command with an explicit AWS target. Local is the default; hardware auto-detection selects a device on the chosen machine, never a cloud launch.

```bash
cp configs/aws-training.yaml configs/aws-training.local.yaml
# Fill in AWS resources and budget before previewing.
python -m modelchef.cli pipeline configs/pretrain-local.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli pipeline configs/pretrain-local.yaml --target aws --aws-config configs/aws-training.local.yaml --run

# Entire sequence on ONE Spot instance; supply its instruction dataset first.
python -m modelchef.cli cook configs/workflow-local.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli cook configs/workflow-local.yaml --target aws --aws-config configs/aws-training.local.yaml --run
```

The command-line recipe overrides `aws_test.pipeline_config` in the AWS settings. All workflow stages share the instance and one overall deadline/budget reservation. Dependencies are collected across stages, checkpoint links resolve on the guest, and the result archive contains stage outputs, workflow reports and the final export. Local preflight exercises the complete sequence with bounded two-step CPU stages before launching. Completion or failure invokes result upload and shutdown; the controller and AWS deadline remain in place. Workflows retain the same single-controller restriction as local `cook`: DDP stages run separately with `pipeline`. Cloud `--resume` is not supported; retrieve results and recover a stage locally. Preview validates/bounds the bundle and makes read-only AWS queries; it does not launch compute.

Copy `configs/aws-training.yaml` to ignored `configs/aws-training.local.yaml`. Follow the AWS credential/IAM tutorial above, then provide:

- `pipeline_config`: the stage or workflow recipe used by the lower-level `aws-train` command; `pipeline`/`cook --target aws` overrides it with the CLI recipe.
- `artifact_bucket`: an existing private S3 bucket in the chosen region, with suitable encryption and retention/lifecycle rules.
- `instance_profile_name`: an existing EC2 instance profile whose role can read the source object and write the result object under the configured prefix.
- Instance type, budgets, runtime, transfer caps and artifact cost reserve.

Your controller also needs S3 PutObject/GetObject permissions on that prefix and IAM PassRole for the selected instance role. The EC2 role must trust `ec2.amazonaws.com`; use temporary instance-role credentials rather than placing IAM keys in the bundle. Have your IAM administrator scope permissions to the actual bucket/prefix and role. This workflow does not create the bucket or instance profile.

```bash
python -m modelchef.cli aws-train configs/aws-training.local.yaml
# Explicit provisioning, upload and cloud training:
python -m modelchef.cli aws-train configs/aws-training.local.yaml --run
```

Before provisioning, the runner performs local tests/lint and an actual bounded two-step CPU data/training preflight. The controller therefore needs PyTorch and pipeline dependencies for managed training, unlike the lightweight `aws-test` controller. It allowlists source/configuration and declared inputs, checks bundle size, reserves budget, uploads encrypted source, launches one Spot instance and attaches the supplied profile. It packages text stages, local Hugging Face parent/teacher models, and multimodal recipes with bounded media/processor inputs using capped safetensors bundles. Pretrained config, tokenizer files and special-token roles are relocated together; remote code, environment files and arbitrary pickle weight downloads are excluded. Bundle limits must accommodate weights and checkpoints, and local preflight still requires enough memory for its CPU probe.

The guest arms a shutdown timer before package/data work, verifies the source checksum, installs pinned CUDA 13 PyTorch and recipe-selected extras, runs the stage or entire cook workflow and attempts a bounded result upload on exit. Compilation, TF32, optimized attention, fused/foreach optimizers, checkpointing and fake FP4 use the same execution code as local training. Custom CUDA sources are allowlisted into the bundle, and matching compiler packages are installed. Native NVFP4 selects Transformer Engine; DDP selects `torchrun --nproc-per-node=gpu`. Choose an AMI with a CUDA 13-compatible NVIDIA driver and an instance whose GPU supports the requested policy. The local CPU preflight validates data/objective/recovery; the guest validates the actual GPU optimizations. DDP plus compile/NVFP4/RLVR/QLoRA remains unsupported by the training engine, as enforced by recipe validation. The existing AWS deadline and controller termination safeguards remain active. Confirm `terminated` and inspect the saved cloud report; a lost/interrupted result upload can leave `results_missing` even after cleanup. Results are downloaded as an ignored `results.tar.gz`, without automatically extracting an external archive. S3 objects persist until your lifecycle rules remove them; the ledger is a conservative estimate, not a hard cap on every AWS charge. A small live Spot workflow has exercised pretraining, midtraining, SFT, export and automatic termination. This verifies execution and recovery, not the quality or cost of a full training run.

#### Remote data preparation and resumable checkpoints

A cook workflow can set `curation_recipe` to acquire and mix its ingredients on
the training machine before its stages start. Bulk Parquet acquisition retains
complete records and records source checksums and attribution. The local AWS
preflight still uses bounded samples: set `preflight_data_dir` in the AWS resource
configuration to their directory. Those samples do not replace the cloud corpus.
See [compose a complete workflow](../README.md#compose-a-complete-workflow) for generic stage linking. Supply your own independently evaluated datasets and stage recipes.

With an artifact bucket configured, the text training guest writes resumable S3
snapshots after checkpoints. A snapshot includes weights, optimizer and RNG states,
tokenizer, configuration, data snapshots and parent-stage lineage. The latest
snapshot pointer changes only after every object has uploaded successfully, so a
failed upload preserves the previous recovery point. `result_mode: metadata`
retrieves small reports while keeping large model snapshots in S3.

Use the `recovery_uri` from the cloud report to download and verify a snapshot:

```bash
python -m modelchef.recovery RECOVERY_URI artifacts/recovered --profile modelchef --region us-east-1 --max-bytes 20000000000
```

Replace `RECOVERY_URI` with the report value and choose a download cap that fits
the saved snapshot. Exact training resume requires the original source bundle,
configuration and workspace paths; restoring files into a different directory is
useful for inspection or inference but does not bypass resume identity checks.
Spot replacement and training restart are not automatic. S3 storage remains
billable after the instance terminates.

#### Persistent API hosting on EC2

This workflow packages a local compatible export and launches one on-demand NVIDIA G-series instance. A systemd service starts vLLM at boot and restarts it after failure. There is no training deadline, shutdown trap, Spot interruption policy or automatic cost-based shutdown. It is a single-host deployment, without autoscaling or high availability.

1. Complete the local export and API configuration described in [inference and API serving](../README.md#serve-your-model-inference-and-api). Copy `configs/inference-public.yaml` to `configs/inference-public.local.yaml`; set the export path, port and context length. The serving bundle rewrites certificate paths for the guest and does not upload your local certificates or token.
2. Prepare an Amazon Linux-compatible x86 GPU image with working drivers, `/opt/pytorch/bin/activate`, systemd, a C compiler and an SSM agent. Supply its AMI ID and actual root-device name explicitly. Provide an existing subnet with outbound package/S3/Secrets Manager access, a restricted security group allowing only intended clients on the API TLS port, and an instance profile. No SSH port is opened by this code.
3. Create a private S3 artifact bucket and a Secrets Manager secret with a JSON object containing `api_key`, `tls_cert` and `tls_key`. The token must be a random URL-safe string of at least 32 characters; the certificate chain and key are PEM strings, with escaped newlines in the JSON. Use a certificate trusted by clients for the hostname they will call. Store the secret through your own secure AWS tooling; never commit its values.
4. Have your IAM administrator grant the controller scoped S3 uploads, EC2 launch/tag/read/stop/start/terminate, PassRole for the serving profile, and SSM SendCommand/read permissions. The instance role needs GetObject for its source prefix, GetSecretValue for this secret, and SSM managed-instance permissions; customer-managed encryption keys also need suitable decrypt permissions. The role receives temporary credentials; controller keys never enter the bundle. This runner reuses existing infrastructure and does not create IAM roles, secrets, buckets or security groups.
5. Copy `configs/aws-serving.yaml` to ignored `configs/aws-serving.local.yaml`. Fill in the resource identifiers, profile/region, instance type, storage and `inference_config: configs/inference-public.local.yaml`. Set a realistic `hourly_estimate_usd` and `alert_after_usd`. These values are estimates, not price lookup or billing limits. Configure AWS Budgets notifications separately for recurring cost alerts.
6. Preview and deploy:

   ```bash
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --run
   ```

   Preview checks the local bundle without making service API calls. Deployment runs local tests/lint, uploads source/model with server-side encryption, launches on-demand EC2 with IMDSv2 and encrypted EBS, and records a deployment ID under ignored `artifacts/aws-serving/`. The guest fetches secret material into a protected directory and runs the API as a dedicated user. A successful launch response means provisioned, not API-ready; check the service logs and make an authenticated TLS request before sharing it.
7. Manage the recorded deployment:

   ```bash
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --action status --deployment HOSTING_ID
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --action logs --deployment HOSTING_ID --run
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --action stop --deployment HOSTING_ID --run
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --action start --deployment HOSTING_ID --run
   python -m modelchef.cli aws-serve configs/aws-serving.local.yaml --action delete --deployment HOSTING_ID --run
   ```

   Logs returns an SSM command ID; retrieve its output with `aws ssm get-command-invocation --command-id COMMAND_ID --instance-id INSTANCE_ID --profile YOUR_PROFILE --region YOUR_REGION`. Lifecycle commands report the requested transition; poll `status` to confirm it. Stop retains EBS and interrupts the API; start boots the installed service again. Delete terminates the host and deletes its configured root volume. Source objects, secrets and other pre-existing infrastructure remain.

Status reports EC2 state, public IP and a coarse estimate since `LaunchTime`; it can include stopped time and reset on start, so it is not a cumulative billing ledger. Crossing the threshold sets `budget_alert: true` but never shuts the API down. Storage and retained objects can continue costing money after stop or deletion. Public IPs can change after stop/start; update your hostname accordingly or manage a stable address externally. Secrets are fetched at deployment, not refreshed automatically; rotate token/certificates securely on the host or redeploy and retire the old host. Protect the local journal: if launch response delivery fails, use its deployment ID/client token to recover the instance before retrying.

Live hosting has not been deployed during development. Tests cover bundle boundaries, on-demand launch semantics, ownership checks and explicit lifecycle operations; cloud bootstrap, TLS reachability and model readiness still require a separately approved AWS run.

**Before sharing or scaling:** record the chosen recipe and evaluation protocol, establish kernel correctness before claiming speedups, and verify live model readiness separately from EC2 launch success. Native FP4, live managed cloud bootstrap and every cross-platform combination are not established by local unit tests. Keep training termination and persistent hosting lifecycle policies distinct.


