# Run the same recipe on Runpod

Runpod is an execution target for text training stages (`pipeline`) and linked workflows (`cook`). Data, architecture, precision, optimizer, DDP and custom CUDA choices remain in your existing recipe. A separate resource configuration selects the GPU, storage, credentials and spending allowances. ModelChef uses Runpod REST API v2 directly; its optional agent plugin and OAuth are not required.

Keep task-specific recipes, datasets and run outputs in your experiment project, such as `recipe-lab`. Install ModelChef from an editable checkout; its allowlisted source bundle is packaged separately from the experiment's declared inputs. This target currently manages temporary training Pods, not persistent vLLM API hosting.

## 1. Configure credentials once

ModelChef first reads `RUNPOD_API_KEY`, then reuses the key saved by the Runpod CLI in `~/.runpod/config.toml` (or its legacy `~/.runpod.yaml`). If your CLI is already configured, no separate login is required. A custom `api_key_env` uses only that environment variable, for explicit account selection.

Otherwise create a Runpod API key in [Credentials](https://console.runpod.io/user/settings) with the Pod read/write and catalog permissions required for training. Configure it locally. Do not paste it into chat, recipes, Git, shell command history or a Pod environment. The control-plane key remains on the local controller; only your public SSH key is sent to the Pod.

In PowerShell, this prompts without displaying the key and saves it in your Windows user environment:

```powershell
$secret = Read-Host 'Runpod API key' -AsSecureString
$credential = [System.Net.NetworkCredential]::new('', $secret)
[Environment]::SetEnvironmentVariable('RUNPOD_API_KEY', $credential.Password, 'User')
$env:RUNPOD_API_KEY = $credential.Password
Remove-Variable secret, credential
```

Windows user environment variables are persistent credentials, not an encrypted secret vault. For shared computers, use your organization's secret manager instead. A newly opened terminal inherits the variable. WSL is a separate environment: export the variable there using your secret manager, or enter it privately for the current session:

```bash
read -rsp 'Runpod API key: ' RUNPOD_API_KEY; echo
export RUNPOD_API_KEY
```

Install the OpenSSH client and configure an SSH key pair. You can reuse a suitable existing pair. Otherwise run `ssh-keygen -t ed25519` interactively and choose its location. ModelChef uses noninteractive key authentication; supply an agent-loaded key or a suitable dedicated automation key. Set the private/public key paths in the resource configuration. Private keys are never packaged.

## 2. Select persistent storage and resource limits

Create a Runpod **network volume** and record its ID. Creating and retaining a volume incurs charges. Select a GPU in that volume's data center. ModelChef requires the existing volume so checkpoints, tokenizer, recipes and logs survive Pod deletion, an interruption or loss of the local controller. It never deletes that volume automatically.

Copy [configs/runpod.yaml](../configs/runpod.yaml) into your experiment project and replace the volume ID. Review GPU count, image, CUDA floor, SSH paths, maximum runtime and budget. The example's dollar amounts are configuration ceilings, not advertised Runpod prices. A CUDA 13-capable host is required for ModelChef's pinned cu130 build. Its shared guest setup installs the same training dependencies and selects optional optimizer/kernel extras from the recipe.

The configured compute ceiling applies to **all GPUs together**. ModelChef checks the live catalog quote before launch and the actual Pod rate after creation. The storage allowance must cover your container disk and network volume during the run. Storage retained afterward continues billing outside that runtime allowance. AWS promotional credits do not fund Runpod usage.

## 3. Preview, then run

From the experiment directory, with ModelChef installed in your environment:

```bash
# Same recipe, different execution target. Local remains the default.
python -m modelchef.cli pipeline configs/my-stage.yaml --target runpod --runpod-config configs/runpod.yaml
python -m modelchef.cli pipeline configs/my-stage.yaml --target runpod --runpod-config configs/runpod.yaml --run

# Or run a complete linked lifecycle on one Pod.
python -m modelchef.cli cook configs/my-workflow.yaml --target runpod --runpod-config configs/runpod.yaml
python -m modelchef.cli cook configs/my-workflow.yaml --target runpod --runpod-config configs/runpod.yaml --run
```

Preview validates and bounds the bundle without an API key or Pod launch. Declared local inputs, parent checkpoints and custom kernels must fit `max_bundle_bytes`; the bundle excludes Git history and unrelated files. Large runs should acquire data through their recipe or use deliberately prepared bounded inputs. Neither the local API key nor other environment secrets are forwarded implicitly.

Execution shows the estimate and asks for confirmation before provisioning. `--yes` approves that reviewed launch for automation; budgets still apply. Local tests and lint run before creation by default. `local_preflight: false` skips those source checks explicitly; recipe/bundle validation and spending controls remain. Record your own pilot evidence before a large job.

## 4. Runtime, results and cleanup

The controller polls guest state, enforces the elapsed-runtime allowance and attempts Pod deletion on completion, failure or interruption. Guest preparation/training also has a timeout, but **stopping a process inside a Pod does not stop Runpod billing**. Keep the local controller and its network connection running. These are controller-enforced safeguards, not a provider-enforced hard spending cap. A killed controller or unreachable API can leave a billable Pod; use the saved run report or Runpod console to terminate it.

Results remain on the network volume by default. Explicitly set `download_results: true` to download into `artifacts/runpod-runs/RUN_ID/results`. See [cloud storage lifecycle](CLOUD_STORAGE.md) for explicit volume deletion commands. The report records the Pod ID, result status, termination confirmation and `/workspace/RUN_ID` recovery path. The entire job directory is on the network volume, including acquired data and processing outputs. Run artifacts contain the raw training snapshot, processed train/validation/test splits, packed token arrays, tokenizer, configuration and model/optimizer checkpoints; these are also included in the result archive. Result transfers are bounded by runtime, size and local disk space. If transfer fails, the network volume retains the job directory and already-written checkpoints. For recovery after Pod deletion, attach that same volume to another Pod and retrieve the saved directory. Cloud `--resume` is not supported in this first target; recover a compatible checkpoint and resume through the local pipeline interface.

```bash
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action status
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action logs
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action fetch
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action terminate
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action terminate --run
```

The termination preview is offline. Status, logs, fetch and termination use the API key. Logs and fetch additionally require the original SSH key and a live Pod; after deletion recover directly from the retained volume on another Pod. Fetch refuses to overwrite an existing results directory. Cleanup reconciles the unique saved job name and refuses conflicting Pod ownership. Ambiguous launch requests are not blindly retried. The budget ledger reserves the full per-run allowance conservatively, including unsuccessful/uncertain runs; it is not an exact Runpod bill or a cross-provider AWS/Runpod budget.

SSH uses first-connection host-key acceptance and pins the key in the job's `known_hosts` file for subsequent transfers. It does not independently attest the initial host. No public inference port or Jupyter server is started.

## Agent setup is optional

The official [agent setup guide](https://docs.runpod.io/agent-setup) installs skills and an optional Runpod MCP connection. ModelChef does not depend on them. An API-key MCP connection can read `RUNPOD_API_KEY`; no additional OAuth sign-in is necessary for that mode. Installed skills load on the next agent session. The guide's documentation-feedback MCP is separate at `https://docs.runpod.io/mcp`; connect it if you want documentation feedback submitted to Runpod.

References: [REST API v2](https://docs.runpod.io/api-reference-v2/overview), [live OpenAPI contract](https://api.runpod.io/v2/openapi.json), [SSH connections](https://docs.runpod.io/pods/configuration/use-ssh), [storage](https://docs.runpod.io/storage/overview).

Validation: lifecycle tests use mocked Runpod API and SSH responses, and exercise success, guest/transfer failure, launch reconciliation, cleanup failure, secret exclusion and archive traversal rejection. These are not evidence of a live GPU deployment. Verify a small paid pilot in your account before scaling.
