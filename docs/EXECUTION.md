# Choose where your recipe runs

The recipe describes the job. `--target` selects its location; omitting it means
`local`. `accelerator: auto` selects hardware on that machine and never provisions cloud compute.
Run commands from the repository root in your prepared Python environment.

## Start locally

```bash
python -m modelchef.cli pipeline configs/pretrain-demo.yaml
python -m modelchef.cli pipeline configs/pretrain-demo.yaml --run
python -m modelchef.cli cook configs/workflow-demo.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --run
```

`pipeline` runs one stage. `cook` runs ordered text stages, connects checkpoints and
optionally exports the last model. These demos use included data and tiny CPU models;
they do not download datasets or create cloud resources.

## Move the same recipe to AWS

Complete the [AWS credentials and IAM tutorial](AWS.md#prepare-aws-access).
Install the controller's test, pipeline and AWS dependencies and hardware-appropriate
PyTorch. Imported-model workflows also need their pretrained/adapter dependencies for
the local CPU probe. Guest dependencies are selected across all stage recipes.

```bash
cp configs/aws-training.yaml configs/aws-training.local.yaml
```

Fill in profile, region, instance type, runtime/budget limits, artifact bucket and
instance profile using the [managed training tutorial](AWS.md#run-a-managed-training-stage).
Then preview the same stage or workflow with a different target:

```bash
python -m modelchef.cli pipeline configs/pretrain-local.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli cook configs/workflow-local.yaml --target aws --aws-config configs/aws-training.local.yaml
```

The command-line recipe overrides `aws_test.pipeline_config`. Preview validates and
bounds the declared inputs, then queries AWS for the launch plan and conservative
cost reservation. It does not provision compute. Supply the local workflow's instruction
dataset before previewing; declared local files must exist for packaging.

Append `--run` only after reviewing the plan. Execution runs local tests/lint and a
bounded CPU preflight before launch. Workflow preflight checks the stage sequence,
checkpoint links and export. The controller uploads approved inputs and launches
**one Spot instance for the entire workflow**, with one reservation and overall deadline.
The guest runs the same stages with their recipe-selected execution settings.

Completion or failure attempts bounded result upload and shutdown. The controller
and AWS deadline also guard termination. Check both job success and `terminated` in
the report; transfer and cleanup can fail independently. Results remain an archive
under ignored `artifacts/aws-runs/`; inspect before extracting. Use `aws-reconcile`
with the recorded run ID and cloud configuration for result/cleanup recovery.

## Move the same recipe to Runpod

Follow the [Runpod setup tutorial](RUNPOD.md). Reuse CLI credentials or privately
configure `RUNPOD_API_KEY`, prepare SSH keys, and select an existing network volume.
Copy `configs/runpod.yaml` to `configs/runpod.local.yaml` and configure GPU/image,
volume ID, SSH paths, transfer caps, runtime and spending allowances.

```bash
python -m modelchef.cli pipeline configs/pretrain-demo.yaml --target runpod --runpod-config configs/runpod.local.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --target runpod --runpod-config configs/runpod.local.yaml
```

Preview is offline and validates the declared bundle; provide the workflow's local
instruction data first. Append `--run` for local source tests/lint, catalog/rate review,
spending confirmation and one Pod running the same training stages. Optional guest
dependencies follow the recipes, as on AWS.

The existing network volume holds acquired/processed data, packed tokens, tokenizer,
configuration, logs and checkpoints. The controller attempts Pod deletion on completion,
failure or timeout. **Keep it running and connected:** there is no independent
provider-enforced spending cutoff. Retained volumes continue billing after deletion.
Use `runpod-job RUN_DIR --action status`, `logs`, `fetch` or `terminate --run` with the
saved job directory. After Pod deletion, attach its volume to another Pod for recovery.
A read-only account connection and mocked lifecycle tests do not establish live GPU
training validation; run a small paid pilot before scaling.

## Boundaries

Before a billable launch, the CLI shows the provider estimate, configured runtime,
per-run reservation and overall budget, then asks you to type `yes`. Other responses
cancel without launching. For automation, explicitly add `--yes` after reviewing the
plan; this never bypasses budget/deadline controls. The review remains visible with
`--quiet`. Lower-level `aws-test` and `aws-train` commands use the same confirmation. AWS and Runpod maintain separate reservation ledgers.

- Cloud `--resume` is not supported. Recover a stage locally from retrieved artifacts
  with its matching configuration. `cook` does not automatically resume a whole workflow.
- `cook` remains a single controller. Run DDP stages separately through `pipeline`.
- Increase bundle/result caps deliberately for imported weights and multiple checkpoints.
- Managed Runpod execution currently covers training; persistent APIs use `aws-serve` with explicit stop/start/delete operations; a temporary
  training host never becomes a persistent API host.
- Lower-level `aws-train CONFIG` remains available and reads its recipe reference from
  the cloud configuration. It also supports multimodal stages; `cook` connects text stages.
- AWS contracts are tested with simulated calls. Live bootstrap and cloud GPU performance
  require a separately approved run.
