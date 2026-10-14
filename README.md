# ModelChef

**Pick your ingredients. Choose a recipe. Cook your model. Serve it.**

ModelChef is a recipe-driven platform for preparing data, training models and using the results. Build a text model from scratch, adapt a pretrained text, vision-language or speech model, evaluate it, then use CLI inference, publish to Hugging Face or serve supported text models through a secured vLLM API.

Versioned YAML recipes define the data, architecture and training stages. Run locally, on AWS or on Runpod, starting with a small test and progressing to precision, kernel and deployment controls as needed.

![ModelChef kitchen: datasets and a recipe become a trained model ready for serving.](docs/assets/modelchef-kitchen.png)

## Quick Start

The Quick Start examples below run on your current machine. ModelChef can also launch cloud GPUs: see [AWS training](#run-a-managed-training-stage), [Runpod training](docs/RUNPOD.md), or [persistent AWS API hosting](#persistent-api-hosting-on-ec2). Cloud launches require configured credentials, budget review and explicit execution with `--run`, and incur charges. Start with a small local experiment; the default training run tests the pipeline rather than producing a useful assistant.

Use the same stage or workflow recipe for each training target; provider resources stay in a separate file:

```bash
python -m modelchef.cli cook configs/workflow-demo.yaml                  # Local preview
python -m modelchef.cli cook configs/workflow-demo.yaml --run            # Local execution
python -m modelchef.cli cook configs/workflow-demo.yaml --target aws --aws-config configs/aws-training.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --target runpod --runpod-config configs/runpod.yaml
```

The cloud commands above preview; add `--run` to request a reviewed launch. `pipeline` accepts the same target options for one stage. Runpod uses an API key rather than an OAuth login, requires a persistent network volume for recovery, and relies on the local controller for Pod termination. Follow the [Runpod setup and budget tutorial](docs/RUNPOD.md) before launching.

Runpod authentication reuses `RUNPOD_API_KEY` or the credentials saved by its CLI. Configure the volume ID and SSH key paths in [configs/runpod.yaml](configs/runpod.yaml); keep keys out of that file. Use `runpod-job` to inspect status/logs, fetch available results or terminate an owned job. The integration has local lifecycle tests and a verified read-only account connection; a paid GPU training deployment remains to be validated.

Managed training recovery preserves the data as well as the model: raw snapshots, processed splits, packed token arrays, tokenizer and configuration. AWS checkpoints commit these files to S3, reusing unchanged content-addressed objects; Runpod writes them directly to its retained network volume. Resume from the last successfully persisted checkpoint, not from updates that were still in memory when compute stopped.

### Prepare your kitchen: set up once

With Git and [uv](https://docs.astral.sh/uv/getting-started/installation/) installed, run these commands in Linux, WSL or macOS Terminal. Windows GPU users should first follow the [WSL setup](#windows-with-an-nvidia-gpu-prepare-wsl2). The repository currently requires GitHub access.

```bash
git clone https://github.com/ashaytamhane/modelchef.git
cd modelchef
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,pipeline]'
python -m modelchef.cli setup
python -m modelchef.cli setup --apply
python -m modelchef.cli doctor
python -m modelchef.cli smoke
```

Routine input and dependency errors give concise messages; add `--debug` to show a traceback. `check-config` validates runtime settings, training recipes or linked workflow recipes without executing them. CLI commands report the work starting, meaningful stage changes and completion on stderr. They do not repeat timer messages while waiting; the last stage message identifies the work underway. Elapsed time is included at completion or failure. `doctor` explains its Python/PyTorch import, device discovery and GPU calculation checks as they happen. Use `--quiet` to suppress CLI progress; result output remains on stdout. Training retains its recipe-controlled detailed or brief narration.

`setup` previews the PyTorch installation; `--apply` installs it. Review the preview for your hardware. If prerequisites are missing, see [local setup](#local-setup). Package installation requires internet access.

`doctor` inspects Python/PyTorch versions, available accelerators and GPU memory. On CUDA, it also checks a small matrix multiplication. Use its output to confirm that the environment sees the device you intend to use.

`smoke` performs one training update on a tiny synthetic model, saves a temporary checkpoint, and verifies that model weights and optimizer state reload correctly. Look for `training_step: passed` and `checkpoint_recovery: passed`. Run it after `doctor` to verify that training and checkpointing work before starting a language-model experiment. Neither command downloads datasets or launches cloud resources.

### Choose your recipe

ModelChef uses **two levels of training recipe**:

- **Workflow recipe:** the overall training plan. It lists stages in order, references their recipe files, connects checkpoints and optionally exports the final model. Run it with `cook`.
- **Stage recipe:** the detailed settings for one stage: data, preprocessing, tokenizer/model, optimizer and evaluation. Its `stage.kind` selects `pretrain`, `midtrain`, `sft`, `dpo`, `rlvr` or `distill`. Run it individually with `pipeline`, or let `cook` run it as part of a workflow.

Prepare the files once; you do not edit a configuration between stages while `cook` is running. For a first experiment, start with one stage as shown below. For a complete training sequence, use [the workflow example](#combine-recipes-into-one-cook).

Copy a complete stage recipe:

```bash
cp configs/pretrain-local.yaml configs/my-run.local.yaml
```

Edit these fields **inside the existing sections** of the copied file, keeping the remaining settings:

```yaml
name: my-first-run
data:
  max_documents: 256       # Bound the corpus sample.
  max_tokens: 200000       # Bound packed data, independently of training steps.
tokenizer:
  vocab_size: 4096
model:
  layers: 6
  hidden_size: 512
  context_length: 128      # Tokens per training sequence.
training:
  accelerator: auto
  precision: auto
  steps: 40
  micro_batch_size: 1      # Reduce activation memory on a small GPU.
  gradient_accumulation: 4 # Effective batch size: 1 * 4 = 4 sequences.
evaluation:
  benchmark: auto
stage:
  kind: pretrain
```

This is an edit guide, not a replacement for the complete copied recipe. Validate with `python -m modelchef.cli pipeline configs/my-run.local.yaml`; add `--run` to execute it. Fields you omit use defaults; some become required when you select another workflow. See [required versus optional](#required-versus-optional). Unsupported fields and incompatible settings fail validation. `*.local.yaml` files stay out of Git. For fine-tuning, use `stage-config` in workflow 3 to generate a recipe matching the parent model instead of manually copying its architecture. See [published recipe patterns](#published-recipes-and-small-scale-adaptations) for starting points and the [complete annotated recipe](#complete-annotated-recipe) for every training field and its alternatives, and [data and recipes](#data-and-recipes) for local-data examples.

### Pick a published starting point

The included recipes also let you explore familiar training patterns at a manageable scale:

| Start with | When to choose it | Ingredients needed |
| --- | --- | --- |
| [Offline demo](configs/pretrain-demo.yaml) | First run: learn the commands and inspect artifacts | Included text; no data download |
| [SmolLM2-inspired](configs/recipe-smollm2-inspired.yaml) | Small-model pretraining with a bounded public corpus | FineWeb-Edu sample downloaded only with `--run` |
| [OLMo 2-inspired](configs/recipe-olmo-inspired.yaml) | Explore curated text and later domain stages | Your local `data/raw/corpus.jsonl` |
| [Llama 3-inspired](configs/recipe-llama-inspired.yaml) | Explore a small decoder using modern dense-model building blocks | Your local `data/raw/corpus.jsonl` |
| [Frontier-model patterns](#frontier-model-patterns-what-can-be-reproduced) | Combine dense/MoE training, instruction data, preferences or verifiable rewards | Separate datasets and recipes for each stage; published references and inferred choices are distinguished |

These are small adaptations, not reproductions of the original models. Copy a recipe to `configs/*.local.yaml`, adjust its inputs and limits, and preview it with `pipeline CONFIG`. The [published recipe guide](#published-recipes-and-small-scale-adaptations) links the original papers and explains what each adaptation preserves and changes. Choose the offline demo below if you are starting today.

### 1. Cook from scratch: try the included ingredients

```bash
python -m modelchef.cli check-config configs/pretrain-demo.yaml
python -m modelchef.cli pipeline configs/pretrain-demo.yaml --run
# Or pretrain, instruction-tune and export in one cook:
python -m modelchef.cli cook configs/workflow-demo.yaml --run
```

After installing dependencies, these examples work offline: they use [included, self-authored text and instruction pairs](examples/starter/README.md), a tiny two-layer CPU/FP32 model, four updates per stage and no external benchmark. They do not download datasets/models or launch cloud resources. The pipeline explains each step and prints the run directory; inspect `report.json` for evaluation and checkpoint-recovery results. This is a software demonstration, not a useful assistant.

For your next experiment, use the larger public-data recipe:

```bash
python -m modelchef.cli pipeline configs/pretrain-local.yaml
python -m modelchef.cli pipeline configs/pretrain-local.yaml --run

# Full sequence: pretraining, instruction tuning and final export.
# First provide data/raw/sft.jsonl with prompt/response records.
python -m modelchef.cli cook configs/workflow-local.yaml
python -m modelchef.cli cook configs/workflow-local.yaml --run
```

This acquires up to 256 FineWeb-Edu documents within a 16 MB corpus-response cap, trains BPE and an approximately 18.6M-parameter decoder for 40 steps, and evaluates a bounded benchmark sample. The first command previews; `--run` permits the configured downloads and training.

### 2. Taste the result: try your checkpoint

`RUN_DIR` means the completed directory printed after training, not a literal folder
name. In Linux/WSL/macOS, assign that path once, then use the variable in each command:

```bash
RUN_DIR="/absolute/path/to/modelchef/artifacts/training/YOUR_COMPLETED_RUN"
# Replace the path above with your actual completed run directory.
python -m modelchef.cli chat "$RUN_DIR" --prompt "Education helps people" --max-tokens 24
python -m modelchef.cli chat "$RUN_DIR"
python -m modelchef.cli export "$RUN_DIR"
```

Interactive mode supports `/reset` and `/exit`. Export creates `RUN_DIR/export/` for compatible inference tools. Expect poor text quality from this short training test. To serve the export to clients, continue with [vLLM API serving](#serve-your-model-inference-and-api); remote access requires TLS and authentication.

#### Try a checkpoint while training continues

You do not have to wait for the final report to inspect a saved native model. From
the repository root, in a second terminal using the same Python environment:

```bash
RUN_DIR="artifacts/training/YOUR_RUNNING_STAGE"
python scripts/chat_checkpoint.py "$RUN_DIR" --max-tokens 150
# Or generate one continuation and exit:
python scripts/chat_checkpoint.py "$RUN_DIR" --prompt "Once upon a time, a little rabbit" --max-tokens 100
```

Replace `YOUR_RUNNING_STAGE` with the stage directory containing `config.json` and
`checkpoints/`, not the parent workflow directory. In PowerShell, assign the variable
with `$RUN_DIR = "artifacts/training/YOUR_RUNNING_STAGE"`. For a WSL installation,
run these commands inside Ubuntu using its virtual environment.

The script loads the newest saved checkpoint on **CPU**, leaving GPU memory available
for training. It supports CPU-compatible native models, not Hugging Face, multimodal
or GPU-only custom-kernel checkpoints. Use `--checkpoint PATH` to select a particular
checkpoint. Type `/exit` to quit; restart to load newer weights. Prompts are independent
continuations: a TinyStories base model completes stories rather than following chat
instructions. The normal `modelchef.cli chat` command above requires a completed run.

To collect comparable samples automatically, use a separate terminal:

```bash
python scripts/sample_checkpoints.py "$RUN_DIR" --watch --every 1000
```

This generates two fixed-prompt continuations from each available checkpoint whose
optimizer step is a multiple of 1,000, including existing checkpoints, and watches
for new ones. It uses CPU, fixed seed, temperature 0.8, top-k 40 and up to 80 generated
tokens per prompt. Samples are appended to `RUN_DIR/checkpoint-samples.jsonl` and
printed on screen. Sampling adds CPU and disk work but does not update model weights
or the training random state. Stop the watcher with Ctrl+C; it also exits when it sees
a final report. This is an optional script, not an automatic training-loop feature.

Compare samples alongside validation loss in `metrics.jsonl`. Better grammar is useful
evidence, but repetition, inconsistent characters and memorization can persist even
as loss falls. This recipe learns its tokenizer and model weights from scratch;
TinyStories supplies synthetic training text, not pretrained model weights. Repeating
a recipe does not guarantee identical outputs when deterministic training is disabled.

### 3. Refine an existing model: fine-tune on your data

Use an existing compatible local Hugging Face model directory and a UTF-8 JSONL file containing `prompt` and `response` strings. Supply at least three distinct accepted examples (prefer more for useful split coverage), with answers short enough for the configured context. [Data preparation](#prepare-your-own-ingredients-data) covers text/CSV conversion; [pretrained acquisition](#start-from-a-pretrained-model) covers an explicit capped download if you do not already have a model.

```bash
uv pip install -e '.[pretrained]'
python -m modelchef.cli import-model LOCAL_MODEL_DIR --context 128
# Replace PARENT_RUN with the imported run directory printed above.
python -m modelchef.cli stage-config PARENT_RUN --kind sft --data data/raw/sft.jsonl --output configs/sft.local.yaml --steps 40
python -m modelchef.cli pipeline configs/sft.local.yaml
python -m modelchef.cli pipeline configs/sft.local.yaml --run
python -m modelchef.cli chat NEW_RUN_DIR
```

This keeps the pretrained tokenizer and weights and learns from your supplied responses. The explicit download/import workflow uses local model files during training. Alternatively, put a Hugging Face reference directly in a stage recipe, as shown below. Choose a model that fits available memory; this workflow defaults to full-parameter fine-tuning; see [optional LoRA](#parameter-efficient-fine-tuning-with-lora) to train adapters instead. Later sections cover domain midtraining, preferences, verifiable rewards, distillation and cloud execution.

#### Download a parent directly from a recipe

`stage.parent_run` accepts either an existing local run path or a bounded Hugging
Face reference. This works for midtraining, SFT, DPO, RLVR and the dedicated
distillation stage. `stage.teacher_run` accepts the same reference format for
optional token distillation. For example:

```yaml
stage:
  kind: sft
  parent_run:
    repo_id: hf-internal-testing/tiny-random-gpt2  # Required model ID.
    revision: main                             # Optional; a commit SHA is better for replay.
    max_download_bytes: 10000000                # Optional; default 500 MB.
    context_length: 64                         # Optional; defaults to recipe context.
```

Run the included tiny integration recipe:

```bash
uv pip install -e '.[train,pipeline,pretrained]'
python -m modelchef.cli pipeline configs/sft-huggingface-tiny.yaml
python -m modelchef.cli pipeline configs/sft-huggingface-tiny.yaml --run
python -m modelchef.cli cook configs/workflow-huggingface-tiny.yaml --run
```

Planning validates the reference without downloading weights. Execution checks
remote file sizes, enforces the download cap, pins the resolved commit, downloads
safetensors assets and imports the parent. Its architecture and tokenizer are
inherited automatically; the recipe still controls data, training and supported
adapter/runtime settings. Context must fit the parent model. Remote model code
is disabled. Gated/private models use your existing Hugging Face login; never
put a token in YAML. `model-references.json` records the requested reference,
resolved commit and checksums; resume uses the original import and rejects changed
references. Ordinary local run-path recipes remain supported.

The tiny model above is randomly initialized test material, not a useful pretrained
assistant. The test proves acquisition, import and training integration. For useful
behavior, choose a suitable trained model and supply representative stage data.

From-scratch `pretrain` does **not** accept parent weights: use `midtrain` for
continued pretraining of a downloaded model. A pretraining recipe may instead
specify a Hugging Face **teacher** with `tokenizer.algorithm: pretrained`; student
weights still start randomly. Workflow stage recipes can choose an explicit remote
parent, or link to an earlier stage with `parent_stage`, but cannot do both.
On managed cloud targets, model references are resolved on the controller and the pinned assets are
included in the training bundle; increase `max_bundle_bytes` for larger models.
Model references alone do not provision compute; cloud execution requires an explicit target and `--run`.

For images and speech, continue with the [multimodal tutorial](docs/MULTIMODAL.md). To share a completed export, follow the [Hugging Face publishing tutorial](docs/PUBLISHING.md). Both workflows use explicit recipes; publishing previews are offline and uploads require `--run`.

### Combine recipes into one cook

**The workflow YAML is your overall training recipe; the referenced stage YAML files hold the details.** For example, the included local workflow connects these files:

```text
configs/workflow-local.yaml             ← overall plan, run with cook
├── configs/pretrain-local.yaml         ← foundation stage: stage.kind = pretrain
└── configs/sft-recipe.yaml             ← instruction stage: stage.kind = sft
    └── final model export
```

`cook` passes the foundation checkpoint to the instruction stage automatically. Each stage retains its own dataset and training settings. You prepare the workflow and stage files before starting, then one command runs the sequence:

The example is still a short **pipeline test**, not a fully trained assistant. Its
pretraining stage selects at most 256 FineWeb-Edu documents; the SFT stage reads your
instruction file up to its own configured cap. Each stage has its own optimizer-step
budget. Training samples batches with replacement, so completion does not mean every
record was visited for several epochs. Adding stages or choosing a cloud target does not increase
those budgets automatically. Inspect held-out loss and stage reports before increasing
data and steps; expect incoherent text at these demonstration budgets.

```bash
python -m modelchef.cli cook configs/workflow-local.yaml
python -m modelchef.cli cook configs/workflow-local.yaml --run
```

Inspect the provided workflow and supply its local instruction dataset before executing it. The first command validates stage configurations and shows the plan; `--run` trains them in order. `cook` defaults to local execution and can acquire data allowed by its recipes. Add `--target aws --aws-config CONFIG` or `--target runpod --runpod-config CONFIG` to run the sequence on one temporary cloud host. It does not start an API. See [compose a complete workflow](#compose-a-complete-workflow) for the stage links and pretrained starting point, and [cloud execution](#cloud-execution) for both providers.

The overall recipe currently covers **text training stages, their evaluation and final export**. Hugging Face publishing and API serving use separate configurations and commands. Cloud resource/budget settings stay in a separate configuration selected with `--aws-config` or `--runpod-config`; the same training/workflow recipe runs on either target. Vision/speech training also uses its separate `multimodal` recipe rather than `cook`.

## The platform at a glance

For text training, the central workflow is a YAML recipe executed by `pipeline`: acquire a bounded sample, prepare and split documents, train or reuse a tokenizer, initialize or load a model, optimize it, evaluate it, and save a reproducible run. The script explains each step on screen and records the same explanations alongside the artifacts. Vision and speech use `multimodal` with their own recipes and pretrained processors.

There are two starting points. **From scratch** learns both the tokenizer and model weights from your corpus. **From a pretrained model** retains its tokenizer and weights so you can specialize it with much less training. Both can lead to later training stages, evaluation, CLI inference and export for vLLM.

| Goal | Starting point | Next workflow |
| --- | --- | --- |
| Learn or experiment with the complete training pipeline | Bounded FineWeb-Edu sample | Local pretraining, then inspect metrics and generated text |
| Adapt a model to a domain | Existing native or imported model | Midtraining on domain documents |
| Teach instruction following or a response style | Existing model and prompt/response pairs | Supervised fine-tuning (SFT) |
| Train from preferred answers | Existing model and chosen/rejected pairs | Direct preference optimization (DPO) |
| Train with automatically checkable answers | Existing model and a verifier | Experimental reinforcement learning with verifiable rewards (RLVR) |
| Transfer behavior to a smaller model | Teacher and student | Sequence or token distillation |
| Adapt a model to images or speech | Local vision-language or Whisper model and paired media/text | Multimodal fine-tuning, held-out evaluation and media inference |
| Share model weights | Complete local export and a model card | Preview, then publish to a private or public Hugging Face repository |
| Let clients call a text model | Compatible local export | Authenticated vLLM API; TLS for remote access |
| Increase execution control | A working baseline recipe | Compilation, attention kernels, custom CUDA, precision, MoE or single-host DDP |
| Move training to AWS or Runpod | Portable training recipe plus provider resources and credentials | Temporary Spot instance or Pod, with persisted data and checkpoints |
| Host a persistent API on AWS | Separate serving recipe and credentials | On-demand EC2 hosting with explicit stop/start/delete |

Start with the local pipeline test before increasing data, model size or cloud spend. A successful short run establishes that the software works; it does not produce a capable assistant or establish benchmark quality. Local CPU/CUDA paths, native LoRA objectives, CUDA QLoRA recovery/export, CPU DDP dense/MoE recovery and the example CUDA kernel have been exercised. Real-device macOS/ROCm, live managed cloud training, full standard suites and native Blackwell FP4 need separate validation.

## Choose your learning path

All three paths follow **data → model/tokenizer → training → evaluation → inference**. “Beginner” here means new to this platform, with ML fundamentals already familiar. Each level adds control over the same pipeline, rather than introducing a different tool.

| Cohort | What you will do | Result |
| --- | --- | --- |
| [Beginner](#beginner-run-and-understand-the-pipeline) | Use a bounded public sample and a provided recipe, inspect the run, recover a checkpoint and try CLI inference | A complete local experiment and an understanding of its artifacts |
| [Intermediate](#intermediate-adapt-models-to-your-data) | Prepare your own data, import or reuse a model, select a training objective, compare results and serve an API | A customized model workflow with explicit data formats and evaluation choices |
| [Advanced](#advanced-control-execution-and-deployment) | Use the full recipe, audit data assumptions, experiment with precision/kernels, run standard suites and manage cloud training/hosting | Reproducible experiments with deeper execution and lifecycle control |

Read [local setup](#local-setup) once. Follow one path or progress through all three. Use the [complete annotated recipe](#complete-annotated-recipe), [code architecture](#code-architecture), [troubleshooting](#troubleshooting) and [CLI reference](#cli-reference-and-repository-layout) as references rather than prerequisites.

### Know which recipe you are editing

Recipes describe different jobs; they are not interchangeable. Paths in the examples assume you run commands from the repository root. `RUN_DIR` is a completed experiment directory, while `CONFIG` is a YAML file.

| Recipe family | Example | Command | What it controls |
| --- | --- | --- | --- |
| Text training stage | `configs/pretrain-demo.yaml` | `pipeline CONFIG` | Data through evaluation for one pretrain, midtrain or post-training stage |
| Multi-stage text workflow | `configs/workflow-demo.yaml` | `cook CONFIG` | Stage order, parent/teacher links and final export |
| Vision or speech training | `configs/multimodal-vision.yaml` | `multimodal CONFIG` | Local paired media, pretrained model, training and evaluation |
| Runtime planning | `configs/local.yaml` | `resolve-runtime CONFIG` | Hardware and precision planning; does not train or provision |
| Local text API | `configs/inference.yaml` | `serve-vllm CONFIG` | Model export, generation settings and API security |
| AWS training | `configs/aws-training.yaml` | `pipeline` or `cook` with `--target aws --aws-config CONFIG` | Resources, transfer limits and budget; the CLI recipe takes precedence |
| Runpod training | `configs/runpod.yaml` | `pipeline` or `cook` with `--target runpod --runpod-config CONFIG` | GPU, network volume, transfer limits, runtime and spending allowances |
| Persistent AWS API | `configs/aws-serving.yaml` | `aws-serve CONFIG` | Serving recipe reference and explicit EC2 lifecycle |
| Model publication | `configs/publish-model.yaml` | `publish-model CONFIG` | Export files, Hub visibility and model-card information |

The commands above preview or resolve settings; add `--run` only to execution commands after reviewing the plan. `setup` uses `--apply` for installation. Other commands such as `chat`, `export`, `convert-data` and `predict-media` perform their work directly; do not assume every command is a dry run. Previews are not uniformly offline: model acquisition and Spot readiness previews can query remote metadata or AWS without downloading weights or launching compute.

**Progress at your own pace:** first complete an offline run and reload it; next change the data or objective and compare results; then export, publish or serve the model. Finally, measure an optimization against that working baseline or move the same job to AWS or Runpod. You can start directly with pretrained or multimodal fine-tuning when that matches your goal.


## Execution and architecture

Use `pipeline CONFIG` for one stage or `cook WORKFLOW` for a sequence. Both default to
local execution and accept `--target aws --aws-config CONFIG` or
`--target runpod --runpod-config CONFIG`. See the
[execution tutorial](docs/EXECUTION.md) for the same recipe on all three targets, preflight,
result recovery and shutdown behavior.

### High-level architecture

```mermaid
flowchart TD
    User[CLI user] --> Recipe[Versioned YAML recipe]
    Recipe --> Target[Execution target: local / AWS / Runpod]
    Target --> Local[Local machine or existing cloud host]
    Target --> AWS[Budgeted EC2 Spot runner]
    Target --> Runpod[Budgeted Runpod Pod runner]
    Runpod --> Pod[Controller-monitored temporary workload]
    Pod --> Data
    Pod --> Volume[Retained network volume: data and checkpoints]
    Pod --> PodCleanup[Controller attempts Pod deletion]
    Local --> Data[Bounded data acquisition and preparation]
    Data --> Scratch[Train BPE and initialize native decoder]
    Data --> Parent[Reuse a trained parent model and tokenizer]
    Download[Explicit capped pretrained download] --> Import[Register local HF model]
    Import --> Parent
    Scratch --> Train[Pretraining]
    Parent --> Stages[Midtraining / SFT / DPO / RLVR / distillation]
    Teacher[Frozen teacher or reference] --> Stages
    Train --> Outputs[Checkpoints, manifests, metrics and reports]
    Stages --> Outputs
    Outputs --> Eval[Held-out evaluation and domain benchmarks]
    Outputs --> Chat[CLI inference: PyTorch or vLLM]
    Outputs --> Export[HF-compatible export]
    AWS --> Guard[Local checks, budget ledger and termination deadlines]
    Guard --> Cloud[Temporary cloud workload]
    Cloud --> Data
    Cloud --> Recovery[S3: data and checkpoint snapshots]
    Cloud --> Cleanup[Collect evidence and confirm termination]
    Export --> Hosting[Separate persistent on-demand hosting]
    Hosting --> API[vLLM API managed by systemd]
    API --> Explicit[Explicit stop / start / delete]
```

Training runs where invoked by default. `pipeline` and `cook` provision temporary cloud compute only with an explicit cloud target, its resource configuration and `--run`. Run artifacts and downloaded models remain outside Git. Post-training objectives and pretrained import/SFT/export are covered by small CPU tests, with native stages also exercised on CUDA. Real-device macOS, live cloud training and native FP4 validation require their respective environments.

| Goal | Where execution happens | What to run | Cloud resources? |
| --- | --- | --- | --- |
| Pretrain on a bounded corpus sample | Current machine (local or an existing EC2 host) | `pipeline CONFIG --run` | No automatic provisioning |
| Talk to a trained checkpoint | Current machine | `chat RUN_DIR` | No |
| Export trained weights | Current machine | `export RUN_DIR` | No |
| Fine-tune a vision-language or Whisper model | Current machine with local model and media | `multimodal CONFIG --run` | No automatic provisioning |
| Run image or speech inference | Current machine | `predict-media CONFIG --media FILE` | No |
| Publish a complete model export | Current machine uploads to Hugging Face | `publish-model CONFIG --run` | No; Hub authentication required |
| Check your laptop/desktop GPU | Current machine | `doctor`, `resolve-runtime`, `smoke` | No |
| Develop code and run unit tests | Current machine | `python -m pytest -q`, `python -m ruff check .` | No |
| Install PyTorch or vLLM | Current machine | `setup --apply`, `setup-vllm --apply` | No |
| Serve an existing local model export | Current Linux/WSL CUDA machine | `serve-vllm ... --run` | No |
| Host an API on EC2 | Separate persistent on-demand host | `aws-serve CONFIG --run` | Yes; ongoing charges until explicitly stopped/deleted |
| Preview an EC2 test | Local controller makes read-only AWS requests | `aws-test` | Credentials, but no instance launched |
| Provision and test a cloud GPU | Local controller launches a temporary EC2 instance; tests run there | `aws-test ... --run` | Yes; incurs AWS charges |
| Launch a stage or full workflow | Local controller packages inputs; one temporary Spot host trains | `pipeline` or `cook` with `--target aws --aws-config CONFIG --run` | Yes; incurs AWS charges |
| Launch a Runpod stage or workflow | Controller packages inputs; one temporary Pod trains | `pipeline` or `cook` with `--target runpod --runpod-config CONFIG --run` | Runpod account, SSH and network volume; incurs charges |
| Inspect/recover a Runpod job | Controller queries the Pod and retrieves available results | `runpod-job RUN_DIR --action status` (or `logs`, `fetch`, `terminate`) | Runpod credentials; retained volume bills after Pod deletion |
| Recover AWS results/cleanup | Local controller queries AWS and removes temporary resources after termination | `aws-reconcile` | Yes; no new instance launched |

Commands run **where you invoke them by default**; `pipeline` and `cook` can explicitly target AWS or Runpod. If you SSH into an existing cloud host and run `doctor`, it inspects that host. It does not provision another machine. `pipeline` or `cook` with `--target aws --aws-config CONFIG --run` launches temporary Spot compute, as do `aws-test --run` and the lower-level `aws-train CONFIG --run`. `aws-serve CONFIG --run` launches a separate persistent on-demand serving host.

`configs/local.yaml` and `configs/aws.yaml` describe runtime/model planning settings. Selecting `aws.yaml` does **not** deploy anything. `configs/aws-test.yaml` is the separate EC2 provisioning/budget configuration.

## Local setup

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
git clone https://github.com/ashaytamhane/modelchef.git
cd modelchef
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,pipeline]'
python -m modelchef.cli --help
python -m pytest -q
python -m ruff check .
```

The repository is private, so GitHub authentication is needed to clone it. Keep WSL working files on the Linux filesystem rather than under `/mnt/c` for development performance. Installation downloads software packages, not training corpora or pretrained models. See [uv installation](https://docs.astral.sh/uv/getting-started/installation/) for other platforms.

### Windows-only controller or CPU development

A Windows controller does not need PyTorch or a GPU to launch AWS tests. With Python 3.12 and Git installed:

```powershell
git clone https://github.com/ashaytamhane/modelchef.git
cd modelchef
py -3.12 -m venv .venv-aws
.\.venv-aws\Scripts\Activate.ps1
python -m pip install -e '.[dev,aws]'
python -m modelchef.cli --help
python -m pytest -q
```

If PowerShell activation is unavailable, invoke `.\.venv-aws\Scripts\python.exe` directly instead of `python`. Local GPU inference through the automatic vLLM setup requires Linux/WSL NVIDIA CUDA.

### macOS: Apple Silicon or Intel MacBook

macOS runs the same Python codebase directly; WSL and CUDA are not required. The hardware detector selects PyTorch's MPS backend when available, otherwise CPU. Automatic precision selection uses FP32 on both. macOS setup paths exist, but a complete run has **not yet been validated on a real MacBook**.

On Apple Silicon, install Git and `uv` using your preferred method, then run these commands in Terminal. The repository requires GitHub authentication:

```bash
git clone https://github.com/ashaytamhane/modelchef.git
cd modelchef
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e '.[dev,pipeline]'
# Preview the macOS PyTorch installation before applying it.
python -m modelchef.cli setup
python -m modelchef.cli setup --apply
python -m modelchef.cli doctor
python -m modelchef.cli smoke --precision fp32
python -m pytest -q
```

Check `doctor` for `backend: mps` and `mps_available: true` to confirm GPU availability. An Apple Silicon Mac can use MPS when its OS and PyTorch build support it; The automatic installer targets Apple Silicon. Official modern PyTorch wheels are unavailable for Intel macOS; an Intel Mac can use an existing compatible source/community build for CPU training, or serve as a lightweight AWS controller. ModelChef does not reject an otherwise compatible existing Intel environment. See the [upstream Intel macOS announcement](https://dev-discuss.pytorch.org/t/pytorch-macos-x86-builds-deprecation-starting-january-2024/1690). If an operation fails on MPS, verify the CPU path explicitly:

```bash
python -m modelchef.cli smoke --accelerator cpu --precision fp32
```

To preview and then execute a bounded pretraining run on the Mac itself:

```bash
python -m modelchef.cli pipeline configs/pretrain-local.yaml
# This downloads the configured small corpus/benchmark samples and trains locally.
python -m modelchef.cli pipeline configs/pretrain-local.yaml --run
```

The recipe's `training.accelerator: auto` selects the available backend. To force CPU training, copy the recipe and set `training.accelerator: cpu` and `training.precision: fp32`. Reduce model dimensions, context length or microbatch size if memory is insufficient; hardware detection does not automatically resize the model.

After training, replace `PATH_TO_COMPLETED_RUN` with the run directory printed by the pipeline:

```bash
python -m modelchef.cli chat PATH_TO_COMPLETED_RUN --backend torch
```

The CLI supports `/reset` and `/exit`. A short training test produces limited text quality; SFT support does not make a tiny validation model a useful assistant. Use the PyTorch backend on macOS: the project's automatic vLLM installation targets Linux/WSL with NVIDIA CUDA. A Mac can also act as an AWS controller after installing `.[aws]` and following the AWS section.

### Hardware support and validation

Optional AWS accelerator integration: [Trainium training and Inferentia2 compilation](docs/NEURON.md)
use a separate Neuron environment. Start with the small feasibility recipe; the
existing CUDA workflows remain unchanged. Neuron hardware validation and full
AWS lifecycle integration are still pending.

| Environment | Training / PyTorch CLI inference | Automatic precision | Validation status |
| --- | --- | --- | --- |
| NVIDIA CUDA on Linux/WSL | CUDA | BF16 when reported supported; otherwise FP32 | Local CUDA pretraining and checkpoint/inference checks exercised |
| macOS with available MPS | Apple GPU through MPS | FP32 | Detection/setup implemented; real MacBook validation pending |
| CPU, including Intel Macs | CPU | FP32 | Small CPU pipeline and checkpoint tests exercised; not validated on macOS hardware |
| Compatible AMD GPU on Linux | ROCm | BF16 when reported supported; otherwise FP32 | Detection/setup implemented; real-device validation pending |

The automatic vLLM installer supports Linux/WSL NVIDIA CUDA only. Backend detection establishes availability, not compatibility of every operation or a guarantee that the configured model fits in memory.

### Install PyTorch and verify the runtime

Creating the base environment does not install PyTorch. First preview the installation:

```bash
python -m modelchef.cli setup
# Or deliberately choose a CPU-only environment:
python -m modelchef.cli setup --backend cpu
```

Review the detected backend, driver and proposed command. Install into the active environment only when ready:

```bash
python -m modelchef.cli setup --apply
python -m modelchef.cli doctor
python -m modelchef.cli check-config configs/local.yaml
python -m modelchef.cli resolve-runtime configs/local.yaml --output artifacts/environment/resolved-local.json
python -m modelchef.cli smoke
```

Expected results: `doctor` describes the available backend; on CUDA it also checks matrix multiplication. `smoke` reports `training_step: passed` and `checkpoint_recovery: passed`. These are synthetic checks, not language-model training or throughput measurements.

The default stack pins PyTorch 2.13.0 and NumPy 2.3.5. The default CUDA 13.0 installation plan requires NVIDIA driver major version 580 or newer. Setup installs packages, not drivers or OS prerequisites. For another GPU/driver combination, select an appropriate official build with `--torch-version` and `--index-url`; check the [PyTorch installation selector](https://pytorch.org/get-started/locally/). ROCm setup needs an explicit compatible wheel index; MPS requires macOS. Those backends still need real-device validation.

### Precision and memory basics

| Setting | Behavior |
| --- | --- |
| `accelerator: auto` | Available CUDA/ROCm, then MPS, otherwise CPU |
| `precision: auto` | BF16 when PyTorch reports support on CUDA/ROCm; otherwise FP32 |
| `fp16` | Explicit CUDA/ROCm training with gradient scaling |
| `fp32` | Default fallback; useful for CPU checks |
| Native NVFP4 training | `model.quantization: nvfp4`; native dense Blackwell/Transformer Engine integration |
| QLoRA | Imported-model CUDA NF4 base with trainable adapters |
| FP4 simulation | `model.quantization: fp4_fake`; experimental software fake quantization with FP32/BF16 master weights |
| Native NVFP4 inference | `inference.quantization: nvfp4_per_token`; supported Blackwell CUDA hardware and vLLM required |

FP16 and BF16 reduce the size of many activation tensors compared with FP32. BF16 retains a wider exponent range; FP16 training uses gradient scaling to reduce underflow risk. Mixed precision does not make all parameters or optimizer state low precision.

BF16 reporting can include emulated support; it does not establish native acceleration or performance. Explicit unavailable backends/precision fail rather than silently switching. The pretraining recipe uses microbatches of 2; runtime planning defaults to 1 when unspecified; automatic model-specific memory calibration is pending.

Examples:

```bash
# Debug checkpoint behavior without using the GPU.
python -m modelchef.cli smoke --accelerator cpu --precision fp32
# Check mixed-precision training on your current CUDA machine.
python -m modelchef.cli smoke --accelerator cuda --precision fp16
```

## Beginner: run and understand the pipeline

**Start with the ingredients:** complete one bounded experiment on your own machine. Start with the provided native model and public corpus recipe; no pretrained weights or AWS resources are required.

### Pick the ingredients: understand your data

`configs/pretrain-local.yaml` selects an ordered FineWeb-Edu sample through the public dataset viewer. The document, response-byte and token caps keep the experiment small; this is not representative random sampling or a full corpus download. Quality filters, regex PII handling and lexical deduplication run before seeded document splits. Benchmark sample text is also excluded by word n-gram overlap checks.

Keep the defaults for your first run. Previewing the recipe does not fetch data; `--run` does. To work offline, follow the intermediate local-document example after installing dependencies.


### Cook your first model: run training

Complete [local setup](#local-setup) and PyTorch installation, then install pipeline dependencies in the same environment:

```bash
uv pip install -e '.[dev,pipeline]'
python -m modelchef.cli doctor
# Validate every recipe section; this command performs no download or training.
python -m modelchef.cli pipeline configs/pretrain-local.yaml
# Download the bounded samples and execute on the current machine.
python -m modelchef.cli pipeline configs/pretrain-local.yaml --run
```

The default recipe requests at most 256 FineWeb-Edu documents, caps corpus responses at 16 MB and benchmark downloads at 2 MB, and caps packed data at 200,000 tokens. It creates an approximately 18.6M-parameter decoder and runs 40 optimizer steps. This is a pipeline test, not enough training for useful conversation. Dataset access uses the public Hugging Face viewer; no pretrained model weights are downloaded.

The pipeline explains each stage on screen: configuration/hardware, acquisition, benchmark selection, cleaning and document splits, BPE training, token packing, model construction, optimization, held-out evaluation and generation. `explain: brief` reduces console detail; the event log remains saved.

Each invocation creates its own directory under `artifacts/training/`. Inspect `report.json`, `processing.json`, source/benchmark manifests, `metrics.jsonl`, `environment.json`, tokenizer and checkpoints there. Data snapshots are cached under `data/cache/`. All run outputs and data caches stay excluded from Git.

For a live browser dashboard, install the optional monitoring extra and point it at
the actual run directory printed by training:

```bash
uv pip install -e '.[dashboard]'
python -m modelchef.cli dashboard artifacts/training/YOUR_RUN_DIRECTORY
```

TensorBoard opens at `http://127.0.0.1:6006`. Its Scalars view shows training loss,
validation metrics when evaluated, learning rate, gradient norm and throughput.
The Text view shows the full saved recipe, pipeline sequence, current reported
stage/activity and the generated sample after completion. `activity/current_status`
separates the pipeline stage from the optimizer step; preparation has step zero.
TensorBoard step numbers count optimizer updates, not stage numbers or log messages.
It follows existing
`metrics.jsonl` and `events.jsonl` files, so starting or stopping the dashboard
does not restart or stop training. It updates every five seconds as new records
arrive; it cannot show operations the run has not reported. GPU memory appears
only if recorded in the run metrics or final performance report. This is a text
pipeline overview, not an interactive architecture diagram.

Use `--port 6007` for another port, `--no-browser` on a remote machine, or `--once`
to export existing metrics without starting a server. The server binds to
localhost; access a remote dashboard through an SSH tunnel rather than exposing
the unauthenticated TensorBoard server publicly. This command monitors one stage
run directory at a time; point it at the next stage directory for a multi-stage
workflow.

#### What this run is doing

Byte-level **BPE** learns frequently occurring byte sequences as tokens, so the vocabulary can represent arbitrary text while compressing common patterns. It is fitted only on training documents. **Packing** joins tokenized documents into fixed-length sequences separated by EOS; this implementation allows attention across document boundaries within a packed sequence.

The native model is a decoder-only transformer: causal attention predicts each next token from preceding tokens. **RoPE** encodes relative position by rotating query/key components; **GQA** shares key/value heads across groups of query heads; **RMSNorm** normalizes activations; and **SwiGLU** gates the feed-forward activation. These are architectural choices exposed by the recipe, rather than independent training stages.

The run separates training, validation and test documents before fitting the tokenizer. Validation tracks optimization; test loss is a held-out check. Neither a loss decrease nor a generated sample establishes usefulness on an application task.

### Checkpoints and reproducibility

To start fresh weights using an existing disk-pretraining corpus, set the optional
`data.prepared_run` to its run directory. Data, processing, tokenizer, evaluation
and context settings must match. Both runs must use disk-based pretraining. ModelChef links the immutable snapshots on the
same filesystem and verifies corpus/token checksums; it does not import model
weights. This local reuse avoids another preprocessing pass and duplicate large
files. Use the original acquisition settings for a separate AWS run.

`training.checkpoint_every: 1000` saves every 1,000 optimizer steps. By default,
local checkpoints are retained; set `MODELCHEF_LOCAL_CHECKPOINT_KEEP=2` in the
training environment to keep the newest two after each successful save. Retention
removes older checkpoints, so copy a checkpoint first if you want to preserve it.

Use the original configuration and a saved checkpoint path:

```bash
python -m modelchef.cli pipeline configs/pretrain-local.yaml --run --resume PATH_TO_CHECKPOINT
```

Resume restores model, optimizer, scaler, sampler and random states and checks configuration/data/tokenizer fingerprints. The configuration must match exactly, and the original data snapshot must remain available. It continues toward the original step target; changing the target or dataset is a new experiment, not checkpoint recovery. CPU tests verify uninterrupted versus resumed weights agree; exact behavior across different hardware/software is not guaranteed.

### Taste-test the model: read evaluation results

Open the printed run directory and inspect `report.json`: compare `initial_validation` with final `validation`, inspect `test`, and check `checkpoint_recovery`. `processing.json` explains which documents were rejected, while source and benchmark manifests identify the exact saved inputs. `metrics.jsonl` records loss, learning rate, gradient norm and estimated throughput over optimizer steps.

Validation and test loss measure next-token prediction on this sample/tokenizer. The small task benchmark checks evaluation wiring; it is not a reliable application-quality estimate. Decide whether data preparation and optimization behaved as intended before increasing scale.


### Serve the result: interact or export

Replace `PATH_TO_COMPLETED_RUN` with the directory printed after successful training:

```bash
python -m modelchef.cli chat PATH_TO_COMPLETED_RUN --backend torch
python -m modelchef.cli chat PATH_TO_COMPLETED_RUN --prompt "Education helps people" --max-tokens 24
python -m modelchef.cli export PATH_TO_COMPLETED_RUN
```

Interactive mode supports `/reset` and `/exit`. This is a CLI completion/chat loop with bounded history, not a tool agent. A pretrained base model has not learned instruction following; the tiny test model will mostly produce incoherent text.

Export writes compatible Llama configuration, tokenizer and safetensors weights into the run's `export/` directory. On supported Linux/WSL CUDA hardware, install the optional inference stack using the [intermediate vLLM setup](#install-and-serve-with-vllm), then run:

```bash
python -m modelchef.cli chat PATH_TO_COMPLETED_RUN --backend vllm --max-tokens 24
```

The vLLM chat backend exports the trained weights if necessary and loads them locally. For a separate server, copy `configs/inference.yaml` to ignored `configs/inference.local.yaml`, point it at that export and use `serve-vllm`. On macOS or CPU, use the PyTorch chat backend.

**Before moving on:** you should have a completed run, understand its data limits and reports, and be able to load its checkpoint. A tiny run may still generate incoherent text. Continue to the intermediate path to change data or train instruction behavior.


## Intermediate: adapt models to your data

**Refine the recipe:** choose data and an objective for a concrete model behavior, then evaluate and serve the result. You can reuse the beginner checkpoint or start from a compatible pretrained model. These are alternative starting points; you do not need to pretrain a new model before fine-tuning an imported one.


### Data and recipes

Start with the recipe structure, then supply a small local dataset. Domain filtering and larger-corpus storage are optional next steps.

#### Understand and customize the recipe

Copy a provided YAML recipe to an ignored `configs/*.local.yaml` file before editing it. `configs/local.yaml` and `configs/aws.yaml` are runtime-planning configs; `configs/pretrain-*.yaml` and `configs/posttrain-*.yaml` are complete training recipes. `check-config CONFIG` validates runtime configs, training recipes and workflow recipes; `pipeline CONFIG` previews a training recipe. The strict schema rejects unsupported fields and incompatible settings; `configs/pretrain.schema.json` describes the full structure. Run `pipeline YOUR_CONFIG` to validate before executing.

Change one family of settings at a time. Document and byte caps control acquisition; the token cap controls packed data; optimizer steps control how often that data is sampled for updates. They are independent limits, so increasing steps can repeatedly train on the same small sample. For a single-device run, effective batch size is `micro_batch_size * gradient_accumulation`. Gradient checkpointing trades extra computation for lower activation memory; it does not reduce model or optimizer-state storage.

| Section | Controls |
| --- | --- |
| Run settings | Name, output directory, detailed/brief explanations |
| `data` | Source, subset, split, offset, document/byte/token caps, domain and keyword filter |
| `processing` | Unicode/HTML handling, quality limits, exact/near deduplication, PII policy, benchmark n-grams and document split fractions |
| `tokenizer` | Byte-level BPE vocabulary, minimum frequency, special tokens, BOS/EOS insertion |
| `model` | Layers, width, attention/KV heads, intermediate width, context, RoPE, RMSNorm, dropout, tied embeddings and gradient checkpointing |
| `training` | Device, precision, seed, determinism, optimizer steps, microbatch, accumulation, AdamW, schedule, clipping, checkpoints and runtime cap |
| `stage` | Training objective, parent/teacher runs, reasoning format, preference and reward settings |
| `evaluation` | Benchmark selection/sample/download cap, loss batches and generation controls |

From-scratch experiments use the native causal RoPE decoder with GQA, RMSNorm and SwiGLU, plus a trained byte-level BPE tokenizer. Imported Hugging Face models use their original architecture and tokenizer through the local pretrained adapter. Optional reasoning supervision, execution kernels and stage objectives are configured in the same recipe. The runtime limit is checked between training steps, not a deadline for the entire acquisition/evaluation workflow. Hardware detection does not automatically choose a model size that fits memory.

### Prepare your own ingredients: data

```bash
python -m modelchef.cli convert-data documents.txt --format text --output data/raw/documents.jsonl --max-bytes 10000000 --max-documents 1000
python -m modelchef.cli convert-data instructions.csv --format csv --output data/raw/sft.jsonl
```

Text conversion treats blank-line-separated paragraphs as documents. CSV headers become JSON field names: use `text` for midtraining or `prompt,response` for SFT. JSONL conversion also exists. Conversion preserves records; training performs objective-specific validation and filtering. PDFs, images, audio and arbitrary schemas require a separate extraction/mapping step. Outputs refuse to overwrite existing files. Choose source licenses and training rights appropriate to your intended use.

### Finance-focused experiment

```bash
python -m modelchef.cli pipeline configs/pretrain-finance.yaml
python -m modelchef.cli pipeline configs/pretrain-finance.yaml --run
```

This recipe searches a bounded FineWeb-Edu slice using explicit finance keywords. It is a domain-filtering experiment, not a curated financial corpus. Filtering can leave too few documents; increase the approved sample limit or supply appropriate local data if that occurs.

Automatic evaluation maps `general` to a small HellaSwag sample and `finance` to FinancialPhraseBank sentiment examples. Other domains require an explicit benchmark override or `benchmark: none`. Respect source licenses: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), [HellaSwag](https://huggingface.co/datasets/Rowan/hellaswag), and [FinancialPhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank), whose license includes noncommercial and share-alike restrictions.

Scores use this project's custom zero-shot continuation-scoring protocol, not official lm-evaluation-harness scores. A handful of examples tests evaluation wiring; it cannot establish model quality. Only the evaluated sample is automatically included in benchmark decontamination.

### Train on your own local documents

Place JSONL data under ignored `data/raw/`, with one object per line containing a `text` field. Copy the local recipe and change:

```yaml
data:
  provider: local_jsonl
  local_path: data/raw/documents.jsonl
  text_field: text
  domain: general
  max_documents: 256
  max_download_bytes: 16000000
  max_tokens: 200000
evaluation:
  benchmark: none
```

Edit these sections in the copied complete recipe; keep the other settings. `benchmark: none` avoids benchmark network access. Install dependencies beforehand for an offline run. Supply enough distinct documents for filtering and all three splits, and enough tokens to form context blocks. Tokenizer merges are learned only from the training split.

Regex PII removal and lexical deduplication are limited heuristics. Preserve the cached raw snapshot and its SHA256 for replay: the viewer API is mutable, and the manifest does not pin an upstream Git revision. Initial slices are bounded samples, not representative random samples of the full corpus.

### Scale your local corpus

For larger local corpora, copy a complete pretraining recipe and change its `data`
section as follows (these are overrides, not a standalone recipe):

```yaml
data:
  provider: local_jsonl
  local_path: null
  local_paths: [data/train-000.jsonl, data/train-001.jsonl]
  storage: disk
  max_documents: 100000
  max_download_bytes: 1000000000
  max_record_bytes: 4000000
  max_tokens: 10000000
```

Shard order and offset define the selected snapshot. Disk mode streams records through
SQLite deduplication, writes deterministic disjoint splits and stores uint32 tokens in
memory-mapped files. Training allocates only selected token batches; resume verifies the
mapped files before using them. It supports local JSONL pretraining/midtraining and DDP.
Use `storage: memory` for structured post-training objectives. Near deduplication uses
approximate LSH candidate search; tokenizer learning and benchmark indexes still use RAM.
Set byte/token/runtime limits deliberately and allow disk space for snapshots and indexes.
Managed AWS training carries these settings into the guest, with the additional bundle cap.
See [quality and acceptance criteria](docs/QUALITY.md) for scale and hardware validation boundaries.


### Start from a pretrained model

Install the optional adapter dependencies alongside your PyTorch environment:

```bash
uv pip install -e '.[pipeline,pretrained]'
# Metadata-only preview: pins the resolved upstream revision and checks total file size.
python -m modelchef.cli download-model HuggingFaceTB/SmolLM2-135M --max-bytes 500000000
# Explicitly execute only after reviewing model license, size and disk/memory requirements.
python -m modelchef.cli download-model HuggingFaceTB/SmolLM2-135M --max-bytes 500000000 --run
python -m modelchef.cli import-model PATH_TO_DOWNLOADED_MODEL --context 128
```

Import registers local model weights and tokenizer as a parent run; it does not train them. Use the printed imported run directory as `PATH_TO_PARENT` in the next section. `midtrain` uses your raw text; `sft` uses your instruction/response data; DPO/RLVR use their corresponding formats. This reuses the pretrained tokenizer and weights, rather than starting from random initialization. Models supported by Transformers' causal-language-model auto classes with fast tokenizers can use the adapter; remote model code and implicit model downloads are disabled. Compatibility and memory use still depend on the selected architecture. A local tiny GPT-2 fixture tests import, SFT and export; this is not a claim that every Hub model has been validated.

The downloader selects safetensors weights and tokenizer/configuration assets, records resolved revision and file hashes, and excludes Python code and pickle weight files. Downloads stay under ignored `artifacts/pretrained/`. An existing local HF export can be imported directly without network access. Imported models keep a dependency on that local source directory; retain it for future stages and exports.

Hugging Face supplies dataset/model distribution, tokenizer tooling and a compatibility format. The native model and training loop are project code; starting from scratch does not load pretrained weights. Imported models use the optional Transformers adapter.

### Refine your model: continue training

Midtraining keeps the next-token objective but changes the corpus, for example to domain documents. SFT instead learns supplied responses while masking prompt tokens from the loss. DPO increases the relative likelihood of a chosen response compared with a rejected response, using a frozen reference to anchor the change. RLVR samples answers and obtains a reward from a deterministic verifier; it needs the model to produce enough varied, sometimes-correct answers for a useful learning signal.

These are separate experiment choices, not a mandatory sequence. Full-parameter fine-tuning is the default. Native and imported models support optional LoRA across continuation stages; imported models additionally support CUDA NF4 QLoRA. Choose the objective and adaptation method independently within the supported combinations.

Use a completed or imported run as the parent. `stage-config` copies its architecture and tokenizer settings, avoiding manual mismatches. Write personal recipes to ignored `configs/*.local.yaml` files:

```bash
python -m modelchef.cli stage-config PATH_TO_PARENT --kind sft --data data/raw/sft.jsonl --output configs/sft.local.yaml --steps 40
python -m modelchef.cli pipeline configs/sft.local.yaml
python -m modelchef.cli pipeline configs/sft.local.yaml --run
python -m modelchef.cli chat PATH_TO_NEW_COMPLETED_RUN
```

Each stage creates a new run and optimizer; it retains the parent's weights and vocabulary. `--resume` recovers a stage using that stage's own saved optimizer and random states. Parent and teacher checkpoint hashes are recorded and checked during recovery. Parent artifacts must remain available.

| Stage / `--kind` | JSONL record fields | Objective |
| --- | --- | --- |
| `midtrain` | `text` | Continued next-token cross entropy on specialized documents |
| `sft` | `prompt`, `response` | Response-only cross entropy; prompt/padding loss masked |
| `dpo` | `prompt`, `chosen`, `rejected` | Preference objective against a frozen parent reference |
| `rlvr` | `prompt`, `answer` | Sampled group-relative policy objective and reference KL with exact-match or integer rewards |
| `distill` | `prompt`, `response` | Teacher-distribution KL mixed with response cross entropy |

Examples, one per JSONL line:

```json
{"prompt":"What does a bank do?","response":"A bank accepts deposits and provides financial services."}
{"prompt":"What is 2 plus 3?","chosen":"5","rejected":"6"}
{"prompt":"What is 2 plus 3?","answer":"5"}
```

Keep separate files for each format. At least three distinct accepted records are required to form document splits. Structured stages use `User: ...` / `Assistant:` formatting. They reject overlong examples rather than silently truncate answers. Duplicate prompts are removed before splitting. Existing PII/quality/decontamination heuristics still apply; provide benchmark files for your evaluation data.

Copy `configs/posttrain-*.yaml` for full native defaults, or prefer `stage-config` for the actual parent. Edit `stage.dpo_beta`, reward/verifier, group size, rollout cap, KL coefficient and optimizer settings deliberately. RLVR is a small GRPO-inspired implementation with one update per sampled group, not a distributed production RL system. Equal rewards give zero policy advantage; mostly wrong samples may provide no useful learning signal. The integer verifier accepts a complete signed integer response, not arbitrary reasoning text or executable code. The DPO and group-relative objectives follow the ideas in [DPO](https://arxiv.org/abs/2305.18290) and [DeepSeekMath](https://arxiv.org/abs/2402.03300).

#### Parameter-efficient fine-tuning with LoRA

LoRA freezes the pretrained weights and trains small low-rank updates in selected layers. This reduces trainable parameters and optimizer memory while preserving the base model. See the original [LoRA paper](https://arxiv.org/abs/2106.09685) and [Hugging Face PEFT documentation](https://huggingface.co/docs/peft/en/package_reference/lora).

Install `uv pip install -e '.[pipeline,pretrained,peft]'`. Generate a continuation-stage recipe with `stage-config` as above, then add this block inside its existing `model:` section:

```yaml
  peft:
    method: lora
    rank: 8
    alpha: 16
    dropout: 0.0
    target_modules: [q_proj, v_proj]
    use_rslora: false
```

Target names depend on the architecture: the native decoder and Llama-style models use `q_proj`/`v_proj`; GPT-2 uses `c_attn`. Validate and run using the same `pipeline` commands. The report includes total and trainable parameter counts. PyTorch CLI chat restores the adapter automatically. `export RUN_DIR` writes separate adapter files under `RUN_DIR/adapter/` and a merged dense model under `RUN_DIR/export/` for vLLM.

Checkpoints retain the full base and adapter state for reproducible resume, so they are not adapter-sized. Keep the original local model directory. Continuing an adapter requires the same adapter settings; changing rank or targets requires exporting, importing the merged model, and starting a new adapter. LoRA supports native and imported models for midtraining, SFT, DPO, RLVR and distillation. It requires an existing parent rather than adapter-only pretraining from random weights. Native adapters use the built-in implementation; imported adapters use PEFT.

#### QLoRA: adapters on a 4-bit base

QLoRA keeps the base in NF4 storage and trains LoRA adapters, reducing base-weight memory as well as optimizer memory. NF4 is a quantization format; it is different from native NVFP4 training arithmetic. See the [QLoRA paper](https://arxiv.org/abs/2305.14314) and [PEFT quantization guide](https://huggingface.co/docs/peft/en/developer_guides/quantization).

Install `uv pip install -e '.[pipeline,pretrained,qlora]'` on a compatible NVIDIA CUDA environment. In a parent-derived Hugging Face continuation recipe, set:

```yaml
model:
  peft:
    method: qlora
    rank: 8
    alpha: 16
    target_modules: [q_proj, v_proj]
    compute_dtype: bf16     # OPTIONAL: bf16 | fp16; choose supported compute precision.
    double_quant: true     # OPTIONAL: quantize the quantization scales too.
```

This fragment edits the existing recipe, not its complete replacement. QLoRA supports the same continuation objectives as LoRA. The original model and quantization metadata remain necessary for recovery. Export merges adapters into a dense HF model for vLLM; allow memory for that full-precision export. The CUDA path has been exercised with a tiny local GPT-2 fixture, including training, export and resume. It does not imply every pretrained architecture or bitsandbytes build is compatible. QLoRA is currently single-device and cannot be combined with projection FP4 simulation or native NVFP4.

#### Optional reasoning supervision

Reasoning supervision defaults to `stage.reasoning_format: plain`. For SFT or distillation, select `think_answer` and supply JSONL records with `prompt`, `reasoning` and `answer` fields. The prepared response contains `<think>…</think>` followed by `<answer>…</answer>`. Allocate enough context for both sections. This trains the supplied explanation format; it does not establish reasoning quality. With verifiable-reward training, the optional format checks the final answer inside a single answer tag.

For example, an SFT record can be `{"prompt":"What is 2 plus 3?","reasoning":"Add three to two.","answer":"5"}`. Derive a recipe with `stage-config PATH_TO_PARENT --kind sft --reasoning think_answer --data data/raw/reasoning.jsonl --output configs/reasoning.local.yaml`. This is optional supervision of an explanation and final-answer format, not a different transformer architecture or a guarantee that the explanation is faithful.


#### Distillation

Distillation is optional during **pretraining, midtraining and SFT**, in addition
to the dedicated `distill` stage. Set `stage.teacher_run` to a completed local
teacher run to enable online token distillation; omit it for normal training.
Existing `stage.distillation_temperature` and `stage.distillation_alpha` control
the temperature and teacher-loss weight: total loss is `alpha * T² * KL +
(1 - alpha) * cross_entropy`. Both losses use all target tokens during
pretraining/midtraining and only response targets during SFT.

For example, add this to a from-scratch native pretraining recipe:

```yaml
tokenizer:
  algorithm: pretrained  # Reuse teacher vocabulary; student weights remain random.
stage:
  kind: pretrain
  teacher_run: artifacts/training/TEACHER_RUN
  distillation_temperature: 2.0
  distillation_alpha: 0.5
```

For midtraining or SFT, retain the parent recipe's tokenizer settings and
`stage.parent_run`, then add `stage.teacher_run`. Teacher and student tokenizers
and special-token identities must match exactly, and teacher context must cover
student context. The teacher stays frozen. Checkpoints record its checksum and
resume rejects a changed teacher. AWS bundles include the teacher's assets, so
the same recipes work through the AWS runner; choose enough GPU memory for both
models. `teacher_kl` appears in saved metrics and the dashboard.

Online teacher loss is rejected for DPO/RLVR. Teacher-generated examples can
still supply supervised data or initialize a policy, but those techniques are
not the same as adding KL distillation to the preference/reward objective.

Two methods are available:

1. **Sequence distillation:** generate responses from a local teacher, then SFT the student on those responses. Teacher and student may have different tokenizers.
2. **Token distillation:** optimize student logits against a frozen teacher, with temperature-scaled KL and configurable hard-label mixing. This requires identical tokenizer contents and special-token identities; architectures may differ, and the teacher context must cover the student context.

For sequence distillation, create a JSONL file with `prompt` strings:

```bash
python -m modelchef.cli teacher-data PATH_TO_TEACHER --prompts data/raw/prompts.jsonl --output data/raw/teacher-sft.jsonl --max-documents 32 --max-tokens 32
python -m modelchef.cli stage-config PATH_TO_STUDENT --kind sft --data data/raw/teacher-sft.jsonl --output configs/sequence-distill.local.yaml
python -m modelchef.cli pipeline configs/sequence-distill.local.yaml --run
```

For a new smaller native student, run `python -m modelchef.cli init-student PATH_TO_TEACHER` first. It creates random weights and copies the teacher vocabulary; use the printed directory as `PATH_TO_STUDENT`. `--architecture FILE` accepts an architecture YAML mapping. A large teacher vocabulary can still dominate student parameter count.

For token distillation:

```bash
python -m modelchef.cli stage-config PATH_TO_STUDENT --kind distill --teacher PATH_TO_TEACHER --data data/raw/sft.jsonl --output configs/distill.local.yaml
python -m modelchef.cli pipeline configs/distill.local.yaml --run
```

Configure `stage.distillation_temperature` and `stage.distillation_alpha` in the recipe. Alpha weights the soft teacher loss; the remaining weight applies to response cross entropy. A weak teacher can teach poor behavior; a short execution test does not establish distillation benefit.

KL divergence measures the mismatch between the student's and teacher's token distributions. A higher temperature softens those distributions so alternatives to the most likely token carry more signal; the implementation scales the soft loss by temperature squared. This requires the same token IDs to mean the same things, which is why token distillation enforces identical tokenizers. Sequence distillation transfers generated text instead and avoids that constraint.

### Compose a complete workflow

The workflow is the overall recipe; each `stages[].recipe` points to a detailed stage recipe. Add stage entries to extend the sequence, and set the objective and dataset in the referenced files. You do not put one global `stage.kind` in the workflow: each stage file has its own. `cook` joins these independently reviewable recipes and connects their resulting runs automatically.

Paths are relative to the project root. A stage normally inherits the preceding model geometry and tokenizer, while explicit adapter and execution choices can override inherited settings. Here is the composition shape; use the checked-in example for execution:

```yaml
schema_version: 1               # OPTIONAL: only version 1.
name: my-model                  # OPTIONAL: artifact prefix.
output_dir: artifacts/workflows # OPTIONAL: workflow reports.
imported_model: null            # OPTIONAL: local HF directory instead of a scratch start.
imported_context: 128           # OPTIONAL: context for an imported starting model.
export: true                    # OPTIONAL: export the final completed stage.
stages:                         # REQUIRED: at least one ordered stage.
  - name: base                  # REQUIRED: unique stage name.
    recipe: configs/pretrain-local.yaml # REQUIRED: training recipe path.
  - name: instructions
    recipe: configs/sft-recipe.yaml
    parent_stage: base          # OPTIONAL: earlier stage; otherwise previous output.
    teacher_stage: null         # CONDITIONAL: earlier teacher stage for distillation.
    inherit_architecture: true  # OPTIONAL: preserve parent geometry/tokenizer.
```

The workflow persists its plan and a report linking each stage to its run. Failed execution records the error and preserves completed artifacts. Recover an individual stage with its checkpoint using `pipeline --resume`; `cook` does not yet resume the entire workflow automatically. Run DDP stages separately with `torchrun`. Training, export and API launch remain explicit operations: use [AWS training](#run-a-managed-training-stage) or [API serving](#serve-your-model-inference-and-api) when those are your chosen execution environments.

### Taste-test and compare: evaluation

Perplexity is the exponentiated mean next-token loss: lower is better on the same held-out data and tokenization, but it is not a fair cross-tokenizer quality ranking. Task benchmarks check specific behavior; use the same dataset/protocol and report sample size, context and truncation. DPO stages additionally report preference accuracy, while RLVR reports verifier-based accuracy.

After downloading/importing any approved baseline models separately:

```bash
python -m modelchef.cli compare PATH_TO_COMPLETED_RUN --baseline PATH_TO_LOCAL_GPT2 --baseline PATH_TO_LOCAL_PYTHIA --baseline PATH_TO_LOCAL_SMOLLM2 --output artifacts/comparison.json
```

This uses the run's saved benchmark examples for every candidate, retaining each model's own tokenizer. Baselines can be completed ModelChef runs or local HF exports. Custom continuation scoring differs from published-suite protocols; tokenizers and context/truncation choices affect comparisons. No baseline weights are fetched by `compare`.

### Serve your model: inference and API

Use the candidate run you evaluated above. Start with the PyTorch CLI to check model behavior, then export it for vLLM. The inference engine changes execution and serving; it does not improve learned model behavior by itself.

#### Install and serve with vLLM

Use the Linux/WSL `.venv` from local setup, a compatible NVIDIA GPU/driver, and a C compiler:

```bash
python -m modelchef.cli setup-vllm
python -m modelchef.cli setup-vllm --apply
python scripts/vllm_smoke.py
```

The installer targets `.venv` in the project root and pins vLLM 0.31.0 with the shared PyTorch/CUDA stack. The smoke script generates from tiny random local weights; it does not test a useful trained language model. Other inference backends need their own [vLLM installation recipe](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

vLLM is an inference engine that manages generation and batching; it does not replace the PyTorch training loop. This setup reuses the project environment.

Set `MODELCHEF_API_KEY` to a random token of at least 32 non-whitespace characters before starting even a loopback server; it is required at launch.

To serve an existing compatible Hugging Face export, set `inference.model_path` in `configs/inference.yaml` to a local directory containing `config.json`, `tokenizer.json` and safetensors or `pytorch_model.bin` weights:

```bash
python -m modelchef.cli serve-vllm configs/inference.yaml
python -m modelchef.cli serve-vllm configs/inference.yaml --run
```

The first command validates files and prints a plan. The second starts foreground serving at `http://127.0.0.1:8000/v1`. Stop it with Ctrl+C. Downloads and remote model code are disabled. A `.pt` checkpoint alone is insufficient; use `export RUN_DIR` to create the compatible model directory. Default memory utilization is 50%, context 2048 and scheduler sequence limit 4; the API boundary separately allows 2 concurrent requests. Set `max_model_len` no higher than your trained context (128 for the local test recipe); adjust for your model, and avoid simultaneous training/serving on a small GPU.

#### Share the API with other clients

Serving runs on the machine where you invoke it: a prepared Linux/WSL CUDA laptop, an existing EC2 host, or another compatible vLLM host. It does not provision a machine. Keep the process running while clients use it; AWS running costs continue for the lifetime of the host.

1. Export a completed run: `python -m modelchef.cli export RUN_DIR`. Use the printed export directory as `inference.model_path`, and set `max_model_len` to the trained context or less. Install the optional vLLM environment using the [vLLM setup instructions](#install-and-serve-with-vllm).
2. Obtain a trusted TLS certificate for the hostname clients will use. Keep its full certificate chain and private key outside Git, restrict the private key's file permissions, and put their paths in a copy of `configs/inference-public.yaml`, named `configs/inference-public.local.yaml`. Do not disable certificate verification on clients.
3. Set `MODELCHEF_API_KEY` in the serving process environment to a cryptographically random token of at least 32 characters. For example, generate a token with `python -c 'import secrets; print(secrets.token_urlsafe(48))'` and load it through your secret manager or a private environment file. Do not put the token in YAML, shell command arguments, Git or shared logs. All clients holding this token share access; rotate it by changing the environment and restarting the server.
4. Preview: `python -m modelchef.cli serve-vllm configs/inference-public.local.yaml`. Then start: `python -m modelchef.cli serve-vllm configs/inference-public.local.yaml --run`. The default local config binds loopback; the public example binds port 8443 with TLS. A missing token or missing TLS files prevents startup. Plans contain the secret's variable name, never its value.
5. Allow inbound TCP 8443 only from intended client networks in your host firewall and, on EC2, its security group. For a laptop, WSL/Windows forwarding and your router or VPN must make that port reachable. The code does not open firewall rules or router ports. Prefer a private VPN for a small group.
6. Clients send `Authorization: Bearer <token>` to `https://YOUR_HOSTNAME:8443/v1/models` or POST text requests to `/v1/completions` and `/v1/chat/completions`. For example, a completions body is `{"model":"artifacts/exports/modelchef","prompt":"Explain gravity","max_tokens":64}`; use the model ID returned by `/v1/models`. Chat requires a suitable model chat template and instruction training; a base checkpoint can use completions.

The middleware in `src/modelchef/api_security.py` authenticates every exposed HTTP request and blocks all routes except these three, including administrative routes and alternate inference paths. It enforces bounded request bodies, process-wide rate/concurrency limits, timeouts, and text-only inputs; request prompt logging and interactive API docs are disabled. TLS files and tokens stay local. Native vLLM API keys alone do not cover all routes; see the [vLLM serving security guidance](https://docs.vllm.ai/en/latest/cli/serve/).

These controls are a secure starting point, not a guarantee of complete security. This mode supports one API process and a shared token; it does not provide individual identities, billing quotas, distributed rate limits, or protection against network-level denial of service. For public multi-user production access, place it behind a managed gateway with identity, quotas and network protection, keep the backend private, and review the deployment. No public endpoint is opened automatically.

**Before moving on:** retain a baseline and a candidate run, compare them on held-out data with a stated protocol, and verify inference uses the intended checkpoint. A serving process runs on its current host; starting vLLM is not an AWS deployment. The advanced path covers managed cloud jobs and persistent hosting.


### Add vision or speech ingredients

Use `configs/multimodal-vision.yaml` for image/prompt/response fine-tuning, or
`configs/multimodal-speech.yaml` for Whisper audio/transcript training. Set the local
model and manifest paths, then preview and execute:

```bash
cp configs/multimodal-vision.yaml configs/vision.local.yaml
# Edit model.path and data.manifest before running these commands.
python -m modelchef.cli multimodal configs/vision.local.yaml
python -m modelchef.cli multimodal configs/vision.local.yaml --run
python -m modelchef.cli predict-media configs/vision.local.yaml --model-path artifacts/multimodal/YOUR_RUN/export --media data/vision/example.png --prompt "Describe this image."
```

The [multimodal tutorial](docs/MULTIMODAL.md) explains manifests, every required/optional
field, assistant-only labels, encoder freezing, media bounds, grouped evaluation,
checkpoint recovery, exports and AWS packaging. Vision/speech use pretrained processors;
native scratch training remains text-only. Speech currently supports recognition and
multilingual Whisper translation, not speech synthesis.

### Share the finished model

Copy `configs/publish-model.yaml` to a local recipe and set the export, Hub repository,
license, training description, evaluation and limitations:

```bash
cp configs/publish-model.yaml configs/publish.local.yaml
# Edit export_dir, repo_id and the model-card fields before previewing.
python -m modelchef.cli publish-model configs/publish.local.yaml
# After reviewing the exact file list/card and authenticating with hf auth login:
python -m modelchef.cli publish-model configs/publish.local.yaml --run
```

Publishing defaults to a private repository and uploads only model/processor files and
an explicit model card. It excludes run history and raw datasets. See the
[publishing tutorial](docs/PUBLISHING.md) for authentication, visibility, licensing and
multimodal pipeline tags.

## Advanced: control execution and deployment

**Goal:** make detailed experiment choices, test them locally, and scale or deploy deliberately. The full recipe exposes every training field; cloud and API lifecycle configurations stay separate so a temporary training job cannot accidentally become a permanent service.

### Data: audit the experiment inputs

Choose sampling and licenses deliberately. The built-in acquisition paths are bounded viewer rows (`hf_rows`), cached bulk Parquet shards (`hf_parquet`), and local JSONL. Bulk Parquet acquisition avoids repeated 100-row viewer requests. Additional corpora require compatible source settings or external extraction to local data. Keyword filtering is a simple domain proxy, not a semantic classifier. Near-deduplication compares word shingles and is intended for small samples, not distributed corpus processing.

Supply your own evaluation text through `processing.benchmark_files` and preserve run snapshots. Automatic decontamination covers only the selected benchmark sample, and regex PII filtering is incomplete. For larger experiments, assess these limitations before treating a saved manifest as evidence of data quality. A manifest identifies bytes; it does not certify the corpus.


### Published recipes and small-scale adaptations

Start with a published training pattern, then size the experiment for your hardware. These are reference recipes, not a universal best configuration or a reconstruction of unpublished proprietary frontier training. Each linked paper describes the original model; the YAML files below are deliberately small **ModelChef adaptations**, with project-chosen learning rates and budgets.

| Pattern | Published recipe / original source | What to carry into an experiment | ModelChef example and limits |
| --- | --- | --- | --- |
| Small, data-focused dense model | [SmolLM2 paper](https://arxiv.org/abs/2502.02737): 1.7B model, approximately 11T tokens, staged web/math/code data curation | Start with clean general text; evaluate, then specialize using curated domain data; finish with instruction examples | [SmolLM2-inspired](configs/recipe-smollm2-inspired.yaml): bounded FineWeb-Edu sample and small native decoder. Later `midtrain`/`sft` runs use separate recipes; this does not reproduce the original mixtures or token budget. |
| Open, staged foundation training | [OLMo 2 paper](https://arxiv.org/abs/2501.00656), [original Ai2 post](https://allenai.org/blog/olmo2), and [official training recipes](https://github.com/allenai/OLMo-core/tree/main/src/scripts/official/OLMo2) | Broad pretraining followed by curated late training; evaluate domain gains and general capability; use SFT/preferences/verifiable rewards when suitable | [OLMo-inspired](configs/recipe-olmo-inspired.yaml): local curated corpus, then separate stages. Native architecture and cosine schedule differ from OLMo 2; its linear annealing and training infrastructure are not reproduced. |
| Large dense frontier-scale reference | [The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783): released family includes a 405B model; report describes pretraining and post-training | Decoder with grouped-query attention, careful data quality, staged capability development and instruction/preference training | [Llama-inspired](configs/recipe-llama-inspired.yaml): small RoPE/RMSNorm/SwiGLU/GQA decoder. No distributed frontier-scale training, original tokenizer/data mixture or long-context curriculum reproduction. |
| Efficient pretrained specialization | [LoRA paper](https://arxiv.org/abs/2106.09685) | Freeze the base and train low-rank updates rather than all weights | Generate an imported-model SFT recipe and enable [LoRA](#parameter-efficient-fine-tuning-with-lora). This is an adaptation method, not a foundation pretraining recipe. |

For frontier architectures beyond dense decoders, the [DeepSeek-V3 technical report](https://arxiv.org/abs/2412.19437) documents MoE, latent attention and multi-token prediction; ModelChef implements a small Mixtral-like MoE, but does not reproduce DeepSeek-style latent attention, multi-token prediction or expert-parallel training. The [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) provides a reasoning/RL reference. ModelChef's optional reasoning supervision, distillation and small verifier-based RLVR experiments are learning workflows, not reproductions of R1's training algorithm or capability. These references help identify future extensions without presenting unsupported options as runnable recipes.

#### Frontier-model patterns: what can be reproduced

Use these as experiment designs, not claims to reproduce frontier capability. A **published pattern** comes from an original report; a **ModelChef adaptation** selects the supported pieces and supplies small-scale hyperparameters. The **inferred assistant pattern** below is a plausible starting point, not a disclosed recipe for GPT, Claude or any other proprietary model. Actual data mixtures, budgets and training algorithms remain unknown where they have not been published.

| Pattern and evidence | Recipe sequence to explore | ModelChef starting point | Difference from the original |
| --- | --- | --- | --- |
| Dense instruction model — [Llama 3 report](https://arxiv.org/abs/2407.21783) | Broad pretraining → instruction SFT → preference optimization → evaluate/export | [Small dense recipe](configs/recipe-llama-inspired.yaml), then `stage-config --kind sft` and `--kind dpo` | No original data mixture, rejection-sampling system, frontier scale or long-context curriculum |
| Sparse expert model — [DeepSeek-V3 report](https://arxiv.org/abs/2412.19437) | MoE pretraining → curated continuation → instruction/preference stages | Copy the dense recipe; set `model.type: decoder_moe`, `num_experts: 4`, `experts_per_token: 2`; derive later stages from its completed run | Uses ModelChef's Mixtral-like MoE; no DeepSeek latent attention, auxiliary-loss-free routing, multi-token prediction or expert parallelism |
| Verifiable reasoning — [DeepSeek-R1 paper](https://arxiv.org/abs/2501.12948) | Capable base → optional reasoning SFT → verifiable-reward experiment; optionally distill outputs to a student | `stage-config --kind sft --reasoning think_answer`, then `--kind rlvr`; see [distillation](#distillation) for a smaller student | Experimental local verifier and rollout objective, not an implementation of the complete R1/GRPO training recipe |
| General assistant — **inferred design** based on the published staged patterns above | Capable pretrained base → domain midtraining if needed → instruction SFT → DPO → general/domain regression checks | Import a compatible local model, derive separate `midtrain`, `sft` and `dpo` recipes | No claim about proprietary architectures, reward models, data or hyperparameters; stage order is an experiment choice |

For a first dense assistant experiment, train the Llama-inspired base or import a suitable existing model, then derive the next recipe:

```bash
python -m modelchef.cli stage-config BASE_RUN --kind sft --data data/raw/sft.jsonl --output configs/assistant-sft.local.yaml --steps 40
python -m modelchef.cli pipeline configs/assistant-sft.local.yaml
python -m modelchef.cli pipeline configs/assistant-sft.local.yaml --run
# Use the newly completed SFT run as the parent, not BASE_RUN.
python -m modelchef.cli stage-config SFT_RUN --kind dpo --data data/raw/preferences.jsonl --output configs/assistant-dpo.local.yaml --steps 40
python -m modelchef.cli pipeline configs/assistant-dpo.local.yaml
```

SFT needs `prompt,response` records; DPO needs `prompt,chosen,rejected`. For reasoning SFT, use the [optional reasoning format](#optional-reasoning-supervision); for RLVR, supply the verifier-compatible records described in [continued training](#refine-your-model-continue-training). These stages need separate data: raw documents are not preference pairs or verified answers. Forty steps test the wiring, not the recipe's effectiveness. Evaluate after each stage and keep the earlier checkpoint if a later stage regresses. Use `cook` once the individual stages work; its workflow recipe links their parents automatically.

Start with ordinary precision and a supported baseline. MoE does not make every run smaller or faster: all experts still occupy storage and routing adds overhead. FP4, custom kernels and DDP are execution choices, not required ingredients of a frontier-inspired recipe. The [compatible-combination table](#choose-a-compatible-combination) explains current restrictions.

**Try a recipe:** copy one to an ignored personal file, inspect its data source and limits, and validate before execution:

```bash
cp configs/recipe-smollm2-inspired.yaml configs/my-small-model.local.yaml
python -m modelchef.cli pipeline configs/my-small-model.local.yaml
python -m modelchef.cli pipeline configs/my-small-model.local.yaml --run
```

The SmolLM2 example downloads only a bounded sample when executed. The other two use `data/raw/corpus.jsonl`: provide your own JSONL records with a `text` field before execution. Validation does not launch cloud resources. Use [managed cloud training](#run-a-managed-training-stage) for an explicitly configured cloud run; these recipes default to local execution; select an AWS or Runpod target and its resource configuration to move them to a managed host.

For a staged experiment, train the base, then use `stage-config BASE_RUN --kind midtrain --data data/raw/domain.jsonl --output configs/domain.local.yaml --steps 40`. After that run, generate `--kind sft` from its output using instruction/response JSONL. Add DPO or RLVR only when you have appropriate preference pairs or verifiable tasks. Each stage has its own recipe, data manifest and checkpoint lineage. Export the chosen final run for CLI inference or vLLM serving.

Do not copy a frontier model's learning rate or batch size blindly. Measure throughput and memory first; compare held-out loss and domain benchmarks across runs. The current trainer samples packed blocks with replacement: `steps × micro_batch_size × gradient_accumulation × context_length` estimates token positions processed, not unique training tokens. Tiny tests establish pipeline correctness, not competitive model quality. For every available field and required/optional rules, see the [complete annotated recipe](#complete-annotated-recipe).

### Complete annotated recipe

The following is a complete, schema-valid training recipe with **every supported text-training field** explicit. Vision/speech have their [own complete schema and tutorial](docs/MULTIMODAL.md). A copy lives in [configs/recipe-reference.yaml](configs/recipe-reference.yaml). Comments show alternatives and when a setting applies; the selected values form a small native pretraining run. Fields for DPO, RLVR and distillation remain inactive until their stage is selected. There is no single run that applies all objectives at once.

#### Required versus optional

Every training field has a default, so **no individual field must be written for the default native pretraining workflow**. Omitting a section uses that section's defaults. A minimal custom recipe can be just `name: my-run`; it still uses the default public corpus and training settings. Explicit fields are recommended for reproducibility and to make download/runtime limits reviewable.

The YAML below labels fields `OPTIONAL` (may be omitted) or `CONDITIONAL` (must be supplied or changed when a selected workflow needs it). Optional does not mean inactive: a default may enable filtering, downloads or evaluation.

| Field | When you must supply or change it |
| --- | --- |
| `data.local_path` | Required when `data.provider: local_jsonl`; the file must exist when executing |
| `data.domain_keywords` | Must be nonempty for any non-general domain |
| `evaluation.benchmark` | Must override `auto` for domains other than general/finance; use a supported name or `none` |
| `stage.parent_run` | Required for midtrain, SFT, DPO, RLVR and distill; execution needs a usable parent run |
| `stage.teacher_run` | Additionally required for token distillation |
| `model.pretrained_path` | Required for the Hugging Face backend; normally filled by `import-model`, then retained by `stage-config` |

All selected values must still satisfy cross-field rules: for example, structured stages require local JSONL, heads must match model width/KV-head geometry, and warmup must be shorter than training. A path or plugin can be optional in the schema but must exist once selected. Fields such as `dpo_beta`, `group_size` and `distillation_alpha` remain optional even in their respective stages because they have defaults.

```bash
cp configs/recipe-reference.yaml configs/full-run.local.yaml
python -m modelchef.cli pipeline configs/full-run.local.yaml
# After reviewing data limits and hardware requirements:
python -m modelchef.cli pipeline configs/full-run.local.yaml --run
```

```yaml
schema_version: 1  # OPTIONAL: defaults are supplied when omitted.
name: full-reference  # OPTIONAL: Used as the run-directory prefix; each execution gets a unique suffix.
output_dir: artifacts/training  # OPTIONAL: defaults are supplied when omitted.
explain: detailed  # OPTIONAL: detailed | brief; event logs are retained in either mode.
data:  # OPTIONAL: defaults are supplied when omitted.
  provider: hf_rows  # OPTIONAL: hf_rows | hf_parquet | local_jsonl; structured stages require local_jsonl.
  dataset: HuggingFaceFW/fineweb-edu  # OPTIONAL: Viewer dataset ID; no full Parquet corpus download.
  subset: sample-10BT  # OPTIONAL: Dataset configuration name, not a random sampling strategy.
  split: train  # OPTIONAL: defaults are supplied when omitted.
  text_field: text  # OPTIONAL: defaults are supplied when omitted.
  prepared_run: null  # OPTIONAL: Reuse matching disk-pretraining snapshots for fresh weights; same filesystem.
  local_path: null  # CONDITIONAL: One JSONL path, or use local_paths below.
  local_paths: []  # OPTIONAL: Ordered JSONL shards; mutually exclusive with local_path.
  storage: memory  # OPTIONAL: memory | disk; disk supports local pretrain/midtrain.
  max_record_bytes: 4000000  # OPTIONAL: Maximum raw JSONL record size.
  offset: 0  # OPTIONAL: Ordered starting row; skipped input bytes still count toward the cap.
  max_documents: 256  # OPTIONAL: Raw records before filtering; at least 3 accepted documents are needed.
  page_size: 50  # OPTIONAL: Viewer page length: 1..100.
  max_download_bytes: 16000000  # OPTIONAL: Corpus response/input-byte cap; separate from benchmark downloads.
  timeout_seconds: 30  # OPTIONAL: defaults are supplied when omitted.
  max_tokens: 200000  # OPTIONAL: Total packing budget across train/validation/test; not a training-token target.
  domain: general  # OPTIONAL: general | finance for auto benchmarks; other names need an explicit benchmark.
  domain_keywords: []  # CONDITIONAL: Any matching keyword qualifies; required for a non-general domain.
processing:  # OPTIONAL: defaults are supplied when omitted.
  unicode_normalization: NFKC  # OPTIONAL: NFC | NFKC.
  strip_html: true  # OPTIONAL: defaults are supplied when omitted.
  min_chars: 200  # OPTIONAL: defaults are supplied when omitted.
  max_chars: 20000  # OPTIONAL: defaults are supplied when omitted.
  min_alnum_fraction: 0.5  # OPTIONAL: defaults are supplied when omitted.
  max_repeated_line_fraction: 0.4  # OPTIONAL: defaults are supplied when omitted.
  exact_dedup: true  # OPTIONAL: defaults are supplied when omitted.
  near_dedup: true  # OPTIONAL: defaults are supplied when omitted.
  near_dedup_jaccard: 0.9  # OPTIONAL: Word-shingle similarity threshold, not embedding similarity.
  dedup_ngram_words: 5  # OPTIONAL: defaults are supplied when omitted.
  pii: redact  # OPTIONAL: redact | drop | off; regex heuristics for emails/phone-like strings.
  benchmark_files: []  # OPTIONAL: Local text files whose n-grams must be excluded; sample benchmark added automatically.
  decontamination_ngram_words: 13  # OPTIONAL: Exact overlapping word n-grams; not semantic decontamination.
  validation_fraction: 0.1  # OPTIONAL: Document fraction; validation + test must be below 1.
  test_fraction: 0.1  # OPTIONAL: defaults are supplied when omitted.
tokenizer:  # OPTIONAL: defaults are supplied when omitted.
  packing: concatenate  # OPTIONAL: concatenate | document; document requires disk pretrain/midtrain and skips oversized examples.
  algorithm: byte_bpe  # OPTIONAL: byte_bpe | pretrained; pretrained is reused from an imported/parent run.
  vocab_size: 4096  # OPTIONAL: Requested BPE size (260..262144); a small corpus can learn fewer tokens.
  training_max_documents: null # Optional BPE fitting sample; null uses every training document.
  min_frequency: 2  # OPTIONAL: defaults are supplied when omitted.
  special_tokens:  # OPTIONAL: Required identities: pad, unk, bos, eos; all entries must be unique.
  - <pad>
  - <unk>
  - <bos>
  - <eos>
  add_bos: false  # OPTIONAL: Raw-text packing only; structured stages apply their own prompt/EOS formatting.
  append_eos: true  # OPTIONAL: Raw-text document separator; structured stages always append response EOS.
model:  # OPTIONAL: defaults are supplied when omitted.
  peft:  # OPTIONAL: Parameter-efficient training; imported HF midtrain/SFT only.
    method: none  # OPTIONAL: none | lora | qlora; adapters require a continuation parent.
    rank: 8  # OPTIONAL: Positive adapter rank.
    alpha: 16  # OPTIONAL: Positive adapter scaling factor.
    dropout: 0.0  # OPTIONAL: Adapter dropout in [0, 1).
    target_modules: [q_proj, v_proj]  # OPTIONAL: Must match the imported model; GPT-2 uses c_attn.
    use_rslora: false  # OPTIONAL: Rank-stabilized scaling.
    compute_dtype: bf16  # OPTIONAL: QLoRA compute: bf16 | fp16.
    double_quant: true  # OPTIONAL: QLoRA nested quantization.
  backend: native  # OPTIONAL: native | huggingface; use import-model to configure a local pretrained adapter.
  pretrained_path: null  # CONDITIONAL: Required local source directory for the huggingface backend.
  pretrained_config_sha256: null  # OPTIONAL: Filled by import-model; validates the original HF config on loading.
  quantization: none  # OPTIONAL: none | fp4_fake | nvfp4; nvfp4 requires native dense Blackwell integration.
  attention_kernel: auto  # OPTIONAL: auto | math | flash | efficient | cudnn; explicit optimized choices require a compatible GPU.
  custom_swiglu_path: null  # OPTIONAL: Native CUDA only; managed AWS bundles declared sources and installs the matching compiler.
  type: decoder_rope  # OPTIONAL: decoder_rope | decoder_moe; applies to native architecture.
  num_experts: 4  # OPTIONAL: MoE expert count.
  experts_per_token: 2  # OPTIONAL: MoE top-k; <= num_experts.
  router_aux_loss_coef: 0.01  # OPTIONAL: MoE load-balancing loss weight.
  layers: 6  # OPTIONAL: defaults are supplied when omitted.
  hidden_size: 512  # OPTIONAL: Must divide evenly by attention_heads; head dimension must be even.
  attention_heads: 8  # OPTIONAL: Must divide evenly by kv_heads.
  kv_heads: 2  # OPTIONAL: defaults are supplied when omitted.
  intermediate_size: 1360  # OPTIONAL: defaults are supplied when omitted.
  context_length: 128  # OPTIONAL: defaults are supplied when omitted.
  norm: rmsnorm  # OPTIONAL: Native architecture supports rmsnorm only.
  norm_eps: 1.0e-05  # OPTIONAL: defaults are supplied when omitted.
  activation: swiglu  # OPTIONAL: Native architecture supports swiglu only.
  rope_theta: 10000.0  # OPTIONAL: defaults are supplied when omitted.
  attention_dropout: 0.0  # OPTIONAL: defaults are supplied when omitted.
  residual_dropout: 0.0  # OPTIONAL: defaults are supplied when omitted.
  bias: false  # OPTIONAL: defaults are supplied when omitted.
  tie_embeddings: true  # OPTIONAL: Share input embedding and output-head weights.
  init_std: 0.02  # OPTIONAL: defaults are supplied when omitted.
  gradient_checkpointing: false  # OPTIONAL: Recompute activations during backward to trade compute for memory.
training:  # OPTIONAL: defaults are supplied when omitted.
  performance_gate: null  # OPTIONAL: measured throughput/memory gate; choose acceptance thresholds from your own measurements.
  distributed: none  # OPTIONAL: none | ddp; DDP launches with torchrun.
  distributed_timeout_seconds: 300  # OPTIONAL: process-group timeout.
  compile: false  # OPTIONAL: Opt in to torch.compile; default leaves execution eager.
  compile_backend: inductor  # OPTIONAL: inductor | aot_eager | eager.
  compile_mode: default  # OPTIONAL: default | reduce-overhead | max-autotune; used only with inductor.
  allow_tf32: false  # OPTIONAL: Permit TF32 for eligible CUDA FP32 matrix multiplies.
  optimizer_kernel: default  # OPTIONAL: default | fused | foreach; fused requires CUDA/ROCm.
  accelerator: auto  # OPTIONAL: auto | cpu | cuda | rocm | mps.
  precision: auto  # OPTIONAL: auto | fp32 | bf16 | fp16; FP16 uses gradient scaling.
  seed: 42  # OPTIONAL: defaults are supplied when omitted.
  deterministic: true  # OPTIONAL: Request deterministic PyTorch operations; not cross-platform bitwise parity.
  steps: 40  # OPTIONAL: Optimizer updates, with training blocks sampled with replacement.
  micro_batch_size: 2  # OPTIONAL: defaults are supplied when omitted.
  gradient_accumulation: 2  # OPTIONAL: Effective single-device batch = micro_batch_size * gradient_accumulation.
  optimizer: adamw  # OPTIONAL: adamw | adamw8bit | adamw_in_backward | adamw8bit_in_backward; optional policies require optimizer_kernel: default; see README restrictions.
  learning_rate: 0.0005  # OPTIONAL: defaults are supplied when omitted.
  min_learning_rate_ratio: 0.1  # OPTIONAL: Final cosine learning-rate floor relative to learning_rate.
  betas:  # OPTIONAL: defaults are supplied when omitted.
  - 0.9
  - 0.95
  weight_decay: 0.1  # OPTIONAL: defaults are supplied when omitted.
  optimizer_eps: 1.0e-08  # OPTIONAL: defaults are supplied when omitted.
  schedule: cosine  # OPTIONAL: cosine | constant; warmup still applies.
  warmup_steps: 4  # OPTIONAL: Must be less than steps.
  gradient_clip_norm: 1.0  # OPTIONAL: 0 disables clipping; required for in-backward optimizers.
  checkpoint_at_start: false  # OPTIONAL: Save initial weights, optimizer and RNG before training.
  checkpoint_every: 20  # OPTIONAL: Optimizer-step interval; final checkpoint is also saved.
  evaluate_every: 10  # OPTIONAL: defaults are supplied when omitted.
  log_every: 5  # OPTIONAL: defaults are supplied when omitted.
  max_runtime_seconds: 600  # OPTIONAL: Checked between optimization steps; not an acquisition/evaluation deadline.
evaluation:  # OPTIONAL: defaults are supplied when omitted.
  benchmark: auto  # CONDITIONAL: auto | none | hellaswag | financial_phrasebank.
  benchmark_examples: 8  # OPTIONAL: 1..100; bounded custom protocol, not a full standard suite.
  benchmark_max_download_bytes: 2000000  # OPTIONAL: Separate benchmark payload-byte cap.
  batches: 4  # OPTIONAL: Positive batch limit, or all for every packed example in each split.
  prompt: Education helps people  # OPTIONAL: defaults are supplied when omitted.
  generate_tokens: 24  # OPTIONAL: defaults are supplied when omitted.
  temperature: 0.8  # OPTIONAL: defaults are supplied when omitted.
  top_k: 30  # OPTIONAL: 0 disables top-k filtering.
stage:  # OPTIONAL: defaults are supplied when omitted.
  oversized_policy: error # OPTIONAL: error | skip; skip counts complete examples that cannot fit context.
  kind: pretrain  # OPTIONAL: pretrain | midtrain | sft | dpo | rlvr | distill; one objective per run.
  parent_run: null  # CONDITIONAL: Required completed/imported/initialized parent directory for later stages.
  teacher_run: null  # CONDITIONAL: Required for token distillation; retain teacher artifacts for resume.
  distillation_temperature: 2.0  # OPTIONAL: Distill only: soften token distributions; soft loss scaled by T squared.
  distillation_alpha: 0.5  # OPTIONAL: Distill only: soft teacher weight; remaining weight is response cross entropy.
  dpo_beta: 0.1  # OPTIONAL: DPO only: reference-adjusted preference-margin scale.
  reward: integer  # OPTIONAL: RLVR only: exact_match | integer.
  group_size: 4  # OPTIONAL: RLVR only: 2..32 sampled answers per prompt.
  rollout_tokens: 16  # OPTIONAL: RLVR only: maximum sampled response length, also bounded by context.
  kl_coefficient: 0.02  # OPTIONAL: RLVR only: frozen-reference regularization weight.
  clip_ratio: 0.2  # OPTIONAL: RLVR only: policy-ratio clipping range.
  reasoning_format: plain  # OPTIONAL: plain | think_answer; optional SFT/distill fields, or RLVR final-answer extraction.
  prompt_format: user_assistant  # OPTIONAL: user_assistant | chat_transcript; transcripts preserve system/persona roles.
```

#### Choose a compatible combination

| Experiment | Changes to make | Prerequisites / constraints |
| --- | --- | --- |
| Offline corpus run | `data.provider: local_jsonl`, set `local_path`, `evaluation.benchmark: none` | Installed packages and enough distinct local documents for all splits |
| Finance sample | `data.domain: finance`, supply `domain_keywords` | Auto selects FinancialPhraseBank; filters may leave too little data |
| Continue training / SFT | Generate with `stage-config`; set `kind` and `parent_run` | Keep parent architecture/tokenizer; SFT uses prompt/response JSONL |
| DPO | `kind: dpo`, `dpo_beta`, parent run | Local prompt/chosen/rejected JSONL; frozen reference also consumes memory |
| RLVR | `kind: rlvr`, verifier, group size, rollout and KL controls | Local prompt/answer JSONL; enough correct/varied samples for useful rewards |
| Token distillation | `kind: distill`, parent/student and teacher runs, temperature/alpha | Identical tokenizers and special IDs; teacher context covers student; both models fit memory |
| Reasoning supervision | `reasoning_format: think_answer` for SFT/distill | Local prompt/reasoning/answer JSONL; enough context for both sections |
| Pretrained adaptation | `import-model`, then `stage-config` | Import fills HF paths, original tokenizer and geometry; native geometry fields do not redesign the imported architecture |
| Execution optimization | Compile, attention backend, optimizer kernel, optional TF32 | Profile separately; explicit GPU kernels may fail for unsupported shapes/dtypes or determinism settings |
| Custom CUDA | `custom_swiglu_path: kernels/swiglu` | Native CUDA host with compatible toolkit; correctness probe required; speedup not established |
| FP4 simulation | `quantization: fp4_fake` | Compatible linear input dimensions divisible by 16; no packed storage/native acceleration |

Keep performance changes separate at first: custom CUDA, compilation, fake FP4, dropout and forced attention backends are not all jointly validated. Schema validation checks structure and invariants, not model memory fit, data availability or kernel compatibility. Resume uses the original recipe; changing an objective, model or step target creates a new experiment.

#### Serving and cloud resources are separate recipes

`pipeline` describes training on the current host. API deployment and cloud lifecycle settings belong in their own files rather than additional training YAML fields:

| Configuration | Full settings / workflow |
| --- | --- |
| [inference.yaml](configs/inference.yaml) | Local vLLM model path, dtype, quantization, GPU memory, context, sequence limits and API resource limits |
| [inference-public.yaml](configs/inference-public.yaml) | Remote bind/port, token environment variable, TLS files and request limits; [serving instructions](#serve-your-model-inference-and-api) |
| [runpod.yaml](configs/runpod.yaml) | Temporary Pod, existing network volume, SSH paths, GPU/image, transfer limits and runtime/budget ceilings |
| [aws-test.yaml](configs/aws-test.yaml) | Temporary Spot instance, per-run/overall reservations, runtime/rate ceilings, storage and cleanup |
| [aws-training.yaml](configs/aws-training.yaml) | Spot lifecycle plus training recipe, S3/profile, source/result caps and artifact reserve |
| [aws-serving.yaml](configs/aws-serving.yaml) | Persistent on-demand host, existing network/image/profile, secret, export config and cost-warning threshold |

For these separate formats, requirements differ from the training schema:

| Format | Required user inputs | Optional/defaulted controls |
| --- | --- | --- |
| vLLM serving | `inference.model_path` referencing a compatible export; token in the configured environment variable at launch; both TLS file paths for non-loopback binding | Host, port, dtype, quantization, memory/context/sequence limits and API limits |
| Persistent AWS serving | Profile, region, instance type, AMI, subnet, security groups, instance profile, bucket, secret ID, inference config, hourly estimate and warning threshold | Root device inferred from AMI if omitted, storage size, bundle byte cap |
| Runpod training | Existing network volume ID and SSH key paths; review the complete resource template before launch | GPU/count, image/CUDA floor, runtime, budgets, transfer caps and optional result download |
| Temporary Spot testing/training | Use the complete AWS template: identity, instance/budget/runtime/rate/storage fields and AMI-resolution settings; training additionally needs its recipe, artifact bucket and instance profile | Explicit AMI/subnet overrides, vLLM installation toggle; training transfer caps/artifact reserve have defaults |

The temporary AWS runner reads several template keys directly; unlike `RunConfig`, it does not supply defaults for an arbitrary minimal mapping. Keep the complete template and edit it. Credentials and API tokens are required at execution, not literal YAML values.

These files compose: for example, `aws_test.pipeline_config` in the AWS training template points to a training recipe, while `aws_serving.inference_config` points to a serving recipe. Temporary training ends with termination; persistent hosting stays up until an explicit lifecycle action. Tokens, certificates, private paths and account-specific settings belong outside Git or in ignored local configs.

### Tune the execution: performance and precision

For large experiments on limited NVIDIA GPU memory, the optional `optimizer-8bit`
extra provides `training.optimizer: adamw8bit`. The experimental
`adamw8bit_in_backward` also frees each gradient after its update. It requires
`gradient_accumulation: 1`, `gradient_clip_norm: 0`, `optimizer_kernel: default`,
BF16/FP32, a single process, no compilation and a supervised training stage.
`adamw_in_backward` uses the same restrictions with ordinary FP32 AdamW state;
8-bit variants require an NVIDIA CUDA GPU and the optional extra.
All optional optimizer policies require `optimizer_kernel: default`.
These choices change optimizer behavior; fitting a larger model does not establish
its quality. Ordinary `adamw` remains the default.

Establish a correctness baseline before changing execution settings. Architecture changes affect what model you train; kernel and compiler choices affect how its operations execute. Compare tokens/second, peak memory and validation behavior under the same recipe, and include compilation/warmup overhead when evaluating short runs.

`attention_kernel: auto` lets PyTorch select an available scaled-dot-product attention implementation. Explicit choices constrain the backend and can fail for unsupported shapes or devices. `torch.compile` captures and optimizes execution; it is not a new model architecture. Fused optimizers combine operations to reduce launch overhead; `foreach` batches operations across tensors. TF32 permits reduced mantissa precision for eligible FP32 matrix operations. None guarantees a speedup on every workload.

#### Custom CUDA and execution settings

Custom CUDA is opt-in for the native decoder. The included example replaces the SwiGLU activation with a CUDA extension implementing both forward and backward operations. PyTorch remains the default. This is a correctness-oriented example; no speedup has been established.

1. Use a Linux CUDA host or WSL2 with a working NVIDIA driver, C++ compiler and a CUDA toolkit compatible with your installed PyTorch. The toolkit's `nvcc` and runtime headers must also match. CPU and Apple MPS cannot execute this extension.
2. Install the small build dependency in your active environment: `python -m pip install -e '.[cuda-kernels]'`. If the toolkit is outside standard locations, set `CUDA_HOME` to its root. This command does not install a CUDA toolkit.

   For the project's Linux x86-64 PyTorch CUDA **13.0** environment, compatible compiler components are available through `python -m pip install -e '.[cuda-kernels,cuda13-toolchain]'`. This pins `nvidia-cuda-nvcc`, `nvidia-cuda-crt` and `nvidia-nvvm` to 13.0.88; it does not replace PyTorch's runtime. Set the toolkit root for these wheels with `export CUDA_HOME="$(python -c 'import site; print(site.getsitepackages()[0])')/nvidia/cu13"`. Use a matching toolkit instead if your PyTorch build uses another CUDA version.
3. Run the forward/gradient check before training:

   ```bash
   python -m modelchef.cli check-cuda-kernel kernels/swiglu
   ```

4. Start with `configs/pretrain-custom-cuda.yaml`, or copy your training recipe and edit the following fields within its existing sections:

   ```yaml
   model:
     custom_swiglu_path: kernels/swiglu
     attention_kernel: auto
   training:
     accelerator: cuda
     precision: bf16
     compile: false
     optimizer_kernel: default
     allow_tf32: false
   ```

5. Run `python -m modelchef.cli pipeline configs/YOUR_RECIPE.yaml --run`. Startup validates FP32, FP16 and supported BF16 outputs and gradients against PyTorch before acquiring data. Source hashes are recorded and checked when resuming. Build products stay under ignored `artifacts/cuda-extensions/`.

`src/modelchef/execution.py` loads the trusted local plugin and runs checks; `model.py` selects it; `kernels/swiglu/kernel.py` provides the autograd wrapper; `swiglu.cpp` exposes bindings; and `swiglu.cu` implements the CUDA operations. Custom SwiGLU currently applies to the native dense decoder, not MoE or the Hugging Face backend. Your own directory must export `swiglu(gate, up)` from `kernel.py` with autograd support. Selecting a plugin executes its local code, so use a source directory you trust. Set `custom_swiglu_path: null` to restore the built-in activation.

Other optional execution controls are `model.attention_kernel: auto|math|flash|efficient|cudnn`, `training.compile`, `compile_backend: inductor|aot_eager|eager`, `compile_mode: default|reduce-overhead|max-autotune`, and `optimizer_kernel: default|fused|foreach`. Explicit optimized attention and fused optimizer selections require a compatible GPU. Measure throughput and memory for your workload; compilation can add startup time. Completed run reports include `performance.training_loop_seconds`, `tokens_per_second` and peak CUDA allocated/reserved bytes. These timings include loop evaluation/checkpoint work and compilation performed inside the loop, while excluding setup and initial validation; DDP reports global tokens and maximum per-rank CUDA peaks. Custom CUDA currently targets native-model SwiGLU. Managed `aws-train` packages the declared kernel sources and installs a CUDA 13 compiler matching the pinned PyTorch build. Native FP4 kernels are a separate capability.

The included kernel has passed FP32, FP16 and BF16 forward/backward checks on an RTX 4060 Laptop GPU with CUDA 13.0 compiler components. NVIDIA runtime wheels that lack an unversioned `libcudart.so` receive a linker alias inside the ignored build directory. A mismatched toolkit still fails visibly rather than bypassing compatibility checks. These checks establish correctness for the tested inputs, not a performance improvement.

#### FP4 experiments

For portable software experiments:

```bash
python -m modelchef.cli pipeline configs/pretrain-fp4-fake.yaml
python -m modelchef.cli pipeline configs/pretrain-fp4-fake.yaml --run
```

`model.quantization: fp4_fake` applies block-scaled E2M1-style fake quantization to compatible linear weights and activations, using straight-through gradients. It requires linear input dimensions divisible by 16. It can run on the current laptop, but **does not pack checkpoints, accelerate GEMMs or guarantee bitwise parity with NVIDIA's NVFP4 recipes**. Master weights, optimizer state and accumulation remain at the configured higher precision. Chat through PyTorch reproduces fake quantization; HF export writes full-precision master weights and records that distinction.

For native online NVFP4 inference, set `inference.quantization: nvfp4_per_token` in a serving recipe and use `serve-vllm`. The project checks for Blackwell CUDA compute capability before selecting it; vLLM performs final kernel/model checks. The RTX 4060 and macOS do not have this native path. Native training is available through the optional Transformer Engine integration described below. See [vLLM online NVFP4](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/quantization/online/nvfp4/) and [TorchAO workflows](https://docs.pytorch.org/ao/stable/workflows/index.html).

#### Native NVFP4 training on Blackwell

Install `uv pip install -e '.[nvfp4]'` with a compatible Linux Blackwell CUDA and Transformer Engine stack. Start from [configs/pretrain-nvfp4.yaml](configs/pretrain-nvfp4.yaml), or set `model.quantization: nvfp4` in a **native dense** recipe. Linear projection dimensions must be multiples of 16; Transformer Engine supplies NVFP4 block-scaled arithmetic while master parameters remain FP32. See [Transformer Engine FP8/FP4 training](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/examples/fp8_primer.html).

This integration requires actual Blackwell hardware; CPU simulation checks dispatch and configuration rather than numerical hardware behavior. Compatible users can run it even though real-device Blackwell validation is pending. It currently excludes MoE, PEFT, gradient checkpointing, custom SwiGLU and DDP. Use fake FP4 for software experiments on other hardware, and report it as simulation.

### Mixture of experts and distributed training

A mixture-of-experts (MoE) decoder routes each token to a subset of feed-forward experts. This increases total parameter capacity without activating every expert for every token. Start from [configs/pretrain-moe.yaml](configs/pretrain-moe.yaml), or set `model.type: decoder_moe`, then configure `num_experts`, `experts_per_token` and `router_aux_loss_coef`. The last setting weights an auxiliary load-balancing loss. Keep `bias: false` for Mixtral-compatible HF export. This is a small top-k routed architecture, not a reproduction of DeepSeek's full system; see [Mixtral](https://arxiv.org/abs/2401.04088).

DDP gives each process a complete model replica and synchronizes gradients. It improves data throughput; it does not shard weights or make an oversized model fit on one device. Copy [configs/pretrain-ddp.yaml](configs/pretrain-ddp.yaml) to `configs/ddp.local.yaml`, or edit a stage recipe to set `training.distributed: ddp`, choose CPU or compatible GPUs, then launch:

```bash
torchrun --standalone --nproc_per_node=2 -m -- modelchef.cli pipeline configs/ddp.local.yaml --run
```

Each rank samples training blocks; the global sequence batch is `micro_batch_size × gradient_accumulation × world_size`. Rank zero prepares shared artifacts and writes reports/checkpoints; recovery preserves per-rank random state and requires the same world size. Dense and MoE two-rank CPU execution and resume have been tested. GPU DDP uses NCCL and needs its own throughput and device validation; MPS DDP is unavailable. DDP currently excludes QLoRA, NVFP4, compilation and RLVR. See [PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html) and [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html).


### Evaluation: run standard suites

For the optional [EleutherAI evaluation harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
uv pip install -e '.[evaluation]'
python -m modelchef.cli export PATH_TO_COMPLETED_RUN
python -m modelchef.cli evaluate-standard PATH_TO_EXPORT --tasks hellaswag piqa arc_easy --limit 100
# Explicit execution can download benchmark datasets.
python -m modelchef.cli evaluate-standard PATH_TO_EXPORT --tasks hellaswag piqa arc_easy --limit 100 --run
```

The default command only prints a plan. `--limit` caps scored examples, **not dataset download bytes**; obtain approval for large datasets before executing. The harness is an optional external integration: full suite execution and real GPT-2/Pythia/SmolLM2 comparisons have not been run in the bounded local validation. Finance selection in the built-in pipeline uses FinancialPhraseBank; choose additional harness tasks explicitly when needed.

### Cloud execution

The training recipe defines **what** to train. Select **where** with `--target`;
local is the default. Keep provider resources and budgets in a separate file, so
changing providers does not require rewriting the model or data recipe.

| Target | Resource configuration | Persistent recovery | Shutdown |
| --- | --- | --- | --- |
| Local | No cloud configuration | Run directory on your disk | Training process ends; no cloud compute |
| AWS | `--target aws --aws-config CONFIG` | Checkpoints and raw/processed data committed to S3 | Temporary Spot instance terminates after training or deadline; AWS deadline provides an additional safeguard |
| Runpod | `--target runpod --runpod-config CONFIG` | Checkpoints and raw/processed data on an existing network volume | Controller attempts Pod deletion after training or timeout; keep the controller connected |

Both cloud targets package the declared inputs and ModelChef source, preserve
recipe-selected precision/optimizer/kernel settings, and request spending approval
before provisioning. Storage retained after compute ends continues to incur charges.
Their ledgers reserve allowances conservatively; they are separate from each other
and are not exact provider bills. See the [execution tutorial](docs/EXECUTION.md).

#### Run a managed training stage

Use the same stage or workflow on either provider. These commands **preview**;
append `--run` only after configuring resources and reviewing the plan.

```bash
python -m modelchef.cli pipeline configs/pretrain-demo.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli pipeline configs/pretrain-demo.yaml --target runpod --runpod-config configs/runpod.local.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --target aws --aws-config configs/aws-training.local.yaml
python -m modelchef.cli cook configs/workflow-demo.yaml --target runpod --runpod-config configs/runpod.local.yaml
```

The workflow demo needs its declared instruction data before packaging. For a
paid pilot, start with a bounded recipe and inspect both training success and cleanup
confirmation. Cloud `--resume` is currently unsupported: retrieve the persisted
run directory and resume a compatible checkpoint through the local interface.

#### AWS execution

Copy [aws-training.yaml](configs/aws-training.yaml) to an ignored
`configs/aws-training.local.yaml`. Configure your CLI profile, region, instance type,
AMI/network, instance profile, S3 bucket, transfer limits, runtime and spending limits.
The controller performs local preflight and launches one temporary Spot host for a
workflow. Optional [EC2 Fleet selection](docs/AWS_FLEET.md) chooses among configured
capacity pools. Guest dependencies follow the training recipe.

#### Prepare AWS access

Follow the [AWS credentials, IAM and infrastructure tutorial](docs/AWS.md#prepare-aws-access)
for CLI setup and permission boundaries. The full [AWS operations guide](docs/AWS.md)
covers readiness tests, budget reservations, launch monitoring and recovery. Use the
same OS environment for credentials and commands; Windows and WSL profiles are separate.

After a run, inspect the recorded instance state and downloaded results. Use
`aws-reconcile RUN_ID --path CONFIG` for result/cleanup recovery; S3 recovery snapshots
remain independent of the terminated training instance.

#### Runpod execution

Copy [runpod.yaml](configs/runpod.yaml) to an ignored `configs/runpod.local.yaml`.
Reuse your Runpod CLI credentials or set `RUNPOD_API_KEY` privately in the controller
environment. Configure an SSH key pair, an **existing network volume**, GPU type/count,
compatible image/CUDA floor, transfer limits, runtime and spending allowances. Keep
credentials out of YAML and Git. Follow the [Runpod setup tutorial](docs/RUNPOD.md)
for private key entry and storage setup.

Runpod previews validate the bundle offline. At execution, the runner checks the
catalog quote and creates one temporary Pod in the network volume's data center.
The shared guest setup installs the recipe's training dependencies; a suitable CUDA
host is still required for CUDA kernels and the pinned PyTorch build.

```bash
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action status
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action logs
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action fetch
python -m modelchef.cli runpod-job artifacts/runpod-runs/RUN_ID --action terminate --run
```

Replace `RUN_ID` with the saved job directory. Logs and fetch require a live Pod;
after deletion, attach the retained network volume to another Pod to retrieve the
job directory. Set `download_results: true` for automatic bounded result download.
**Keep the controller running:** Runpod runtime/budget safeguards depend on it;
an unreachable API or killed controller can leave billable compute. Network volumes
survive Pod deletion and continue billing until explicitly deleted.

#### Remote data preparation and resumable checkpoints

Both targets persist raw corpus snapshots, processed splits, packed tokens, tokenizer,
configuration and model/optimizer checkpoints. Recover the last successfully persisted
snapshot; in-memory updates cannot be recovered. See [storage lifecycle](docs/CLOUD_STORAGE.md),
[AWS checkpoint recovery](docs/AWS.md#remote-data-preparation-and-resumable-checkpoints)
and [Runpod recovery](docs/RUNPOD.md#4-runtime-results-and-cleanup).

#### Persistent API hosting on EC2

Managed persistent hosting currently uses `aws-serve`, separately from temporary
training. Export the model and follow the [EC2 serving tutorial](docs/AWS.md#persistent-api-hosting-on-ec2).
This on-demand host remains running until explicitly stopped/deleted. Runpod's managed
target currently supports training, not persistent API deployment. On an existing
compatible Linux/WSL CUDA host from either provider, `serve-vllm` can serve an export;
you manage that host's networking, security and lifetime yourself.

## Code architecture

### Low-level component map

```mermaid
flowchart TD
    CLI[cli.py: parse commands] --> Config[run_config.py: strict recipe validation]
    CLI --> Cook[workflow.py: ordered cook stages and final export]
    Cook --> Pipeline
    Config --> DDP[distributed.py: torchrun replicas and recovery]
    DDP --> Pipeline
    CLI --> HW[hardware.py: detect and resolve device / precision]
    CLI --> PT[pretrained.py: download, import and stage recipes]
    CLI --> MM[multimodal.py: vision SFT and Whisper training / inference]
    MM --> Media[multimodal_data.py: bounded assets and grouped splits]
    CLI --> Pub[publishing.py: model card and explicit Hub commit]
    CLI --> Tools[data_tools.py: format conversion and teacher responses]
    CLI --> Compare[comparison.py: saved samples and standard-suite plans]
    Config --> Pipeline[pipeline.py: narrate, orchestrate, checkpoint and resume]
    HW --> Pipeline
    Pipeline --> Corpus[corpus.py: capped acquisition and filtering]
    Corpus --> Disk[disk_data.py: JSONL streaming, SQLite/LSH and mapped tokens]
    Pipeline --> Bench[benchmarks.py: domain selection and continuation scoring]
    Pipeline --> Pack[Tokenizer training and next-token packing]
    Corpus --> Objective[objectives.py: structured data and training losses]
    Pack --> Engine[engines.py: native / local HF adapters]
    Objective --> Engine
    Pipeline --> Execution[execution.py: compiler and kernel policy]
    Execution --> CUDA[kernels/swiglu: custom forward / backward]
    Engine --> Native[model.py: RoPE, GQA, RMSNorm, SwiGLU decoder]
    Engine --> HF[Transformers: local causal language model]
    Engine --> Quant[quantization.py: experimental FP4 fake quantization]
    Pipeline --> Saved[Ignored artifacts: config, snapshots, metrics, checkpoint]
    Saved --> Chat[chat.py: load, generate and export]
    Chat --> VLLM[inference.py: optional vLLM setup / serving]
    VLLM --> Security[api_security.py: authentication and request limits]
    CLI --> Spot[aws_runner.py: preflight, ledger, Spot and cleanup]
    CLI --> PodRunner[runpod_runner.py: budget, SSH, Pod lifecycle]
    PodRunner --> Bundle
    PodRunner --> PodGuest[runpod_guest.py: shared CUDA setup and job state]
    PodGuest --> Guest
    PodRunner --> Volume[Network volume: persisted data and checkpoints]
    CLI --> Hosting[aws_serving.py: persistent on-demand host and lifecycle]
    Hosting --> Service[systemd: restart vLLM after failure / reboot]
    Hosting --> Secret[Secrets Manager: API token and TLS material]
    Spot --> Bundle[aws_training.py: capped bundle and guest training script]
    Bundle --> Guest[aws_guest.py: recipe dependencies and CUDA/DDP launch]
    Guest --> Pipeline
    Bundle --> S3[S3: encrypted source and result transfer]
    Tests[tests/: configuration, objectives, causality and recovery] --> Pipeline
    Tests --> Spot
```

| Code | Responsibility |
| --- | --- |
| `cli.py`, `run_config.py` | Command dispatch and strict, versioned configuration |
| `diagnostics.py`, `guest_cli.py` | Shared environment checks and minimal EC2 readiness command entry point |
| `hardware.py`, `engines.py` | Backend selection and native/local pretrained model adapters |
| `corpus.py` | Bounded source snapshots, quality/PII/dedup/decontamination heuristics and document splits |
| `disk_data.py` | Bounded JSONL shards, SQLite/LSH indexes and memory-mapped token batches |
| `multimodal_config.py`, `multimodal_data.py`, `multimodal.py` | Strict media recipes, bounded assets, vision/Whisper training, recovery, evaluation and inference |
| `publishing.py` | Offline export/card preview and explicit authenticated Hugging Face publication |
| `workflow.py` | Ordered `cook` stages, parent/teacher links, reports and final export |
| `distributed.py` | CPU/GPU DDP synchronization and per-rank checkpoint recovery |
| `adapters.py`, `low_precision.py` | Native LoRA and optional CUDA QLoRA/NVFP4 integration |
| `pipeline.py` | Stage explanations, BPE/packing, optimizer loop, checkpoints, recovery and run reports |
| `model.py` | Custom decoder implementation used for from-scratch experiments |
| `objectives.py` | Response masking, DPO reference likelihoods, verifiable-reward rollouts and teacher KL |
| `pretrained.py` | Explicit capped model download, local import and parent-derived stage recipes |
| `data_tools.py` | Bounded text/CSV/JSONL conversion and local teacher-response generation |
| `comparison.py` | Local baseline comparisons and optional standard-suite execution plans |
| `quantization.py` | Software FP4 simulation with straight-through gradients; no packed-weight acceleration |
| `execution.py`, `kernels/swiglu/` | Attention/compiler policy, custom CUDA source verification and differentiable SwiGLU extension |
| `api_security.py` | vLLM route authentication, text-only request checks and process-wide resource limits |
| `benchmarks.py` | Domain benchmark acquisition and custom sampled evaluation protocol |
| `chat.py`, `inference.py` | Checkpoint inference, exports and optional vLLM serving |
| `runpod_runner.py`, `runpod_guest.py` | Runpod API/SSH lifecycle, spending allowances, shared guest setup and network-volume recovery |
| `aws_runner.py` | Local preflight, conservative budget reservations, Spot lifecycle and shutdown safeguards |
| `aws_training.py`, `aws_workflow.py` | Allowlisted stage/workflow bundles, local preflight, source verification and bounded result upload |
| `aws_guest.py` | Recipe-selected guest dependencies, CUDA toolchain and single-host training/DDP launch |
| `aws_serving.py` | Persistent on-demand deployment, bounded model bundle, systemd bootstrap and explicit stop/start/delete |
| `tests/` | Small unit/integration tests; hardware availability determines device coverage |

## Verification and project boundaries

See [quality and acceptance criteria](docs/QUALITY.md) for evidence required across usability, reliability, scale, model quality, performance and operations. Hardware simulations are identified separately from real-device validation.

Run the checks used for changes to the platform:

```bash
python -m ruff check src tests scripts/check_docs.py
python scripts/check_docs.py
python -m pytest -q
```

GitHub CI checks CPU workflows on Python 3.11, 3.12 and 3.13 and the Windows controller. The documentation check verifies local links, CLI examples, example recipes, complete annotated fields and generated schemas. Hardware-specific tests skip when their prerequisites are unavailable; a skipped test does not establish support on a real device. See [contributing](CONTRIBUTING.md) and [security boundaries](SECURITY.md).

Runs record package versions, the source-code fingerprint and available Git revision/dirty state in `environment.json`. Resume preserves that original evidence and appends current runtime provenance to `resume-environments.jsonl`. Comparison reports identify checkpoint/model and tokenizer bytes. Resume requires saved snapshots and matching identities; missing or changed inputs stop the run instead of silently downloading replacements. A timeout is an incomplete run, produces a nonzero CLI exit, and prevents `cook` from starting later stages.

PII filtering changes processed training inputs. Original text remains in raw run snapshots and download caches; filtering does not scrub those originals. Choose access controls and retention appropriate to your data. Small sampled benchmarks and regex filters support experiments, not certifications of data quality or model safety.

API serving limits output tokens, batch size and number of completions through `inference.security.max_output_tokens`, `max_batch_size` and `max_choices`. Defaults are at most 256 output tokens (bounded by context), one prompt and one completion. Requests exceeding these limits are rejected; omitted output limits receive the configured default. TLS/shared-token controls do not replace per-user identity or production traffic protection.

AWS launch journals are saved before provisioning steps so interrupted/lost launch responses can be reconciled. AWS training preflight requires a completed local report. Runpod runs local source checks by default and validates the declared bundle; its API/SSH lifecycle is covered by mocked tests, and paid GPU deployment remains unvalidated. Managed `aws-train` preserves recipe optimization settings, installs optional dependencies, and launches DDP with one worker per visible GPU. DDP requires at least two GPUs; native NVFP4 requires Blackwell. Unsupported combinations still fail recipe validation. Deployment-contract tests cover these paths; live cloud GPU validation remains outstanding.

## Troubleshooting

| Symptom | Next step |
| --- | --- |
| Runpod credentials missing | Set `RUNPOD_API_KEY` privately or configure its CLI in the controller environment; see [Runpod setup](docs/RUNPOD.md) |
| Runpod GPU unavailable | Check capacity in the network volume's data center and the configured CUDA/image requirements |
| Runpod cleanup unconfirmed | Inspect the saved job and Pod in the console; use `runpod-job ... --action terminate --run`; keep retained storage for recovery |
| Runpod results unavailable after deletion | Attach the same network volume to a new Pod; SSH fetch needs a live Pod |
| `aws` not found | Install AWS CLI v2; reopen the terminal |
| Profile/credentials unavailable | Configure/login in the same OS environment as the controller |
| `AccessDenied` | Review the denied action with your IAM administrator, including PassRole/bootstrap permissions |
| Spot capacity unavailable | Preview another supported type or public subnet/AZ; retries currently require another explicit run |
| GPU Spot quota exceeded | Request the appropriate regional quota in Service Quotas |
| Local preflight fails | Read `artifacts/aws-preflight.json`; install `dev` dependencies and fix tests/lint |
| Budget exhausted | Review retained ledger reservations and intentionally adjust budget; do not delete the journal to bypass it |
| CUDA unavailable | Check driver, installed wheel and `doctor`; choose an appropriate official PyTorch build |
| WSL HTTPS timeout | Test networking inside Ubuntu before setup; this project does not repair host networking |
| Dataset viewer HTTP 429 | Allow the announced retry cooldown; optionally use `hf auth login` in the same environment. Authentication is not a speed guarantee. |
| vLLM export missing | Provide local config/tokenizer/weights; a checkpoint alone cannot be served |
| Termination timeout | Check EC2 state; use reconciliation after termination and retain the AWS deadline |

## CLI reference and repository layout

| Command | Purpose |
| --- | --- |
| `cook WORKFLOW [--target local\|aws\|runpod] [--aws-config CONFIG] [--runpod-config CONFIG] [--run]` | Ordered stages and export; local default, or one temporary AWS/Runpod host |
| `pipeline CONFIG [--target local\|aws\|runpod] [--aws-config CONFIG] [--runpod-config CONFIG] [--run] [--resume CHECKPOINT]` | One training stage; local default; checkpoint resume currently local only |
| `chat RUN_DIR [--backend torch\|vllm]` | CLI inference from a completed training run |
| `export RUN_DIR` | Export trained text model/tokenizer for HF-compatible inference |
| `multimodal CONFIG [--run] [--resume CHECKPOINT]` | Preview/train local vision-language or Whisper recipes |
| `predict-media CONFIG --media FILE [--model-path EXPORT]` | Generate text from an image or speech clip |
| `publish-model CONFIG [--run]` | Preview/publish an allowlisted model export and card to Hugging Face |
| `stage-config PARENT --kind STAGE --data FILE --output CONFIG` | Derive a post-training recipe from its parent |
| `download-model REPO [--run]` | Preview or execute revision-pinned, capped pretrained acquisition |
| `init-student TEACHER [--architecture FILE]` | Initialize a random native student with the teacher vocabulary |
| `import-model LOCAL_DIR` | Register a local pretrained model for later stages |
| `convert-data SOURCE --format FORMAT --output FILE` | Bounded text/CSV/JSONL conversion |
| `teacher-data TEACHER --prompts FILE --output FILE` | Generate local sequence-distillation data |
| `compare RUN_DIR [--baseline LOCAL_DIR]` | Compare models using saved benchmark examples |
| `evaluate-standard LOCAL_EXPORT [--run]` | Plan or invoke the optional standard harness |
| `runpod-job RUN_DIR --action status\|logs\|fetch\|terminate [--run]` | Inspect a saved Runpod job, fetch results or explicitly terminate its Pod |
| `aws-train CONFIG [--run]` | Plan or execute an S3-backed text or multimodal Spot training stage |
| `aws-serve CONFIG [--action ACTION] [--deployment ID] [--run]` | Plan/deploy persistent hosting or inspect/manage its lifecycle |
| `curate CONFIG [--run]` | Preview or acquire bounded, attributed ingredient collections |
| `dashboard RUN_DIR [--port PORT]` | Serve a local training dashboard; install the dashboard extra |
| `doctor` | Inspect current hardware and PyTorch/CUDA |
| `check-cuda-kernel PATH` | Validate local custom SwiGLU outputs and backward gradients on CUDA |
| `setup [--apply]` | Plan/install hardware-aware PyTorch packages |
| `check-config CONFIG` | Validate runtime, text, multimodal, workflow or publication configuration |
| `resolve-runtime CONFIG [--output FILE]` | Resolve device/precision and optionally persist the result |
| `smoke [--accelerator ...] [--precision ...]` | Synthetic training and checkpoint recovery |
| `setup-vllm [--apply]` | Plan/install optional shared vLLM stack |
| `serve-vllm CONFIG [--run]` | Validate/plan or start local serving |
| `aws-test [CONFIG] [--instance-type TYPE] [--run]` | Preview or execute a budget-reserved Spot test |
| `aws-reconcile RUN_ID [--path CONFIG]` | Recover cloud results and finish cleanup after termination |
| `aws-export RUN_ID` | Verify downloaded AWS results and recover the exported model and inference recipe |
| `chat-api --endpoint HTTPS_URL [--prompt TEXT]` | Send authenticated completions to a remote model; interactive mode omits `--prompt` |

To try a saved checkpoint from an interrupted or ongoing native training run,
use `python -m modelchef.cli chat YOUR_RUN_DIRECTORY --checkpoint step-002000.pt`.
Replace the filename with an actual saved checkpoint in that run's `checkpoints`
directory. This supports the torch and optional vLLM backends and does not mark training complete.
The vLLM path exports the selected checkpoint and reserves 70% of GPU memory;
use it when sufficient GPU memory is available, especially during ongoing training.
For pretraining/midtraining checkpoints, interactive inputs are plain text
continuations unless `stage.prompt_format: chat_transcript` is declared; that
format inserts user/assistant roles and preserves conversation history.

Use `python -m modelchef.cli COMMAND --help` for flags.

- `src/modelchef/`: CLI, corpus processing, tokenizer/model training, benchmarks, inference, hardware policy and AWS/Runpod runners.
- `configs/`: versioned portable defaults; `*.local.yaml` stays local.
- `tests/`: unit tests and small integration checks.
- `scripts/`: optional setup and diagnostic helpers.
- `artifacts/` (including `aws-runs/` and `runpod-runs/`), `.aws-runs/`: ignored reports/checkpoints, cloud journals and budget ledgers.

Credentials, local configurations, historical setup notes, dependency snapshots and run artifacts are excluded from Git. Avoid account-specific IDs and past-run details in committed documentation. As the pipeline evolves, keep each new command's execution location, downloads, costs, reproducibility inputs and validation scope explicit in this tutorial.
