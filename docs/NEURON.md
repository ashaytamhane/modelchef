# Trainium training and Inferentia2 compilation

ModelChef has an isolated **NeuronX feasibility integration** for its native dense
decoder. CUDA training, existing recipes and vLLM serving remain unchanged. This
command runs on the machine where you invoke it; it does **not** provision AWS.

Trainium executes training through PyTorch/XLA. Inferentia2 executes compiled
inference graphs through `torch_neuronx`. This implementation uses the supported
NeuronX stack; it does not implement AWS's newer TorchNeuron Native backend.

## 1. Prepare a separate environment

Use an AWS Neuron DLAMI and its supported Python, PyTorch, compiler and runtime
versions. Follow the [official Neuron setup instructions](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/torch-neuronx.html).
Do not install ModelChef's `train` or `inference` extras into this environment:
their CUDA PyTorch/vLLM versions may conflict with the Neuron SDK.

Inside the Neuron environment, install only the project and configuration support:

```bash
python -m pip install -e '.[pipeline]'
```

SDK libraries, NumPy and PyTorch must already be supplied by the compatible Neuron
environment. No AWS credentials or tokens belong in a recipe.

## 2. Prepare packed training data

Copy `configs/neuron-native.yaml` to your own recipe. Set `tokens_path` to an
existing disk-backed pretraining run's `tokens-train.bin` (little-endian uint32
packed training tokens) and set
`vocab_size` to its **actual** tokenizer vocabulary size. Preserve that tokenizer
alongside your data. The integration does not download data or train a tokenizer.

Preview without installing the Neuron SDK or accessing AWS:

```bash
python -m modelchef.cli neuron configs/neuron-native.yaml
```

## 3. Train on Trainium

```bash
python -m modelchef.cli neuron configs/my-neuron.yaml --run
python -m modelchef.cli neuron configs/my-neuron.yaml --run \
  --resume artifacts/neuron/native-probe/step-000005.pt
```

Training uses one logical Neuron core, FP32 AdamW, fixed-size next-token batches
and portable causal attention. The first step compiles the training graph. Logs
report actual optimizer steps and loss. Checkpoints contain model, optimizer,
sampler state and recipe/data fingerprints; resume requires identical inputs.
Dropout is disabled, so no accelerator dropout RNG restoration is needed.
Existing checkpoint files are never overwritten. To resume from an earlier step,
remove later checkpoints yourself or restore a directory containing only the
checkpoint you want to continue; the recipe must stay identical.

## 4. Compile inference for Inferentia2

On a compatible Neuron environment, with the original recipe and checkpoint:

```bash
python -m modelchef.cli neuron configs/my-neuron.yaml --action compile
python -m modelchef.cli neuron configs/my-neuron.yaml --action compile --run \
  --checkpoint artifacts/neuron/native-probe/step-000010.pt
```

The command saves `compiled/model.pt` and a manifest with shape and artifact
hashes. Load it using `torch.jit.load` on compatible Neuron hardware. Inputs must
match the compiled batch/sequence shape and tokenizer. Outputs are vocabulary
logits for each input position. This is **not** a KV-cache generation engine or
vLLM API server. Compiler success alone does not verify numerical accuracy.

## Validation and boundaries

Local CPU tests compare this decoder's logits and gradients with the existing
native decoder. Mocked SDK tests verify checkpoint/resume orchestration, not
accelerator correctness. Hardware execution, compiler operator coverage, memory
fit, throughput and numerical parity still require Trainium/Inferentia tests.

### Managed AWS feasibility probe

The AWS runner additionally accepts `job_backend: neuron_probe` in its `aws_test`
configuration, paired with `pipeline_config` naming a Neuron recipe. This narrow
path requires `trn1.2xlarge`, four optimizer steps, checkpoints every two steps,
batch one, a tiny model and a token file no larger than 1 MB. It uses the AWS
Neuron PyTorch 2.9 Ubuntu DLAMI environment. Set an AWS-owned Neuron AMI, artifact
bucket, existing instance profile, hourly ceiling, budget and a runtime of at
most 60 minutes; root storage is limited to 100 GB.

```bash
python -m modelchef.cli aws-train configs/aws-neuron-probe.local.yaml
python -m modelchef.cli aws-train configs/aws-neuron-probe.local.yaml --run
```

This probe reuses local tests, encrypted source/result uploads, the shared budget
ledger, guest shutdown timer, external AWS deadline and controller cleanup. It
trains, restores a checkpoint, replays steps, compiles and compares compiled
logits with CPU. Results are retained in S3 after termination. It tests Neuron
execution on Trainium; a separate Inferentia2 run is still needed to verify that
deployment target. Synthetic tokens are suitable for this engineering check,
but provide no evidence about assistant quality. This is not the full training
workflow or per-checkpoint S3 recovery integration.

The initial integration excludes HF models, MoE, PEFT, quantization, custom CUDA,
activation checkpointing, dropout, distributed training and the full `cook`
lifecycle. The standalone `neuron` command has no automatic S3 uploads, spending
monitor or instance shutdown. Do not use it for the funded full run until those integrations are
validated. Launching instances remains a separate explicitly authorized action;
our current project constraint is **us-east-1 only**. G-series quota does not
establish Trainium or Inferentia quota.

References: [AWS NeuronX training guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/programming-guide/training/pytorch-neuron-programming-guide.html),
[NeuronX inference tracing API](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/frameworks/torch/torch-neuronx/api-reference-guide/inference/api-torch-neuronx-trace.html).
