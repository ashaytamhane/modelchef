# A reasoning and emotionally aware assistant, trained from scratch

This research recipe mixes general knowledge, reasoning, social context and sensory
concepts during **pretraining from random initialization**. Personality is supplied
by the system prompt; it is not one permanent identity. The goal is contextual
judgment: understand what people may feel, consider alternative interpretations,
and respond usefully without treating speculation as fact.

Learning descriptions of pain is not evidence that the model experiences pain.
Likewise, a fluent explanation is not evidence of correct reasoning. These are
capabilities to evaluate, not assumptions about consciousness or neuroscience.

## 1. Ingredients

The executable [curation recipe](../configs/data-assistant-persona-sample.yaml)
records authors, source links, licenses, transformations and snapshot hashes.

| Ingredient | Role | Source and terms recorded in the recipe |
|---|---|---|
| FineWeb-Edu | General educational text | [Penedo et al., FineWeb](https://arxiv.org/abs/2406.17557); ODC-BY dataset, underlying document rights retained |
| Cosmopedia v2 | Explanatory educational text | [Ben Allal et al., SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus); ODC-BY |
| Smol-SmolTalk | Instruction following | [Ben Allal et al., SmolLM2](https://arxiv.org/abs/2502.02737); Apache 2.0, constituent attribution retained |
| SmolTalk system conversations | Prompt-controlled personas | [SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2), original systemchats source; Apache 2.0 |
| PersonaChat | Dialogue conditioned on a supplied profile | [Zhang et al.](https://arxiv.org/abs/1801.07243); CC BY 4.0, bavard mirror credited |
| EmpatheticDialogues | Emotional situations and listener responses | [Rashkin et al.](https://arxiv.org/abs/1811.00207); CC BY-NC 4.0, brianist mirror credited |
| SmolTalk2 everyday reasoning | Complete thought traces and final answers | SmolLM team's Qwen3-32B-generated conversations; Apache 2.0. Synthetic traces are not verified solutions |
| Lancaster Sensorimotor Norms | Sensory and action associations, including internal bodily sensations | [Lynott, Connell, Brysbaert, Brand and Carney](https://doi.org/10.3758/s13428-019-01316-z); [official OSF data](https://osf.io/7emr6/), CC BY 4.0 |

The sensory table is serialized into labeled text with its original numeric values
and scale interpretation. It contains **human ratings of concepts**, not neuron
activity, pain intensity measurements or causal mappings from emotions to neurons.
The sample takes 100 ordered CSV rows, not a representative sensory vocabulary.
The other samples are also bounded ordered source slices.

Emotion is currently represented mainly through conversations. Detailed first-person
accounts of physical pain, grief and life experiences remain a data-curation gap;
the presence of empathy data does not establish broad coverage of those subjects.
DEAP, WESAD and EmoPain are candidates for a future physiological-signal experiment,
not ingredients silently included in this text recipe. They require suitable
encoders, paired supervision, access checks and subject-disjoint evaluation.

## 2. Cook the local integration test

Use an environment with the project's training and pipeline dependencies installed.
Run from the repository root:

```bash
python -m modelchef.cli check-config configs/data-assistant-persona-sample.yaml
python -m modelchef.cli curate configs/data-assistant-persona-sample.yaml
python -m modelchef.cli curate configs/data-assistant-persona-sample.yaml --run
python -m modelchef.cli cook configs/workflow-assistant-persona-sample.yaml
python -m modelchef.cli cook configs/workflow-assistant-persona-sample.yaml --run
```

Preview commands perform no acquisition or training. Curation stores raw snapshots,
prepared JSONL collections and `attribution.json` under the configured data folder.
Checksum-verified reuse avoids downloads. The **16 MB corpus-response cap is only a
sample-test setting**, independent of whether execution is local or on AWS.

The workflow connects three stages automatically:

1. **Integrated pretraining:** educational documents, full social/reasoning
   transcripts and sensory ratings. Next-token prediction trains on all tokens,
   including the context and reasoning, rather than masking the input prompt.
2. **Reasoning/social midtraining:** continued next-token training on the complete
   transcripts, using the same tokenizer and inherited model weights.
3. **Assistant SFT:** response-masked instruction/persona/empathy training, with reasoning replay. This
   teaches assistant expression after reasoning data has already entered pretraining.
   Direct-answer and reasoning-enabled examples retain their mode instructions.
   Inclusion does not establish that the tiny model has learned reliable mode control.

Complete reasoning traces are preserved; oversized targets are skipped rather than
clipped. Related examples retain a stable `split_group` across text pretraining and
SFT so a reused conversation/persona does not move from holdout into training.
Keep seed and split fractions identical across stages to preserve this policy.
Near duplicates across unrelated sources are not guaranteed to be isolated.

The local model has 631,424 parameters: two layers, width 128, two attention heads,
one KV head, a 2,048-token byte-BPE vocabulary and context length 4,096. It runs four
optimizer updates per stage on CPU FP32. This is an integration test, **not a useful
assistant or a quality benchmark**. Downloaded/prepared counts are not the number
of examples consumed in four updates; inspect token manifests and training reports.

## 3. Try prompt-controlled personality

Use the actual `final_run` path printed by `cook`; replace the placeholder below.

```bash
python -m modelchef.cli chat YOUR_FINAL_RUN --system-prompt "Be warm and thoughtful. Consider uncertainty." --prompt "My friend cancelled our plans. Should I be angry?" --max-tokens 64
python -m modelchef.cli chat YOUR_FINAL_RUN --system-prompt "Be concise and reserved. Consider uncertainty." --prompt "My friend cancelled our plans. Should I be angry?" --max-tokens 64
```

The tiny test will produce poor text. These commands verify inference and prompt
handling, not whether personality or judgment has been learned. The native export
includes a chat template for the configured transcript format. A vLLM server and
AWS resources are not launched by this local workflow.

### Move beyond the four-update test

The [longer local pilot](../configs/workflow-assistant-persona-pilot.yaml) uses the
same cached ingredients and tiny architecture, with 200 pretraining updates,
100 midtraining updates and 200 SFT updates. Warmup spans 20, 10 and 20 updates
respectively. SFT uses a peak learning rate of 0.0005 rather than the integration
test's 0.0001. These are explicit experimental settings, not universal optimal values.

```bash
python -m modelchef.cli cook configs/workflow-assistant-persona-pilot.yaml
python -m modelchef.cli cook configs/workflow-assistant-persona-pilot.yaml --run
```

Each stage uses `evaluation.batches: all`, evaluating every **packed example** in
its validation and test splits, rather than just the first batch. Reports include
coverage and evaluated token counts. The 5M packing-token allowance covers this
sample; it does not acquire more data. Text packing still drops an incomplete final
context block. Check actual coverage before interpreting results for a larger corpus.

Before interpreting weak loss improvement, test whether the SFT path can fit a
small subset of a completed native SFT run:

```bash
python scripts/overfit_probe.py YOUR_COMPLETED_SFT_RUN
python scripts/overfit_probe.py YOUR_COMPLETED_SFT_RUN --examples 4 --steps 200 --run
```

This generic CPU diagnostic selects four short training examples and four from
each saved holdout, preserves their assignments, uses a constant learning rate
after warmup, and saves `overfit-report.json`. It compares full-subset loss before
and after training. A large training-loss reduction establishes that this path can
memorize; it does not establish generalization. Worsening holdout loss is expected
in this deliberately overfitted experiment. It supports unquantized native SFT runs
and never downloads data or launches AWS resources.

Save qualitative probes from checkpoints using identical greedy decoding:

```bash
python scripts/checkpoint_samples.py YOUR_RUN configs/assistant-persona-pilot-prompts.yaml --output artifacts/checkpoint-probes.json
```

Use this on any completed stage, or on checkpoints already saved during a run.
The supplied probes cover anger, pain-related perspective, reasoning-mode requests,
personality changes and an arithmetic control. They are unscored diagnostics;
fluent prose or declining loss alone does not demonstrate the target capabilities.
The helper intentionally runs on CPU to avoid interfering with a GPU training job.

## 4. Scale with a measured opening phase

[The 350M-class workflow](../configs/workflow-assistant-persona-350m.yaml) is a
**scale recipe**, not a promised quality result. Its `curation_recipe` points to
`configs/data-assistant-persona-full.yaml`: complete public Parquet records are
adapted and mixed on disk under `data/raw/assistant-persona-350m` before training.
The full acquisition cap is 100 GB; the 16 MB cap belongs only to the local sample.
Do not copy the small sample into the full-data folder.

The recipe uses a 28-layer, width-1,024 decoder, grouped-query attention,
32,768-token vocabulary and 2,048-token context. It requests CUDA BF16, activation
checkpointing, automatic attention-kernel selection and fused AdamW. Compilation
remains opt-in. With the requested vocabulary this is 349,234,176 parameters.
CUDA/custom-kernel speedups must be benchmarked on the chosen AWS
GPU; the CPU test does not validate them.

The scale recipes now measure their opening training phase on the real corpus and
continue the same model only when the runtime projection and memory check pass.
The first ten updates warm up the GPU; updates 11–50 measure throughput. A
checkpoint is saved before the gate decision, and a rejected gate prevents later
stages from starting. There is no separate pilot model to discard.
The current step counts, 100 GB input cap and 8B token cap are hypotheses, not a
$2,000 cost estimate. Concatenating collection files gives observed counts, not
exact token-weighted sampling. Bulk curation uses deterministic hash ordering and
SQLite, while text processing and token arrays use disk storage. BPE fits on a
5,000-document training-only sample; every accepted model-training document can
still be tokenized. Structured stages count and skip examples that exceed their
context window, preserving complete targets. Exact deduplication is enabled;
approximate near-deduplication is disabled in this budgeted recipe.

`training.performance_gate` is optional. These recipes use a 1.5× runtime margin,
30 minutes of overhead, and a 90% peak allocated GPU-memory ceiling. `cook`
includes the configured work in all later stages when projecting remaining time.
The AWS guest supplies its deadline before setup begins, so acquisition and setup
consume the same budgeted runtime. On a local machine the gate uses the stage's
runtime limit. Its decision is saved in `performance-gate.json` and narrated on
screen. A throughput gate estimates feasibility; it does not guarantee quality,
future throughput or exact billing.

The scale recipes use BF16, automatic PyTorch attention selection, fused AdamW,
TF32 where applicable and activation checkpointing. Microbatch eight with four
accumulations preserves 65,536 token positions per optimizer update. Compilation
remains an explicit opt-in; the recipe does not assume an unmeasured speedup.
Recheck batch size on GPUs with less memory than an L40S.

Set the instance, hourly ceiling, runtime and per-run reservation in an AWS
resource configuration. The reservation must include EBS, IPv4, artifact storage
and a safety allowance and fit the shared overall budget. A long runtime ceiling
is a cutoff, not a promise to consume that time: successful completion or a
failed gate terminates the guest earlier. No AWS spending is part of local tests.

Copy [the full AWS resource template](../configs/aws-assistant-full.yaml) to
`configs/aws-assistant-full.local.yaml`, fill in your private bucket and instance
profile, and verify the region, subnet, hourly ceiling and shared ledger. It is
configured for a four-vCPU L40S Spot host with a $1,500 reservation within a $2,000
overall limit, including a $200 artifact allowance. Its 400-hour deadline is an
upper bound; the measured gate may stop it much earlier. Check current prices and
credit expiry before choosing this envelope.

```bash
# Preview; no instance launch or full dataset download:
python -m modelchef.cli cook configs/workflow-assistant-persona-350m.yaml --target aws --aws-config configs/aws-assistant-full.local.yaml

# Run after reviewing the displayed cloud reservation:
python -m modelchef.cli cook configs/workflow-assistant-persona-350m.yaml --target aws --aws-config configs/aws-assistant-full.local.yaml --run
```

### Check learning before scaling

Use the longer local pilot to compare initial and final loss over all packed
held-out examples, confirm checkpoint reload and inspect identical prompts at
successive checkpoints. The separate four-example overfit probe tests whether
the optimizer can memorize a small training set; worsening held-out loss during
that probe is expected and must not be reported as generalization. Falling loss
alone does not establish emotional understanding, useful reasoning or personality
control. Keep run-specific metrics and generated responses in ignored artifacts.
PII filtering preserves decimal sensorimotor ratings while redacting recognized
phone and email patterns; it remains a heuristic.

### Recover from a cloud interruption

Managed text training commits an encrypted S3 snapshot at step zero, every saved
checkpoint, stage completion and final export. It retains weights, optimizer and
RNG states, tokenizer, exact raw/split/token snapshots, attribution and parent-stage
checkpoints. A `latest.json` pointer is written only after all referenced uploads
finish, so an interrupted transfer leaves the previous complete snapshot available.
The guest retains the latest two local checkpoints; committed S3 snapshots remain.

The managed AWS report records `recovery_uri`. Fetch it without launching compute:

```bash
python -m modelchef.recovery s3://BUCKET/RUN_PREFIX/recovery ./recovered \
  --profile modelchef --region us-east-1
```

Downloads verify file sizes and SHA256 and refuse to overwrite different files.
The manifest records the original workspace. For exact training continuation,
restore to that same path on a replacement CUDA guest, retrieve the run's saved
`source.tar.gz` code bundle, and use the restored stage config and checkpoint:

```bash
python -m modelchef.cli pipeline artifacts/training/RUN/config.json --run \
  --resume artifacts/training/RUN/checkpoints/step-NNNNNN.pt
```

ModelChef restores the optimizer, sampler and random states and verifies data and
tokenizer identity. It resumes a stage; later workflow stages can then continue
from its completed output. It does not automatically provision a replacement Spot
instance. The guest shutdown and independent AWS deadline terminate compute after
completion or timeout. S3 artifacts are deliberately retained and continue to
incur storage charges until removed.

For large runs, AWS resource settings can use `result_mode: metadata`: the
controller downloads small reports instead of loading multi-gigabyte model
archives into RAM. Complete checkpoints and the final export remain available
through the recovery URI. Workflows with remote curation require
`preflight_data_dir` pointing to representative local JSONL samples for the CPU
check; those samples are used only for preflight, not cloud training.

## 5. Taste-test judgment, then publish

Evaluate general knowledge and verifiable reasoning alongside emotional
understanding. Keep [EmoBench](https://aclanthology.org/2024.acl-long.326/) and other
chosen evaluations out of training. Check alternative explanations, uncertainty,
perspective-taking and constructive responses; warmth alone is insufficient.
Compare the same questions across different personality prompts to ensure style
changes do not undermine accuracy. Compare against a general-data-only baseline
before claiming emotional training improves reasoning.

[The publication example](../configs/publish-assistant-research.yaml) requests a
private, noncommercial research release and points to the attribution manifest.
Fill in the real export path, repository, evaluation and limitations, then preview:

```bash
python -m modelchef.cli publish configs/publish-assistant-research.yaml
```

Only add `--run` when ready to publish. Verify that the manifest describes the
actual training ingredients, including all upstream constituent restrictions.
For a full training run, replace the sample attribution path with its full-corpus
manifest. Dataset licenses are not automatically relicensed by the model card;
noncommercial access alone does not settle every redistribution question.
