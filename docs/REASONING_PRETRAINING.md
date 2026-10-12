# Reasoning-first pretraining from scratch

This experimental recipe trains a native approximately 1B decoder from random
weights on complete worked reasoning examples. It has one `pretrain` stage;
there is no imported parent, response-only SFT, RL, or live teacher model.
The data itself contains synthetic demonstrations from a teacher.

Each training document is ordered as:

```text
User: problem
Assistant: <think>
reasoning demonstration
</think>
final answer
```

Next-token cross entropy covers the entire document, including the question,
thought and answer. At inference, the thought tokens precede and condition the
answer. They remain generated text, not independently verified internal thoughts.

## Ingredients and provenance

[OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k)
provides math, science, code and puzzle examples, with DeepSeek-R1 reasoning and
solutions in its `metadata` split. The dataset declares Apache-2.0. Its original
problems have constituent sources and applicable upstream terms; preserve those
sources rather than treating the dataset license as proof of every problem's
rights. The adapter retains the `source` and `domain` fields.

The current bounded snapshot retains only mathematics from OpenThoughts after
the short-example filter. The upstream dataset's broader domains should not be
confused with this snapshot's coverage. See the [dataset card](REASONING_DATASET_CARD.md)
for the measured composition and remaining coverage gaps.

Cite [Guha et al., OpenThoughts: Data Recipes for Reasoning Models
(2025)](https://arxiv.org/abs/2506.04178), including the complete author list from
the paper when publishing. The original work trained pretrained models; this
from-scratch recipe is an experiment, not a reproduction of its quality claims.

[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) supplies
short worked mathematics examples. It declares Apache-2.0 and retains original
source categories; preserve their terms. Cite [Li et al., NuminaMath
(2024)](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf).
The recipe records all authors listed in its dataset citation.

[SmolTalk2](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2) supplies
short Qwen3-32B reasoning traces for everyday conversation, system-conditioned
roles and persona-derived multi-turn instruction following. These new subsets
declare Apache-2.0. They are demonstrations, not verified social or emotional
judgments. Original system instructions are retained, rather than selecting one
fixed persona for the model.

## Prepare and run

Install the pipeline and optional low-memory optimizer dependencies in your
CUDA environment, then curate the bounded corpus:

```bash
uv pip install -e '.[pipeline,optimizer-8bit,dashboard]'
python -m modelchef.cli curate configs/data-reasoning-first.yaml --run
python -m modelchef.cli curate configs/data-reasoning-social.yaml --run
python scripts/compose_corpus.py configs/compose-reasoning-first.yaml
export MODELCHEF_DATA_ATTRIBUTION="$PWD/data/raw/reasoning-first-balanced/attribution.json"
export MODELCHEF_LOCAL_CHECKPOINT_KEEP=1
python -m modelchef.cli pipeline configs/reasoning-first-1b.yaml --run
```

The technical-source acquisition cap is 2.8 GB; the social-source cap is 512 MB.
The observed combined download was about 1.70 GB. Curation reads up to 90,000 OpenThoughts source examples and 50,000 NuminaMath examples, and
retains complete problem/reasoning/answer triples up to 3,800 UTF-8 bytes. This
strongly favors short traces; source-domain proportions are measured, not fixed.
NuminaMath final boxed answers are extracted with balanced-brace parsing; the
preceding worked solution becomes the thought trace. Known NuminaMath sources
`gsm8k`, `math` and `amc_aime` are excluded, without claiming complete benchmark
decontamination. Problem-identity groups are shared across both ingredients.
Downloaded Parquet files and output collections have SHA256 manifests.
The composition recipe applies seeded per-source caps to avoid letting the
largest ingredient dominate. Its initial output has 16,368 examples: 7,991
NuminaMath, 2,112 OpenThoughts, 4,000 system-conditioned, 1,987 multi-turn and 278
everyday examples. Byte/token proportions differ from example proportions.
Composition deduplicates exact text and establishes normalized prompt groups.

The model has 40 layers, width 1,536, 24 query heads, 6 KV heads and a 4,096-wide
SwiGLU feed-forward network. Its requested vocabulary is 8,192. Context is
2,048 tokens, microbatch is 1, and BF16 computation uses experimental
`adamw8bit_in_backward` with activation checkpointing. The 8-bit optimizer
reduces optimizer-state memory; master weights remain FP32. This policy requires
accumulation 1, no global gradient clipping, no compilation and a single GPU.

`tokenizer.packing: document` pads each complete example to one window. Padding
and targets crossing to the following example are masked. Oversized examples
are skipped whole, with counts recorded. No answer is silently cut off to fit.
Trailing masked padding is trimmed before the training forward pass, so short
examples do not need full-window computation.
Because padding is included in stored token-file sizes, use `tokens_seen`,
`target_tokens_per_second` and
`nonpadding_tokens` when assessing real training exposure.

Training is bounded to 10 hours per invocation, with an initial recovery
checkpoint and subsequent checkpoints every 1,000 updates. Only the latest is
retained when the environment above is set. Saving briefly requires room for
both old and new files. A runtime cutoff saves a resumable checkpoint and then
reports the cutoff; it does not mean the planned token budget was completed.

## Reuse prepared data with fresh weights

To try a different initialization or model size without repeating preprocessing,
copy the stage recipe to an ignored `configs/*.local.yaml` and set
`data.prepared_run` to the existing run directory. Keep its data, processing,
tokenizer, evaluation and context settings identical. This starts fresh weights;
use `--resume` below to continue existing weights and optimizer state instead.
The source and new run must be on the same filesystem because large immutable
snapshots are hard-linked. Do not edit linked corpus or token files.

## Inspect and resume

Use the run directory printed by training:

```bash
python -m modelchef.cli dashboard RUN_DIR --port 6009
python scripts/chat_checkpoint.py RUN_DIR --max-tokens 1024
python -m modelchef.cli pipeline configs/reasoning-first-1b.yaml --run \
  --resume RUN_DIR/checkpoints/step-NNNNNN.pt
```

For generation, enter `User: What is 12 plus 7? Assistant:`. The CPU checkpoint
script prints the raw continuation, including any generated think markers and
answer. Early checkpoints may not respect the format or solve the problem.
Loading a checkpoint does not supply the correct reasoning or answer for it.

Validation and test groups are separated by problem identity. Validation loss
is measured every 1,000 steps, but it is not a reasoning-accuracy score. Inspect
held-out answers and format completion as well as loss. Synthetic traces can
contain errors and benchmark overlap; no standard-suite decontamination or
published benchmark score is claimed by this recipe.
