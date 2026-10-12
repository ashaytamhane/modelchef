# Reasoning-first balanced v1: dataset card

This research prototype teaches a randomly initialized decoder to generate a
worked trace before an answer. It combines existing public demonstrations;
curation does not establish that the demonstrations are correct or that the
trained model reasons reliably.

## Composition

| Ingredient | Retained examples | Demonstration source |
| --- | ---: | --- |
| OpenThoughts short examples | 2,112 | DeepSeek-R1; retained examples are mathematics |
| NuminaMath worked solutions | 7,991 | Published mathematical solutions |
| SmolTalk2 system-conditioned reasoning | 4,000 | Qwen-generated traces with varied system instructions |
| SmolTalk2 multi-turn reasoning | 1,987 | Persona-derived instruction-following conversations |
| SmolTalk2 everyday reasoning | 278 | Everyday conversation traces |
| Total composed | 16,368 | Counts describe examples, not token mixture weights |

Training quality filters retain 15,925 documents. Problem-group splitting gives
14,334 training, 761 validation and 830 test documents; one oversized training
document is skipped whole at token packing. The trainable split contains about
8.19 million nonpadding tokens. Repeated passes increase exposure, not unique
data diversity.

The composed corpus SHA256 is
`bec8fb04ba5d00e77b795ce255038f22b7a18f86916cf466248608a034c20cbd`.
Run manifests record the processed splits and tokenizer separately.

## Preparation and reproducibility

The recipes read bounded source prefixes, retain complete examples up to 3,800
UTF-8 bytes, and apply seeded per-source composition caps and exact-text
deduplication. Mathematical final boxed answers are extracted with balanced
brace parsing. Each target is represented as `<think>…</think>` followed by a
nonempty answer. The 16,368 composed targets passed this structural check;
structure does not verify their content.

Normalized problem identities group related examples before split assignment.
Document packing preserves complete examples and masks padding and targets
crossing document boundaries. PII redaction is disabled in this experiment to
avoid damaging mathematical expressions; this is not a privacy clearance for
redistribution. Raw data and run histories remain excluded from Git.

Reproduce acquisition and composition with
[the tutorial](REASONING_PRETRAINING.md). The local
`data/raw/reasoning-first-balanced/attribution.json` records source manifests,
hashes, retained counts, licenses and citation references.

## Coverage and quality limitations

This is primarily mathematics plus conversational instruction following.
Varied system prompts do not establish psychological personality coverage.
There is no validated emotion, pain, sensory grounding, science or coding
coverage, and no claim of human-like thought or experience. Synthetic traces
can be plausible but incorrect, repetitive or biased.

The NuminaMath source categories `gsm8k`, `math` and `amc_aime` are excluded.
This is partial source filtering, not complete benchmark decontamination.
Validation loss measures next-token prediction; separate held-out answer
correctness, trace completion and persona-following evaluations are needed
before making capability claims. The active v1 snapshot is frozen; broader
coverage belongs in a separately versioned experiment.

## Attribution, rights and publication

[OpenThoughts-114k](https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k),
[NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT), and the used
new [SmolTalk2 subsets](https://huggingface.co/datasets/HuggingFaceTB/smoltalk2)
declare Apache-2.0. Preserve constituent source terms and teacher provenance;
dataset-level metadata alone does not settle every upstream right.

Cite [Guha et al., OpenThoughts](https://arxiv.org/abs/2506.04178),
[Li et al., NuminaMath](https://github.com/project-numina/aimo-progress-prize/blob/main/report/numina_dataset.pdf),
and the [SmolLM3/SmolTalk2 release](https://huggingface.co/blog/smollm3).
Before Hugging Face publication, include complete author citations and source
notices, review constituent licenses and privacy, and publish the exact data
and model provenance with measured evaluations. This card is not publication
approval or a claim of ownership over upstream material. No dataset or model
has been published by this experiment.
