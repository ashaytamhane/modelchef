# Quality and acceptance criteria

ModelChef's quality is assessed against reproducible evidence. A working integration,
a hardware simulation, a measured speedup and a capable trained model are different results.

| Dimension | What the project provides | Acceptance evidence still required |
| --- | --- | --- |
| Accessibility | Offline starter data; progressive tutorials; strict annotated recipes; progress and actionable errors | Independent users completing the tutorials without assistance |
| Workflow coverage | Scratch training, midtraining, SFT, DPO, RLVR, distillation, pretrained models, LoRA/QLoRA, vision-language/Whisper fine-tuning, export, media inference, Hub publishing, CLI chat and secured text API serving | Real workload coverage across architectures and deployment environments |
| Reliability | Checkpoint recovery, snapshot/checksum validation, failure reports, local preflight, Linux/Windows CI and build checks | Repeated clean installations and sustained workloads across supported dependency combinations |
| Data scale | Bounded acquisitions, JSONL shards, SQLite/LSH deduplication and mapped-token batching | Corpus-scale throughput, disk use and peak-memory measurements; distributed preprocessing and structured-data streaming |
| Model quality | Isolated evaluation samples, domain benchmark routing, local baseline comparison and standard evaluation harness | Published trained artifacts and complete comparable suites with confidence intervals; meaningful training budgets |
| Performance | SDPA choices, compilation, fused/foreach optimizers, mixed precision, checkpointing, custom CUDA and optional native FP4 | Repeated measurements against the same unoptimized recipe on each target GPU; report compilation cost and memory alongside throughput |
| Operations | Explicit AWS lifecycle, shutdown deadlines, budget ledger, launch reconciliation, durable serving and API request limits | Live managed training/serving, interruption drills, load/security testing, immutable environment images and ongoing operational ownership |

## Reproduce the software checks

The [execution tutorial](EXECUTION.md) describes `pipeline`/`cook` target selection.
AWS workflow tests run relocated bundles through real local training and export, and
simulate cloud dispatch/guest launch. This verifies recipe links and packaging; it
does not establish live Spot availability or general cloud reliability. A bounded
two-stage scratch/SFT workflow has also passed on EC2 T4 with FP16, including
checkpoint reload, result upload/download, export recovery and confirmed termination.
Live vLLM hosting, interruption drills and sustained cloud workloads still need validation.

Use a Python 3.11, 3.12 or 3.13 environment with the documented platform dependencies:

```bash
python -m ruff check src tests scripts/check_docs.py
python scripts/check_docs.py
python -m pytest -q
python -m pip wheel --no-deps . --wheel-dir artifacts/build
```

CPU CI exercises the offline pipeline, dense/MoE DDP, recovery, export and simulated
cloud contracts. CUDA tests exercise real training when a compatible GPU is visible;
CPU runners skip those tests. Native NVFP4 prerequisite checks and simulations do not
establish numerical quality or performance on Blackwell. A simulator cannot establish
real CUDA compilation compatibility, NCCL performance, MPS behavior or AWS cleanup.
Multimodal integrations use tiny real PaliGemma/Whisper models and synthetic media;
Hub publishing uses mocked API calls, so CI cannot establish live upload permissions.

## Measure a recipe rather than assuming a speedup

Copy the same recipe into baseline and candidate files. Keep data, seed, architecture,
precision, batch size and number of optimizer steps identical. Change only the execution
policy being investigated. Run each several times, preserving their reports, environment
manifests and metrics. Compare median steady-state tokens per second, total elapsed time
including compilation, peak accelerator memory and evaluation loss. Confirm recovery and
export correctness for the candidate. Faster kernels are not automatically better when
compilation dominates a short run or reduced precision damages quality.

For AWS, choose a compatible AMI/driver and GPU, preview the budgeted job, then explicitly
launch it. The CPU preflight establishes data/objective/recovery compatibility; it does
not substitute for the guest's GPU execution. Spot tests and training terminate after
completion; persistent API hosting follows its explicit lifecycle instead.

## Boundaries

### Consecutive sentence cleanup

Raw-text pretraining/midtraining preparation supports optional document-local cleanup:

```yaml
processing:
  collapse_consecutive_duplicate_sentences: true
  minimum_sentence_words: 6
```

The shared memory/disk cleaning path keeps the first sentence in an adjacent run,
compares whitespace-normalized text, and preserves case and punctuation. Short
sentences and intervening content break runs. It uses conservative `.?!` sentence
boundaries with common English abbreviation guards; it is not a multilingual
sentence parser, semantic repetition detector, or corpus-wide deduplicator. No
language model or tokenizer download is required. Existing quality/line-repetition
filters still apply. Counts include `sentence_cleanup_documents`,
`duplicate_sentences_removed`, and `duplicate_sentence_chars_removed` (including
whitespace); these count cleanup operations even if a later filter drops a record.
The option is disabled by default and rejected for structured training stages.

For already-split corpora, clean only explicitly selected files without recomputing
split membership. This command streams one bounded JSONL record at a time:

```bash
modelchef clean-text train.jsonl --output train-cleaned.jsonl
```

It preserves document order/metadata, updates existing ModelChef text `sha256`
fields when text changes, and reports input/output file hashes and removal counts.
Unchanged records retain their original bytes. `--text-field`,
`--minimum-sentence-words`, and `--max-record-bytes` control the bounded operation.
An existing output is refused. An explicit `--in-place` instead of `--output`
uses a temporary file in the same directory, flushes it, then atomically replaces
the selected path. Other hard links retain the original bytes. Failed processing
leaves the previous destination intact; a filesystem crash can leave a temporary
file. Atomic publication requires filesystem support for rename (in-place) or
exclusive Windows rename or POSIX hard links (new output). Temporary disk space
must fit the cleaned output.

This is text cleanup only: it does not regenerate packed tokens or update snapshot
manifests. Prefer a new output and repack in a new preparation directory using the
existing tokenizer; preserve source snapshots and fixed validation/test files.
Enabling the recipe option during fresh preparation cleans before splitting, so
it can change hash-based split assignments. Do not rerun a split to clean an
existing training partition. Changed processing settings prevent silent reuse of
an incompatible prepared snapshot. This feature makes no claim to fix model
generation loops or improve factual accuracy.

Disk mode currently covers local JSONL pretraining and midtraining. Exact deduplication
uses a SQLite index; near deduplication uses deterministic MinHash/LSH candidate search
and exact Jaccard verification. LSH can miss duplicate pairs. Token batches use mapped
files, but tokenizer vocabulary learning still has its own memory cost. Benchmark
decontamination indexes and individual documents must fit in memory. Disk space grows
with snapshots and preprocessing indexes; plan capacity and retention before scaling.

Budget estimates and shutdown mechanisms reduce exposure; they are not a guarantee that
all AWS charges stop at an exact dollar amount. Account billing is the authoritative
source. Production API operators must own credential rotation, TLS, network restrictions,
patching, monitoring, backups and availability requirements described in SECURITY.md.

No software change alone establishes a 10/10 model-quality, scalability or production
rating. Those dimensions require the acceptance evidence above; scores should follow
the evidence rather than serve as a completion switch.

When both `processing.exact_dedup` and `processing.near_dedup` are false, disk processing streams filtered documents directly to split JSONL files without SQLite. Split membership remains seeded; rows retain input order within each split. Immutable disk snapshots use hard links on the same filesystem and byte copies otherwise.
