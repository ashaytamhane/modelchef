# Memory planning and sharded training

DDP replicates weights, gradients and optimizer state. FSDP2 shards those tensors
and temporarily gathers block weights for computation. Both process different
examples on each rank. This implementation does not add tensor/pipeline parallelism,
DeepSpeed, CPU/NVMe offload or automatic cloud GPU selection.

## Plan without allocation or spending

```bash
modelchef plan-memory configs/pretrain-demo.yaml --gpu-count 4 --memory-gib 24
```

The command counts unique native parameters algebraically, including tied embeddings,
GQA, biases and MoE experts. It reports FP32 weights (4 bytes/parameter), gradients
(4) and standard AdamW moments (8). Autocast does not convert ModelChef's persistent
master parameters to BF16. FSDP2 keeps master shards in FP32.

Activation and two-unit gather estimates are illustrative scenarios, not bounds.
Full FP32 logits can be significant for large vocabularies; materialized attention
matrices grow quadratically with sequence length. SDPA kernel selection, recomputation,
DDP buckets, foreach/fused optimizer temporaries, padding, allocator fragmentation,
CUDA/NCCL and checkpoint staging affect actual peaks. A low estimate never establishes
that a run fits. Requested tokenizer vocabulary can differ from the learned vocabulary.
Imported/PEFT models and optional optimizer/quantization policies are rejected by this
analytical planner rather than receiving misleading standard-Adam estimates.

Use measured global throughput from the exact hardware/execution configuration:

```bash
modelchef plan-memory configs/pretrain-demo.yaml --gpu-count 4 --hourly-usd 2 \
  --measured-tokens-per-second 10000 --remaining-tokens 1000000000 --topology
```

Hourly price is the combined compute price of all GPUs. Dollars per million tokens
equals `hourly_price * 1e6 / (3600 * global_tokens_per_second)`. Remaining compute
hours equal `remaining_tokens / global_tokens_per_second / 3600`. Storage, setup and
idle time are additional. Do not reuse one configuration's throughput for another.

`--topology` reads local `nvidia-smi topo -m` with a bounded timeout. The raw connectivity
matrix is evidence of links, not a bandwidth benchmark or a promise of scaling.
NVLink/PCIe labels alone cannot identify the cheapest cloud configuration. Remote
catalog GPU names do not establish the actual interconnect. No resource is rented.

## Measure phases

Set `training.profile_phases: true` to record synchronized phase durations and CUDA
allocated/reserved peaks. It is disabled by default because synchronization adds
overhead and may reduce communication/computation overlap. CPU timing is supported,
but CPU memory is reported as unknown rather than zero GPU memory presented as RAM.

Single-process reports include initialization, forward, backward, optimizer updates,
validation, checkpoint writing and reload. DDP reports per-rank phases. FSDP2 reports
per-rank initialization, combined forward/backward, optimizer, validation, checkpoint
and reload phases. Peaks are absolute live allocations, not incremental allocations.
Saved reports on failed runs may not contain a completed phase summary. These are
diagnostic timings, not an unbiased steady-state throughput benchmark.

## Run FSDP2 explicitly

Copy a suitable native recipe outside the repository or to an ignored local recipe.
Set:

```yaml
training:
  distributed: fsdp2
  accelerator: cuda
  precision: bf16
  optimizer: adamw
  profile_phases: true
evaluation:
  evaluate_test: false
  benchmark: none
```

Launch on one node with at least two NVIDIA GPUs and a shared run directory:

```bash
torchrun --standalone --nproc_per_node=2 -m -- modelchef.cli pipeline configs/fsdp.local.yaml --run
```

The initial scope is native dense unquantized pretraining/midtraining, FP32 or supported
BF16, standard AdamW, optional activation checkpointing and ordinary SDPA. Unsupported
PEFT, MoE, FP16, compilation, custom kernels, teacher objectives and posttraining stages
fail validation. FSDP does not replace test isolation; `evaluate_test: false` suppresses
test evaluation and checkpoint reload always probes development validation.

Rank zero prepares data and a CPU initial checkpoint, reusing the existing preparation
interface. **Preparation still requires host RAM for the full model and dataset
preparation selected by the recipe.** This is not yet a host-memory-free 70B initializer.
Each rank builds the training model on the meta device, shards each transformer block
and then the root group, materializes only local parameter shards and broadcasts the
initial CPU state from rank zero. Nonpersistent RoPE buffers are reconstructed. Tied
embedding/head parameters remain together in the root group, which can be the largest
gather group. BF16 compute uses FP32 gradient reductions and FP32 master/optimizer shards.
Every rank participates in validation forwards; rank-zero-only FSDP evaluation would hang.
Gradient accumulation currently reduces each microbatch, favoring bounded sharded
gradient storage over the extra memory of deferred full-gradient synchronization.

Global batch sequences are `micro_batch_size * gradient_accumulation * world_size`.
Blocks are sampled with replacement using per-rank saved generators, as in the existing
pipeline. This feature does not change a recipe to exact epoch coverage.

## Save and recover

FSDP2 checkpoints are committed directories, `checkpoints/step-XXXXXX.dcp`, containing
sharded DCP model/optimizer data, per-rank RNG/sampler state and a checksum manifest.
Writes go to a new incomplete directory; publication happens after all ranks finish.
Preparation, checkpoint metadata/I/O and S3 transfers use CPU/Gloo coordination,
separate from the shorter NCCL training timeout. The CPU I/O group uses the remaining
cloud deadline, with a five-minute minimum grace (one hour without a cloud deadline);
the independent cloud shutdown deadline remains authoritative.
Existing committed checkpoints are never overwritten. Resume from an earlier step can
write a new suffixed checkpoint while preserving the existing one. A failed write can
leave an incomplete directory for diagnosis; retention ignores it. Shared storage must
support atomic directory rename. Storage checks estimate space for the new checkpoint
while previous checkpoints still exist; filesystem quotas and power-loss durability
require operational validation.

```bash
torchrun --standalone --nproc_per_node=2 -m -- modelchef.cli pipeline configs/fsdp.local.yaml \
  --run --resume PATH_TO_RUN/checkpoints/step-000100.dcp
```

Exact resume requires the same recipe, prepared snapshots, tokenizer, PyTorch version
and world size. DCP supports tensor resharding, but a changed world size also changes
per-rank RNG and batch semantics, so ModelChef rejects that as an exact continuation.
Checkpoint reload is checked on validation logits without allocating a second model.
`MODELCHEF_LOCAL_CHECKPOINT_KEEP` controls new-run DCP retention (default two); it never
deletes the checkpoint explicitly used to resume or the original CPU parent checkpoint.
Protected resume paths are resolved to absolute identities. Retention orders updates
numerically, ignores incomplete directories and rejects paths escaping the checkpoint
directory before deleting anything.

These directories are not ordinary `.pt` checkpoints and cannot yet be passed directly
to ModelChef chat/export or used as a new stage's parent. Use the same FSDP recipe's
resume interface. Consolidation/export is a separate future capability; silently
gathering the entire model during checkpointing would defeat the memory contract.

The Runpod guest recognizes the torchrun launch policy, retaining checkpoints on the
network volume. Managed AWS also recognizes this policy and commits complete DCP
directories to its existing S3 recovery location. Only rank zero uploads, while all
ranks wait for success or receive the failure before retention or further updates.
A complete sharded step-zero snapshot is committed before optimizer updates begin;
the CPU preparation `.pt` is not advertised as an FSDP recovery checkpoint. Lazy
AdamW state materialization during that save must leave its step counters at zero.

S3 recovery uploads content-addressed shards, RNG, tokenizer, prepared snapshots and
lineage, then writes an immutable snapshot manifest and finally `latest.json`. A failed
upload leaves the prior complete pointer available. Restoring stages the checkpoint
directory, verifies object sizes/checksums and its embedded DCP manifest, then publishes
the committed path with a directory rename. Existing different checkpoints are refused.
Failed downloads may leave a staging directory for diagnosis. Old S3 snapshots/objects
are retained; this feature does not perform remote garbage collection.

Use the existing recovery CLI with the prefix shown in the managed AWS report:

```bash
python -m modelchef.recovery s3://BUCKET/RUN_PREFIX/recovery /opt/modelchef-training --max-bytes 250000000000
```

It prints the restored `.dcp` directory to pass to the torchrun `--resume` command
above. Restore to the original workspace path for exact recipe/data identities. The
existing cloud controller does not automatically provision a replacement instance or
launch a resumed job. Prefer metadata-only controller results for large runs; complete
checkpoints remain available through S3 recovery independently of result-archive size.
S3 storage and transfer charges remain additional. No storage/cleanup policy changes
or multi-node launch are introduced. Live multi-GPU cloud behavior remains unverified.

## Evidence and limits

Synthetic two-process CPU/Gloo FSDP2 tests check tied weights, a clipped AdamW update
against an unsharded global-batch reference, exact next-update recovery, RNG persistence,
changed-identity rejection, wrong-world-size rejection and checksum corruption. They
exercise real PyTorch sharding and DCP, not mocked collectives. They do not establish
NCCL behavior, CUDA peaks, BF16 numerical quality, GPU throughput or cloud reliability.
A separate two-CUDA-GPU pipeline/resume test skips when hardware is unavailable.
The same real CPU/Gloo checkpoint also round-trips through mocked S3 and reproduces
the next AdamW update exactly. Failure-injection tests cover mid-upload interruption,
missing/corrupt downloads, preservation of the prior S3 pointer and propagation of
rank-zero upload failures to all ranks without hanging.

## Primary references

- [PyTorch FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html): block sharding, meta initialization, mixed precision and DTensor gradient clipping.
- [FSDP2 API](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html): reshard/prefetch policies and their memory tradeoffs.
- [Distributed Checkpoint](https://docs.pytorch.org/docs/stable/distributed.checkpoint.html): distributed state and optimizer restoration; version compatibility limitations.
- [CUDA memory semantics](https://docs.pytorch.org/docs/stable/notes/cuda.html#memory-management): allocated versus reserved memory and allocator behavior.
- [NVIDIA System Management Interface](https://docs.nvidia.com/deploy/nvidia-smi/index.html): topology matrix semantics.
