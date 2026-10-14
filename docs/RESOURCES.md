# Resource-aware preparation

Hardware discovery reports GPU backend, precision support, VRAM and free GPU
memory, plus CPU preparation capacity. CPU capacity uses the minimum of host CPU
count, process count/affinity where available, and container CPU quota. A container
with 128 visible CPUs and a 27.2-core quota is treated as having 27.2 usable cores.

Disk-backed preparation parallelizes independent cleaning and tokenization with
spawned processes. Output remains in input order. Split assignment, deduplication
indexes and file writes stay in the parent process, preserving deterministic data
semantics. Tokenizer fitting still uses training documents only.

The initial worker count leaves one usable core for coordination, caps at 32, and
allows at most one worker per 512 MiB of available memory. Memory availability is
bounded by readable container limits; unknown memory capacity caps workers at four.
This is a conservative planning heuristic, not a guaranteed peak-memory bound.
Small inputs with one batch use the sequential path.

Each batch contains at most 64 documents and approximately 2 MiB of text, except
that a single oversized source record must fit by itself. At most twice the worker
count of batches is queued. Native tokenizer parallelism is disabled inside workers
to prevent nested thread oversubscription. Token limits, BOS/EOS insertion,
document padding and token file checksums retain their original semantics.

Set `MODELCHEF_PREPROCESS_WORKERS=1` to force sequential preparation, or a positive
integer to lower the automatic worker limit. It cannot raise the detected resource
limit. Hardware reports and preparation logs expose the selected worker count.

This does not automatically change the model, training batch size, optimizer,
precision or distributed training configuration. CPU utilization is not a goal by
itself: process startup, IPC, storage bandwidth and parent-side work can limit speed.
Measure end-to-end time on representative data before claiming a speedup. Running
jobs keep the source snapshot uploaded at launch and are not patched in place.
