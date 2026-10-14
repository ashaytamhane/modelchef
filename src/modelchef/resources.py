"""Usable container resources and bounded, ordered CPU work scheduling."""
import math
import multiprocessing
import os
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def cpu_resources(cgroup=Path('/sys/fs/cgroup')):
    host = os.cpu_count() or 1
    usable = float(host)
    affinity = None
    if hasattr(os, 'sched_getaffinity'):
        affinity = len(os.sched_getaffinity(0))
        usable = min(usable, affinity)
    process_count = getattr(os, 'process_cpu_count', lambda: None)()
    if process_count:
        usable = min(usable, process_count)
    quota = None
    try:
        value, period = (cgroup / 'cpu.max').read_text().split()
        if value != 'max':
            quota = int(value) / int(period)
    except (OSError, ValueError, ZeroDivisionError):
        try:
            value = int((cgroup / 'cpu/cpu.cfs_quota_us').read_text())
            period = int((cgroup / 'cpu/cpu.cfs_period_us').read_text())
            if value > 0 and period > 0:
                quota = value / period
        except (OSError, ValueError):
            pass
    if quota is not None:
        usable = min(usable, quota)
    return {'host_logical_cpus': host, 'affinity_cpus': affinity,
            'quota_cpus': quota, 'usable_cpus': max(.01, usable)}


def available_memory(cgroup=Path('/sys/fs/cgroup')):
    available = None
    try:
        lines = Path('/proc/meminfo').read_text().splitlines()
        available = next(int(line.split()[1]) * 1024 for line in lines
                         if line.startswith('MemAvailable:'))
    except (OSError, StopIteration, ValueError):
        pass
    for limit_file, used_file in [('memory.max', 'memory.current'),
                                  ('memory/memory.limit_in_bytes',
                                   'memory/memory.usage_in_bytes')]:
        try:
            remaining = max(0, int((cgroup / limit_file).read_text()) -
                            int((cgroup / used_file).read_text()))
            available = remaining if available is None else min(available, remaining)
        except (OSError, ValueError):
            continue
    return available


def preparation_plan():
    cpu = cpu_resources()
    memory = available_memory()
    # Leave capacity for the reader/writer and bound process startup/IPC overhead.
    workers = min(32, max(1, math.floor(cpu['usable_cpus']) - 1))
    if memory is not None:
        workers = min(workers, max(1, memory // (512 * 1024**2)))
    else:
        workers = min(workers, 4)
    override = os.environ.get('MODELCHEF_PREPROCESS_WORKERS')
    if override is not None:
        requested = int(override)
        if requested < 1:
            raise ValueError('MODELCHEF_PREPROCESS_WORKERS must be positive')
        workers = min(workers, requested)
    return {**cpu, 'available_memory_bytes': memory, 'workers': workers,
            'max_inflight_batches': workers * 2, 'batch_records': 64,
            'batch_text_bytes': 2 * 1024**2}


def record_batches(rows, plan):
    batch, size = [], 0
    for row in rows:
        text = row.get('text', '')
        estimate = len(text) * 4 if isinstance(text, str) else 0
        if batch and (len(batch) >= plan['batch_records'] or
                      size + estimate > plan['batch_text_bytes']):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += estimate
    if batch:
        yield batch


def ordered_work(rows, function, initializer, initargs):
    """Preserve input order, cap prefetched text, propagate errors, close workers."""
    plan = preparation_plan()
    batches = iter(record_batches(rows, plan))
    first = next(batches, None)
    if first is None:
        return
    second = next(batches, None)
    if plan['workers'] == 1 or second is None:
        initializer(*initargs)
        yield from function(first)
        if second is not None:
            yield from function(second)
        for batch in batches:
            yield from function(batch)
        return
    print(f"CPU preparation: {plan['workers']} workers; "
          f"{plan['usable_cpus']:.2f} usable CPUs; bounded ordered batches", flush=True)
    pool = ProcessPoolExecutor(max_workers=plan['workers'],
                               mp_context=multiprocessing.get_context('spawn'),
                               initializer=initializer, initargs=initargs)
    pending = deque()
    try:
        pending.extend([pool.submit(function, first), pool.submit(function, second)])
        while len(pending) < plan['max_inflight_batches']:
            batch = next(batches, None)
            if batch is None:
                break
            pending.append(pool.submit(function, batch))
        while pending:
            yield from pending.popleft().result()
            batch = next(batches, None)
            if batch is not None:
                pending.append(pool.submit(function, batch))
    finally:
        for future in pending:
            future.cancel()
        pool.shutdown(wait=True, cancel_futures=True)
