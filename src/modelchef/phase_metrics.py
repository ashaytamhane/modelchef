"""Opt-in synchronized measurements for individual execution phases."""
import time
from contextlib import contextmanager


class PhaseMetrics:
    def __init__(self, device, enabled=False):
        self.device, self.enabled, self.rows = device, enabled, {}

    @contextmanager
    def measure(self, phase):
        if not self.enabled:
            yield
            return
        import torch
        cuda = self.device == 'cuda'
        if cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            yield
        finally:
            if cuda:
                torch.cuda.synchronize()
            row = self.rows.setdefault(phase, {'calls': 0, 'seconds': 0.0,
                'peak_allocated_bytes': None, 'peak_reserved_bytes': None})
            row['calls'] += 1
            row['seconds'] += time.perf_counter() - started
            if cuda:
                row['peak_allocated_bytes'] = max(row['peak_allocated_bytes'] or 0,
                                                  torch.cuda.max_memory_allocated())
                row['peak_reserved_bytes'] = max(row['peak_reserved_bytes'] or 0,
                                                 torch.cuda.max_memory_reserved())

    def cuda_peaks(self):
        return {key: max((row[key] or 0 for row in self.rows.values()), default=0)
                for key in ['peak_allocated_bytes', 'peak_reserved_bytes']}
