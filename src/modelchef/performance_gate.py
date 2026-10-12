"""Project remaining work from a measured training window before committing runtime."""
import math


def project_run(*, measured_tokens, measured_seconds, remaining_tokens,
                available_seconds, safety_multiplier, overhead_seconds,
                peak_allocated_bytes=0, total_memory_bytes=0,
                max_allocated_memory_fraction=0.9):
    values = [measured_tokens, measured_seconds, safety_multiplier]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Performance gate requires positive finite measurements and runtime')
    if remaining_tokens < 0 or overhead_seconds < 0:
        raise ValueError('Remaining work and overhead must be nonnegative')
    if not math.isfinite(available_seconds) or available_seconds < 0:
        raise ValueError('Available runtime must be finite and nonnegative')
    throughput = measured_tokens / measured_seconds
    projected = remaining_tokens / throughput * safety_multiplier + overhead_seconds
    fraction = peak_allocated_bytes / total_memory_bytes if total_memory_bytes else None
    passed = projected <= available_seconds and (
        fraction is None or fraction <= max_allocated_memory_fraction)
    return {'passed': passed, 'measured_tokens_per_second': throughput,
            'projected_remaining_seconds': projected, 'available_seconds': available_seconds,
            'remaining_tokens': remaining_tokens, 'safety_multiplier': safety_multiplier,
            'peak_allocated_memory_fraction': fraction,
            'scope': 'Runtime projection, not a quality forecast or exact AWS invoice.'}
