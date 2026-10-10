"""Human-readable CLI progress on stderr, independent of result output."""

import threading
import time
from contextlib import contextmanager

_state = threading.local()

_activities = {
    'doctor': 'Checking Python, PyTorch, hardware availability and GPU arithmetic.',
    'setup': 'Inspecting hardware and preparing the requested PyTorch installation.',
    'setup-vllm': 'Checking inference prerequisites and preparing the vLLM installation.',
    'serve-vllm': 'Validating serving configuration; execution starts the API server.',
    'smoke': 'Testing device selection, forward/backward training and checkpoint recovery.',
    'pipeline': 'Validating the recipe; execution prepares data, trains and evaluates the model.',
    'download-model': 'Resolving model metadata and size; execution downloads approved files.',
    'import-model': 'Checking local model/tokenizer files and recording an imported parent run.',
    'export': 'Loading the checkpoint and writing an inference-compatible model export.',
    'chat': 'Loading the model and tokenizer for text generation.',
    'compare': 'Loading checkpoints and scoring models on the saved evaluation examples.',
    'teacher-data': 'Loading the teacher and generating responses from the supplied prompts.',
    'evaluate-standard': 'Preparing the benchmark command; execution runs the evaluation harness.',
    'check-cuda-kernel': 'Loading/compiling the kernel and checking forward/backward correctness.',
    'aws-test': 'Preparing the Spot test; execution runs local preflight and monitors cloud cleanup.',
    'aws-train': 'Preparing the training bundle; execution monitors the Spot job and cleanup.',
    'aws-serve': 'Processing the requested persistent API hosting action.',
    'aws-reconcile': 'Recovering the recorded cloud run status and reconciling cleanup.',
    'convert-data': 'Reading bounded input records and writing normalized JSONL.',
    'init-student': 'Loading teacher vocabulary and initializing the student model.',
    'stage-config': 'Reading parent settings and writing the requested training-stage recipe.',
    'check-config': 'Reading and validating the configuration file.',
    'resolve-runtime': 'Detecting hardware and resolving the requested device and precision.',
}


def report(message):
    import sys
    if not getattr(_state, 'quiet', False):
        print(f'[LoomLM] {message}', file=sys.stderr, flush=True)


@contextmanager
def operation(command, quiet=False, interval=10):
    previous = getattr(_state, 'quiet', False)
    _state.quiet = quiet
    started = time.monotonic()
    stop = threading.Event()

    def heartbeat():
        while not stop.wait(interval):
            import sys
            print(f'[LoomLM] {command} still running ({time.monotonic() - started:.0f}s elapsed).',
                  file=sys.stderr, flush=True)

    worker = None
    report(f'Starting {command}. Progress appears here; results follow on stdout.')
    report(_activities.get(command, 'Executing the requested operation.'))
    if not quiet:
        worker = threading.Thread(target=heartbeat, daemon=True)
        worker.start()
    try:
        yield
    except BaseException:
        report(f'{command} did not complete ({time.monotonic() - started:.1f}s elapsed).')
        raise
    else:
        report(f'{command} completed ({time.monotonic() - started:.1f}s elapsed).')
    finally:
        stop.set()
        if worker:
            worker.join()
        _state.quiet = previous
