"""Human-readable CLI progress on stderr, independent of result output."""

import threading
import time
from contextlib import contextmanager

_state = threading.local()

_activities = {
    'cook': 'Reading the ordered training-stage recipes and selected execution target.',
    'doctor': 'Checking Python, PyTorch, hardware availability and GPU arithmetic.',
    'setup': 'Inspecting hardware and preparing the requested PyTorch installation.',
    'setup-vllm': 'Checking inference prerequisites and preparing the vLLM installation.',
    'serve-vllm': 'Validating serving configuration; execution starts the API server.',
    'smoke': 'Testing device selection, forward/backward training and checkpoint recovery.',
    'pipeline': 'Reading the training recipe and selected execution target before preparing data or compute.',
    'multimodal': 'Checking bounded image/audio inputs and local model files; execution fine-tunes and exports.',
    'predict-media': 'Loading the local multimodal model and processor to generate text from media.',
    'publish-model': 'Checking export files and model card; only --run creates a Hugging Face commit.',
    'download-model': 'Resolving model metadata and size; execution downloads approved files.',
    'import-model': 'Checking local model/tokenizer files and recording an imported parent run.',
    'export': 'Loading the checkpoint and writing an inference-compatible model export.',
    'chat': 'Loading the model and tokenizer for text generation.',
    'chat-api': 'Connecting to the configured HTTPS model endpoint.',
    'aws-export': 'Verifying the completed cloud archive and preparing its serving export.',
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
        print(f'[ModelChef] {message}', file=sys.stderr, flush=True)


@contextmanager
def operation(command, quiet=False):
    previous = getattr(_state, 'quiet', False)
    _state.quiet = quiet
    started = time.monotonic()
    report(f'Starting {command}.')
    report(_activities.get(command, 'Executing the requested operation.'))
    try:
        yield
    except BaseException:
        report(f'{command} did not complete ({time.monotonic() - started:.1f}s elapsed).')
        raise
    else:
        report(f'{command} completed ({time.monotonic() - started:.1f}s elapsed).')
    finally:
        _state.quiet = previous
