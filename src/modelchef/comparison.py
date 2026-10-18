"""Compare local checkpoints/exports on identical saved benchmark examples."""
import json
from contextlib import nullcontext
from pathlib import Path

from modelchef.corpus import digest, file_digest
from modelchef.hardware import detect_hardware, resolve_runtime
from modelchef.progress import report as progress


def compare(run_dir, baselines, output):
    import torch

    from modelchef.benchmarks import evaluate_choices
    from modelchef.chat import checkpoint_for
    from modelchef.engines import create_model, load_tokenizer
    from modelchef.run_config import RunConfig
    folder = Path(run_dir)
    examples_path = folder / 'benchmark-examples.json'
    examples = json.loads(examples_path.read_text())
    if not examples:
        raise ValueError('Run has no saved benchmark examples')
    device = resolve_runtime({'precision': 'fp32'}, detect_hardware())['device']
    results = []
    for path in [folder, *map(Path, baselines)]:
        progress('Loading the next comparison model and its tokenizer.')
        if (path / 'report.json').exists():
            config = RunConfig.model_validate_json((path / 'config.json').read_text())
            tokenizer = load_tokenizer(path)
            model = create_model(config.model, tokenizer.get_vocab_size())
            model.load_state_dict(torch.load(checkpoint_for(path), map_location='cpu', weights_only=True)['model'])
            context = config.model.context_length
        else:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            from modelchef.engines import TokenizerRoles
            tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
            if not tokenizer.is_fast:
                raise ValueError('Comparison requires fast tokenizers')
            tokenizer = TokenizerRoles(tokenizer.backend_tokenizer)
            raw = AutoModelForCausalLM.from_pretrained(path, local_files_only=True,
                trust_remote_code=False, torch_dtype=torch.float32)
            class Adapter(torch.nn.Module):
                def __init__(self, network):
                    super().__init__()
                    self.network = network

                def forward(self, ids):
                    return self.network(ids).logits
            raw.to(device).eval()
            model = Adapter(raw)
            context = getattr(raw.config, 'max_position_embeddings', 128)
        model.to(device).eval()
        progress(f'Scoring this model on {len(examples)} saved benchmark examples.')
        scores = evaluate_choices(model, tokenizer, examples, device, context, nullcontext)
        weight_paths = [checkpoint_for(path)] if (path / 'report.json').exists() else (
            sorted(path.glob('*.safetensors')) or sorted(path.glob('*.bin')))
        tokenizer_path = path / 'tokenizer.json'
        results.append({'model': str(path), 'context': context, 'scores': scores,
                        'weights_sha256': {p.name: file_digest(p) for p in weight_paths},
                        'tokenizer_sha256': file_digest(tokenizer_path) if tokenizer_path.exists() else None})
        del model
        if not (path / 'report.json').exists():
            del raw
    report = {'benchmark_sha256': digest(examples_path.read_bytes()), 'examples': len(examples),
              'results': results, 'protocol': 'Shared custom sampled continuation scoring; not official suite scores'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def standard_plan(model_path, tasks=None, limit=100, output='artifacts/standard-evaluation', run=False,
                  device='auto', *, checkpoint=None, batch_size=1, cpu_threads=None):
    """Explicit harness execution may download task data; preview is entirely local."""
    import os
    import subprocess
    import sys
    model = Path(model_path).resolve()
    if not (model / 'config.json').exists():
        raise ValueError('Provide an existing HF export or ModelChef run directory')
    if batch_size < 1 or (cpu_threads is not None and cpu_threads < 1):
        raise ValueError('Batch size and CPU threads must be positive')
    config = json.loads((model / 'config.json').read_text())
    native_run = 'training' in config and 'tokenizer' in config
    preparation = []
    selected = None
    if native_run:
        from modelchef.chat import checkpoint_for
        if checkpoint is None and not (model / 'report.json').is_file():
            raise ValueError('Unfinished run: select an existing checkpoint with --checkpoint')
        selected = checkpoint_for(model, checkpoint)
        if not selected.is_file():
            raise ValueError('Evaluation checkpoint does not exist')
        if not (model / 'tokenizer.json').is_file():
            raise ValueError('Evaluation requires the run tokenizer')
        source = model
        model = Path(output).resolve() / 'model'
        if model.exists():
            raise ValueError('Evaluation model export already exists; choose a fresh --output directory')
        preparation = [[sys.executable, '-m', 'modelchef.cli', 'export', str(source),
                        '--checkpoint', str(selected), '--output', str(model)]]
    elif checkpoint is not None:
        raise ValueError('--checkpoint requires a ModelChef run directory')
    if not native_run and not any(model.glob('*.safetensors')) and not any(model.glob('*.bin')):
        raise ValueError('HF evaluation requires local model weights')
    if not native_run and not any((model / name).is_file() for name in ['tokenizer.json', 'tokenizer.model', 'vocab.json']):
        raise ValueError('HF evaluation requires local tokenizer files')
    runtime = resolve_runtime({'accelerator': device, 'precision': 'fp32'}, detect_hardware())
    if limit < 1:
        raise ValueError('Evaluation limit must be positive')
    tasks = tasks or ['hellaswag', 'piqa', 'arc_easy']
    if any(not task.replace('_', '').isalnum() for task in tasks):
        raise ValueError('Task names must be alphanumeric/underscore')
    if ',' in str(model):
        raise ValueError('Harness model path cannot contain commas')
    command = [sys.executable, '-m', 'lm_eval', 'run', '--model', 'hf', '--model_args',
               f'pretrained={model},trust_remote_code=False,local_files_only=True', '--tasks', *tasks,
               '--limit', str(limit), '--batch_size', str(batch_size), '--device', runtime['device'],
               '--output_path', str(output)]
    environment = {}
    if cpu_threads is not None:
        environment = {'OMP_NUM_THREADS': str(cpu_threads), 'MKL_NUM_THREADS': str(cpu_threads)}
    if run:
        import importlib.util
        if importlib.util.find_spec('lm_eval') is None:
            raise ImportError('Install the evaluation extra before running: pip install "modelchef[evaluation]"')
        env = {**os.environ, **environment}
        for prepare_command in preparation:
            subprocess.run(prepare_command, check=True, env=env)
        progress('Starting the standard evaluation harness; its logs below report dataset loading and task progress.')
        subprocess.run(command, check=True, env=env)
    return {'command': command, 'executed': run, 'device': runtime['device'],
            'preparation_commands': preparation, 'environment': environment,
            'checkpoint': str(selected) if selected else None,
            'note': 'Execution may download benchmark datasets; --limit bounds examples per task, not download bytes. '
                    'A limited task selection is a development diagnostic, not an official CORE score.'}
