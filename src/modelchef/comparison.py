"""Compare local checkpoints/exports on identical saved benchmark examples."""
import json
from contextlib import nullcontext
from pathlib import Path

import torch

from modelchef.benchmarks import evaluate_choices
from modelchef.chat import checkpoint_for
from modelchef.corpus import digest
from modelchef.engines import create_model, load_tokenizer
from modelchef.hardware import detect_hardware, resolve_runtime
from modelchef.run_config import RunConfig


def compare(run_dir, baselines, output):
    folder = Path(run_dir)
    examples_path = folder / 'benchmark-examples.json'
    examples = json.loads(examples_path.read_text())
    if not examples:
        raise ValueError('Run has no saved benchmark examples')
    device = resolve_runtime({'precision': 'fp32'}, detect_hardware())['device']
    results = []
    for path in [folder, *map(Path, baselines)]:
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
        scores = evaluate_choices(model, tokenizer, examples, device, context, nullcontext)
        results.append({'model': str(path), 'context': context, 'scores': scores})
        del model
        if not (path / 'report.json').exists():
            del raw
    report = {'benchmark_sha256': digest(examples_path.read_bytes()), 'examples': len(examples),
              'results': results, 'protocol': 'Shared custom sampled continuation scoring; not official suite scores'}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def standard_plan(model_path, tasks=None, limit=100, output='artifacts/standard-evaluation', run=False):
    """Explicit harness execution may download task data; preview is entirely local."""
    import subprocess
    import sys
    model = Path(model_path).resolve()
    if not (model / 'config.json').exists():
        raise ValueError('Provide an existing HF export directory')
    if limit < 1:
        raise ValueError('Evaluation limit must be positive')
    tasks = tasks or ['hellaswag', 'piqa', 'arc_easy']
    if any(not task.replace('_', '').isalnum() for task in tasks):
        raise ValueError('Task names must be alphanumeric/underscore')
    if ',' in str(model):
        raise ValueError('Harness model path cannot contain commas')
    command = [sys.executable, '-m', 'lm_eval', 'run', '--model', 'hf', '--model_args',
               f'pretrained={model},trust_remote_code=False', '--tasks', *tasks,
               '--limit', str(limit), '--batch_size', '1', '--output_path', str(output)]
    if run:
        subprocess.run(command, check=True)
    return {'command': command, 'executed': run,
            'note': 'Execution may download benchmark datasets; --limit bounds scored examples, not download bytes.'}
