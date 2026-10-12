"""Diagnose an SFT training path by fitting a few existing training examples.

Uses saved train/validation/test assignments from a completed run, performs no
downloads, and never treats memorization as evidence of generalization.
"""
import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from modelchef.corpus import group_split
from modelchef.engines import create_model, load_tokenizer
from modelchef.pipeline import evaluate, run_pipeline
from modelchef.run_config import RunConfig


def select_examples(folder, examples, config):
    rows = {}
    for split in ['train', 'validation', 'test']:
        records = [json.loads(line) for line in (folder / (split + '.jsonl')).read_text(
            encoding='utf-8').splitlines()]
        if any('prompt' not in row or 'response' not in row for row in records):
            raise ValueError('Probe requires saved prompt/response SFT splits')
        # Deterministic small-subset selection; short examples make the probe simpler.
        records.sort(key=lambda row: (len(row['prompt']) + len(row['response']), row['source_index']))
        if len(records) < examples:
            raise ValueError(f'{split}: fewer than {examples} examples available')
        rows[split] = records[:examples]
        for row in rows[split]:
            if 'split_group' not in row:
                # Preserve original assignments for legacy document-shuffled runs too.
                nonce = 0
                while True:
                    key = f"overfit:{split}:{row['source_index']}:{nonce}"
                    if group_split(key, config) == split:
                        row['split_group'] = key
                        break
                    nonce += 1
            if group_split(row['split_group'], config) != split:
                raise ValueError('Saved group assignment differs from the configured seed/fractions')
    return rows


def score(folder, checkpoint):
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    tokenizer = load_tokenizer(folder)
    model = create_model(config.model, tokenizer.get_vocab_size())
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True)['model'])
    blocks = torch.load(folder / 'tokens.pt', weights_only=True)
    return {name: evaluate(model, parts[:2], 'all', 1, 'cpu', nullcontext)
            for name, parts in blocks.items()}


def probe(folder, root, examples=4, steps=200, learning_rate=.001, execute=False):
    folder, root = Path(folder).resolve(), Path(root).resolve()
    saved = json.loads((folder / 'report.json').read_text())
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    if saved.get('status') != 'completed' or config.stage.kind != 'sft':
        raise ValueError('Probe requires a completed SFT run')
    if config.model.backend != 'native' or config.model.quantization != 'none':
        raise ValueError('This CPU probe supports unquantized native models')
    if examples < 1 or steps < 2 or learning_rate <= 0:
        raise ValueError('Positive examples/LR and at least two steps are required')
    rows = select_examples(folder, examples, config)
    plan = {'parent_run': str(folder), 'training_examples': examples,
            'heldout_examples_per_split': examples, 'steps': steps,
            'learning_rate': learning_rate, 'device': 'cpu',
            'scope': 'Memorization diagnostic; heldouts are not expected to improve.'}
    if not execute:
        return plan
    from uuid import uuid4
    data = root / 'data' / 'probes' / ('overfit-' + uuid4().hex[:12])
    data.mkdir(parents=True)
    path = data / 'examples.jsonl'
    path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n'
        for split in ['train', 'validation', 'test'] for row in rows[split]), encoding='utf-8')
    config = config.model_copy(deep=True)
    config.model.attention_kernel = 'auto'
    config.model.custom_swiglu_path = None
    config.name = 'overfit-probe'
    config.data.local_paths = []
    config.data.local_path = str(path)
    config.data.max_tokens = 1000000
    config.processing.near_dedup = False
    config.stage.parent_run = str(folder)
    config.training.accelerator, config.training.precision = 'cpu', 'fp32'
    config.training.distributed, config.training.compile = 'none', False
    config.training.optimizer_kernel = 'default'
    config.training.steps = steps
    config.training.micro_batch_size = 1
    config.training.gradient_accumulation = 1
    config.training.learning_rate = learning_rate
    config.training.weight_decay = 0
    config.training.schedule = 'constant'
    config.training.warmup_steps = min(10, steps - 1)
    config.training.evaluate_every = max(1, steps // 4)
    config.training.log_every = max(1, steps // 10)
    config.training.checkpoint_every = max(1, steps // 4)
    config.training.max_runtime_seconds = 1200
    config.evaluation.benchmark, config.evaluation.batches = 'none', 'all'
    (data / 'recipe.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json'),
                                                   sort_keys=False), encoding='utf-8')
    prepared = run_pipeline(config, root, prepare_only=True)
    run = Path(prepared['run_dir'])
    before = score(run, prepared['checkpoint'])
    trained = run_pipeline(config, root, resume=prepared['checkpoint'])
    after = score(run, trained['checkpoint'])
    result = {**plan, 'run_dir': str(run), 'before': before, 'after': after,
              'training_loss_reduction': before['train']['loss'] - after['train']['loss'],
              'checkpoint_recovery': trained['checkpoint_recovery'],
              'skipped_optimizer_updates': trained['skipped_optimizer_updates']}
    (run / 'overfit-report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--examples', type=int, default=4)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    print(json.dumps(probe(args.run_dir, Path.cwd(), args.examples, args.steps,
                           args.learning_rate, args.run), indent=2))
