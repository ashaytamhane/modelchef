"""Reuse immutable disk snapshots for fresh model experiments without copying weights."""
import json
import os
import shutil

from modelchef.run_config import RunConfig


def reuse_prepared_data(config, root, destination):
    source = (root / config.data.prepared_run).resolve()
    previous = RunConfig.model_validate_json((source / 'config.json').read_text())
    if config.stage.kind != 'pretrain' or previous.stage.kind != 'pretrain' or config.data.storage != 'disk':
        raise ValueError('Prepared reuse requires disk-based pretraining snapshots')
    for section in ['data', 'processing', 'tokenizer', 'evaluation']:
        current = getattr(config, section).model_dump()
        original = getattr(previous, section).model_dump()
        if section == 'data':
            current.pop('prepared_run')
            original.pop('prepared_run')
        if section == 'evaluation':
            for key in ['evaluate_test', 'batch_size', 'batches']:
                current.pop(key)
                original.pop(key)
        if current != original:
            raise ValueError(f'Prepared reuse requires matching {section} settings')
    if config.model.context_length != previous.model.context_length:
        raise ValueError('Prepared reuse requires matching context length')
    required = ['raw.jsonl', 'source-manifest.json', 'benchmark-examples.json',
                'benchmark-manifest.json', 'processing.json', 'train.jsonl',
                'validation.jsonl', 'test.jsonl', 'tokenizer.json', 'tokens.pt',
                'tokens-train.bin', 'tokens-validation.bin', 'tokens-test.bin']
    optional = ['processing-sources.json', 'benchmark-text.txt', 'tokenizer-roles.json',
                'data-attribution.json']
    for name in required:
        if not (source / name).is_file():
            raise ValueError(f'Prepared snapshot missing {name}')
    for name in required + optional:
        original, target = source / name, destination / name
        if not original.exists():
            continue
        if name.endswith(('.bin', '.jsonl')):
            # Immutable snapshots: never write these links during the fresh run.
            os.link(original, target)
        else:
            shutil.copyfile(original, target)
    (destination / 'prepared-source.json').write_text(json.dumps({'run': str(source)}))
