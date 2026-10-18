"""Shared allocation-free memory previews for local and cloud launch surfaces."""
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig
from modelchef.training_memory import memory_plan


def memory_preview(raw, root, *, gpu_count=None, memory_bytes=None, hourly_usd=None):
    root = Path(root)
    if 'stages' in raw:
        return {'status': 'per_stage', 'stages': {stage['name']: memory_preview(
            yaml.safe_load((root / stage['recipe']).read_text(encoding='utf-8')), root,
            gpu_count=gpu_count, memory_bytes=memory_bytes, hourly_usd=hourly_usd)
            for stage in raw['stages']}}
    if 'modality' in raw:
        return {'status': 'unavailable', 'reason': 'Analytical memory planning supports native text models only'}
    try:
        config = RunConfig.model_validate(raw)
        if config.stage.parent_run:
            parent = config.stage.parent_run
            if not isinstance(parent, str) or not (root / parent / 'config.json').is_file():
                return {'status': 'unavailable', 'reason': 'Parent architecture/tokenizer must be resolved first'}
            parent_config = RunConfig.model_validate_json((root / parent / 'config.json').read_text())
            if config.model != parent_config.model or config.tokenizer != parent_config.tokenizer:
                return {'status': 'unavailable', 'reason': 'Parent overrides require resolved architecture/tokenizer metadata'}
        result = memory_plan(config, gpu_count=gpu_count or 1, memory_bytes=memory_bytes,
                             hourly_usd=hourly_usd)
    except ValueError as error:
        return {'status': 'unavailable', 'reason': str(error)}
    world_size = 1 if config.training.distributed == 'none' else gpu_count
    result['configured_strategy'] = config.training.distributed
    result['execution_world_size'] = world_size
    result['global_batch_sequences'] = (config.training.micro_batch_size *
        config.training.gradient_accumulation * world_size) if world_size is not None else None
    result['global_batch_input_tokens'] = (result['global_batch_sequences'] *
        config.model.context_length) if world_size is not None else None
    result['status'] = 'estimated'
    result['gpu_count_verified'] = gpu_count is not None
    result['capacity_verified'] = memory_bytes is not None
    if gpu_count is None:
        result['limitations'].append('GPU count is unknown; the displayed one-GPU scenario is illustrative.')
    return result
