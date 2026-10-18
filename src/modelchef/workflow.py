"""Compose stage recipes into one explicit local model-development workflow."""
import copy
import json
import os
import uuid
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from modelchef.run_config import RunConfig, Section


class WorkflowStage(Section):
    name: str = Field(pattern=r'^[a-zA-Z0-9_-]+$')
    recipe: str
    parent_stage: str | None = None
    teacher_stage: str | None = None
    inherit_architecture: bool = True


class Workflow(Section):
    schema_version: Literal[1] = 1
    name: str = Field(default='cook', pattern=r'^[a-zA-Z0-9_-]+$')
    output_dir: str = 'artifacts/workflows'
    imported_model: str | None = None
    imported_context: int = Field(default=128, ge=8)
    stages: list[WorkflowStage] = Field(min_length=1)
    export: bool = True
    curation_recipe: str | None = None

    @model_validator(mode='after')
    def ordered(self):
        if self.schema_version != 1:
            raise ValueError('Only workflow schema version 1 is supported')
        seen = set()
        for stage in self.stages:
            if stage.name in seen:
                raise ValueError('Workflow stage names must be unique')
            for target in [stage.parent_stage, stage.teacher_stage]:
                if target and target not in seen:
                    raise ValueError('Stage references must point to an earlier stage')
            seen.add(stage.name)
        return self


def cook(path, root, execute=False):
    root, path = Path(root).resolve(), Path(path).resolve()
    config = Workflow.model_validate(yaml.safe_load(path.read_text()))
    recipes, previews = {}, {}
    previous_preview = None
    if config.imported_model:
        from modelchef.pretrained import inspect_model
        previous_preview, _, _ = inspect_model(root / config.imported_model, config.imported_context)
    for stage in config.stages:
        # Recipe paths are project-root relative, matching ordinary pipeline commands.
        recipes[stage.name] = yaml.safe_load((root / stage.recipe).read_text())
        if not isinstance(recipes[stage.name], dict):
            raise TypeError('Stage recipe must contain a mapping')
        for section in ['training', 'stage', 'model', 'tokenizer']:
            if section in recipes[stage.name] and not isinstance(recipes[stage.name][section], dict):
                raise ValueError(f'Stage {stage.name}: {section} must be a mapping')
        if recipes[stage.name].get('training', {}).get('distributed', 'none') != 'none':
            raise ValueError('Run distributed stages separately with torchrun; cook is a single controller')
        # Substitute placeholders for links that execution will resolve, then validate every stage.
        preview = dict(recipes[stage.name])
        preview['stage'] = dict(preview.get('stage', {}))
        if preview['stage'].get('kind', 'pretrain') != 'pretrain':
            parent = previews[stage.parent_stage] if stage.parent_stage else previous_preview
            explicit_parent = preview['stage'].get('parent_run')
            if stage.parent_stage and isinstance(explicit_parent, dict):
                raise ValueError('Choose parent_stage or a recipe parent_run, not both')
            if parent is None and not explicit_parent:
                raise ValueError('A continuation stage requires an earlier run or imported_model')
            use_explicit = isinstance(explicit_parent, dict) or (parent is None and bool(explicit_parent))
            if not use_explicit:
                preview['stage']['parent_run'] = 'resolved-at-execution'
            if stage.inherit_architecture and not use_explicit:
                supplied = preview.get('model', {})
                preview['model'] = parent.model.model_dump()
                for key in ['peft', 'quantization', 'attention_kernel', 'custom_swiglu_path', 'gradient_checkpointing']:
                    if key in supplied:
                        preview['model'][key] = supplied[key]
                preview['tokenizer'] = parent.tokenizer.model_dump()
                if 'packing' in recipes[stage.name].get('tokenizer', {}):
                    preview['tokenizer']['packing'] = recipes[stage.name]['tokenizer']['packing']
        elif stage.parent_stage:
            raise ValueError('A pretraining stage cannot reference parent weights')
        if stage.teacher_stage:
            preview['stage']['teacher_run'] = 'resolved-at-execution'
        previous_preview = RunConfig.model_validate(preview).validate_execution()
        previews[stage.name] = previous_preview
    plan = {'name': config.name, 'mode': 'local', 'stages': config.model_dump()['stages'],
            'imported_model': config.imported_model, 'export': config.export,
            'resolved_recipes': {name: preview.model_dump(mode='json') for name, preview in previews.items()},
            'note': 'Plan only; --run executes stage recipes in order. This does not launch AWS or serve an API.'}
    from modelchef.planning import memory_preview
    plan['memory_plan'] = {name: memory_preview(preview.model_dump(mode='json'), root, gpu_count=1)
                           for name, preview in previews.items()}
    if not execute:
        return plan
    previous_attribution = os.environ.get('MODELCHEF_DATA_ATTRIBUTION')
    previous_remaining = os.environ.get('MODELCHEF_REMAINING_STAGE_TOKENS')
    if config.curation_recipe:
        from modelchef.curation import curate
        curated = curate(root / config.curation_recipe, root, execute=True)
        from modelchef.curation import CurationConfig
        data_config = CurationConfig.model_validate(yaml.safe_load((root / config.curation_recipe).read_text()))
        os.environ['MODELCHEF_DATA_ATTRIBUTION'] = str(root / data_config.output_dir / 'attribution.json')
        plan['curation'] = curated
    from modelchef.chat import export_model
    from modelchef.pipeline import run_pipeline
    from modelchef.pretrained import register_model
    from modelchef.progress import report
    folder = root / config.output_dir / (config.name + '-' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    (folder / 'workflow.json').write_text(config.model_dump_json(indent=2))
    result = {'status': 'running', 'workflow_dir': str(folder), 'stages': {}}
    try:
        previous = register_model(root / config.imported_model, root, config.imported_context) if config.imported_model else None
        for stage_index, stage in enumerate(config.stages):
            recipe = copy.deepcopy(recipes[stage.name])
            recipe['stage'] = dict(recipe.get('stage', {}))
            kind = recipe['stage'].get('kind', 'pretrain')
            if kind != 'pretrain':
                explicit_parent = recipe['stage'].get('parent_run')
                parent = (explicit_parent if isinstance(explicit_parent, dict) else
                          (result['stages'][stage.parent_stage]['run_dir'] if stage.parent_stage else
                           previous or explicit_parent))
                if not parent:
                    raise ValueError('A continuation stage requires an earlier run or imported_model')
                recipe['stage']['parent_run'] = parent
                parent_config = (RunConfig.model_validate_json((Path(parent) / 'config.json').read_text())
                                 if isinstance(parent, str) else None)
                if stage.inherit_architecture and parent_config is not None:
                    supplied = recipe.get('model', {})
                    recipe['model'] = parent_config.model.model_dump()
                    # Architecture comes from the parent; only runtime/adapter choices override it.
                    for key in ['peft', 'quantization', 'attention_kernel', 'custom_swiglu_path', 'gradient_checkpointing']:
                        if key in supplied:
                            recipe['model'][key] = supplied[key]
                    recipe['tokenizer'] = parent_config.tokenizer.model_dump()
                    if 'packing' in recipes[stage.name].get('tokenizer', {}):
                        recipe['tokenizer']['packing'] = recipes[stage.name]['tokenizer']['packing']
            if stage.teacher_stage:
                recipe['stage']['teacher_run'] = result['stages'][stage.teacher_stage]['run_dir']
            report(f'Cooking stage {stage.name}: {kind}.')
            os.environ['MODELCHEF_REMAINING_STAGE_TOKENS'] = str(sum(
                p.training.steps * p.training.micro_batch_size * p.training.gradient_accumulation *
                p.model.context_length for p in
                [previews[s.name] for s in config.stages[stage_index + 1:]]))
            recipe['name'] = config.name + '-' + stage.name
            result['stages'][stage.name] = run_pipeline(RunConfig.model_validate(recipe), root)
            stage_result = result['stages'][stage.name]
            if stage_result.get('status') != 'completed':
                raise ValueError(f"Stage {stage.name} ended with status {stage_result.get('status')}; "
                                 "later stages were not started. Inspect its saved report before resuming.")
            previous = stage_result['run_dir']
            (folder / 'report.json').write_text(json.dumps(result, indent=2))
        result.update(status='completed', final_run=str(previous))
        if config.export:
            result['export_dir'] = str(export_model(previous))
        from modelchef.chat import checkpoint_for
        from modelchef.recovery import checkpoint_to_s3
        checkpoint_to_s3(root, Path(previous), checkpoint_for(Path(previous)))
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        if previous_remaining is None:
            os.environ.pop('MODELCHEF_REMAINING_STAGE_TOKENS', None)
        else:
            os.environ['MODELCHEF_REMAINING_STAGE_TOKENS'] = previous_remaining
        (folder / 'report.json').write_text(json.dumps(result, indent=2))
        if config.curation_recipe:
            if previous_attribution is None:
                os.environ.pop('MODELCHEF_DATA_ATTRIBUTION', None)
            else:
                os.environ['MODELCHEF_DATA_ATTRIBUTION'] = previous_attribution
    return result
