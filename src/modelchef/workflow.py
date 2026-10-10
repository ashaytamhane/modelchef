"""Compose stage recipes into one explicit local model-development workflow."""
import json
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
            if parent is None:
                raise ValueError('A continuation stage requires an earlier run or imported_model')
            preview['stage']['parent_run'] = 'resolved-at-execution'
            if stage.inherit_architecture:
                supplied = preview.get('model', {})
                preview['model'] = parent.model.model_dump()
                for key in ['peft', 'quantization', 'attention_kernel', 'custom_swiglu_path', 'gradient_checkpointing']:
                    if key in supplied:
                        preview['model'][key] = supplied[key]
                preview['tokenizer'] = parent.tokenizer.model_dump()
        elif stage.parent_stage or stage.teacher_stage:
            raise ValueError('A pretraining stage cannot reference a parent or teacher stage')
        if stage.teacher_stage:
            preview['stage']['teacher_run'] = 'resolved-at-execution'
        previous_preview = RunConfig.model_validate(preview).validate_execution()
        previews[stage.name] = previous_preview
    plan = {'name': config.name, 'mode': 'local', 'stages': config.model_dump()['stages'],
            'imported_model': config.imported_model, 'export': config.export,
            'resolved_recipes': {name: preview.model_dump(mode='json') for name, preview in previews.items()},
            'note': 'Plan only; --run executes stage recipes in order. This does not launch AWS or serve an API.'}
    if not execute:
        return plan
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
        for stage in config.stages:
            recipe = recipes[stage.name]
            recipe['stage'] = dict(recipe.get('stage', {}))
            kind = recipe['stage'].get('kind', 'pretrain')
            if kind != 'pretrain':
                parent = result['stages'][stage.parent_stage]['run_dir'] if stage.parent_stage else previous
                if not parent:
                    raise ValueError('A continuation stage requires an earlier run or imported_model')
                recipe['stage']['parent_run'] = str(parent)
                parent_config = RunConfig.model_validate_json((Path(parent) / 'config.json').read_text())
                if stage.inherit_architecture:
                    supplied = recipe.get('model', {})
                    recipe['model'] = parent_config.model.model_dump()
                    # Architecture comes from the parent; only runtime/adapter choices override it.
                    for key in ['peft', 'quantization', 'attention_kernel', 'custom_swiglu_path', 'gradient_checkpointing']:
                        if key in supplied:
                            recipe['model'][key] = supplied[key]
                    recipe['tokenizer'] = parent_config.tokenizer.model_dump()
            if stage.teacher_stage:
                recipe['stage']['teacher_run'] = result['stages'][stage.teacher_stage]['run_dir']
            report(f'Cooking stage {stage.name}: {kind}.')
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
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (folder / 'report.json').write_text(json.dumps(result, indent=2))
    return result
