"""Resolve bounded HF recipe references into pinned, local imported runs."""
import json
from pathlib import Path

from modelchef.run_config import HuggingFaceModel, RunConfig


def resolve_references(config, root, resume=None):
    from modelchef.pretrained import download_model, register_model

    root = Path(root).resolve()
    references = {field: value for field in ['parent_run', 'teacher_run']
                  if isinstance(value := getattr(config.stage, field), HuggingFaceModel)}
    if not references:
        return config, {}
    resolved = config.model_copy(deep=True)
    manifest = {}
    shared = {}
    if resume:
        folder = Path(resume).resolve().parent.parent
        manifest = json.loads((folder / 'model-references.json').read_text())
        saved = RunConfig.model_validate_json((folder / 'config.json').read_text())
    for field, reference in references.items():
        requested = reference.model_dump(mode='json')
        if resume:
            if manifest.get(field, {}).get('requested') != requested:
                raise ValueError('Hugging Face reference changed; resume uses its original pinned model')
            imported = getattr(saved.stage, field)
        else:
            context = reference.context_length or config.model.context_length
            key = (reference.repo_id, reference.revision, reference.max_download_bytes, context)
            if key not in shared:
                acquisition = download_model(reference.repo_id, root, reference.revision,
                                             reference.max_download_bytes, run=True)
                imported = register_model(acquisition['path'], root, context)
                shared[key] = imported, acquisition
            imported, acquisition = shared[key]
            manifest[field] = {'requested': requested, 'resolved_revision': acquisition['revision'],
                               'sha256': acquisition['sha256'], 'imported_run': str(imported)}
        setattr(resolved.stage, field, str(imported))
        if field == 'parent_run':
            parent = RunConfig.model_validate_json((Path(imported) / 'config.json').read_text())
            overrides = {key: getattr(config.model, key) for key in
                         ['peft', 'quantization', 'attention_kernel', 'custom_swiglu_path',
                          'gradient_checkpointing']}
            resolved.model = parent.model.model_copy(update=overrides, deep=True)
            resolved.tokenizer = parent.tokenizer.model_copy(deep=True)
    return RunConfig.model_validate(resolved.model_dump(mode='json')).validate_execution(), manifest
