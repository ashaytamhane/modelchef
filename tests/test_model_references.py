import json

import pytest

from modelchef.model_references import resolve_references
from modelchef.run_config import RunConfig


def test_model_reference_resolution_inherits_parent_and_pins_resume(tmp_path, monkeypatch):
    parent = tmp_path / 'parent'
    parent.mkdir()
    imported = RunConfig(model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4,
        'kv_heads': 2, 'intermediate_size': 64, 'context_length': 32}, tokenizer={'vocab_size': 300})
    (parent / 'config.json').write_text(imported.model_dump_json())
    downloads = []

    def download(repo, root, revision, max_bytes, run):
        downloads.append((repo, revision, max_bytes, run))
        return {'path': str(parent), 'revision': 'fixed-sha', 'sha256': {'model.safetensors': 'hash'}}

    monkeypatch.setattr('modelchef.pretrained.download_model', download)
    monkeypatch.setattr('modelchef.pretrained.register_model', lambda *args: str(parent))
    source = RunConfig(stage={'kind': 'sft',
        'parent_run': {'repo_id': 'org/tiny', 'context_length': 32, 'max_download_bytes': 1000},
        'teacher_run': {'repo_id': 'org/tiny', 'context_length': 32, 'max_download_bytes': 1000}},
        data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'})
    resolved, manifest = resolve_references(source, tmp_path)
    assert len(downloads) == 1  # Shared parent/teacher acquisition is not duplicated.
    assert resolved.model.hidden_size == 32 and resolved.tokenizer.vocab_size == 300
    assert resolved.stage.parent_run == resolved.stage.teacher_run == str(parent)
    assert manifest['parent_run']['resolved_revision'] == 'fixed-sha'
    run = tmp_path / 'run'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'config.json').write_text(resolved.model_dump_json())
    (run / 'model-references.json').write_text(json.dumps(manifest))
    again, _ = resolve_references(source, tmp_path, run / 'checkpoints/step-000001.pt')
    assert len(downloads) == 1 and again == resolved
    changed = source.model_copy(deep=True)
    changed.stage.parent_run.revision = 'different'
    with pytest.raises(ValueError, match='reference changed'):
        resolve_references(changed, tmp_path, run / 'checkpoints/step-000001.pt')


def test_from_scratch_rejects_parent_weights():
    with pytest.raises(ValueError, match='choose midtrain'):
        RunConfig(stage={'kind': 'pretrain', 'parent_run': {'repo_id': 'org/tiny'}})


def test_reference_schema_rejects_unbounded_or_invalid_downloads():
    for reference in [{'repo_id': '../secrets'}, {'repo_id': 'org/tiny', 'max_download_bytes': 0}]:
        with pytest.raises(ValueError):
            RunConfig(stage={'kind': 'midtrain', 'parent_run': reference})


def test_workflow_remote_parent_plan_stays_offline(tmp_path, monkeypatch):
    import yaml

    from modelchef.workflow import cook
    monkeypatch.setattr('modelchef.pretrained.download_model',
                        lambda *args, **kwargs: pytest.fail('Planning must not acquire a model'))
    recipe = {'stage': {'kind': 'sft', 'parent_run': {'repo_id': 'org/tiny'}},
              'data': {'provider': 'local_jsonl', 'local_path': 'data.jsonl'}}
    (tmp_path / 'stage.yaml').write_text(yaml.safe_dump(recipe))
    workflow = {'name': 'remote', 'stages': [{'name': 'sft', 'recipe': 'stage.yaml'}]}
    path = tmp_path / 'workflow.yaml'
    path.write_text(yaml.safe_dump(workflow))
    assert cook(path, tmp_path)['resolved_recipes']['sft']['stage']['parent_run']['repo_id'] == 'org/tiny'
