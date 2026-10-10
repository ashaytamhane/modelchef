import json
import sys
from types import SimpleNamespace

import pytest

from modelchef.publishing import PublishConfig, publish


def export(root):
    folder = root / 'export'
    folder.mkdir()
    for name in ['config.json', 'tokenizer_config.json', 'tokenizer.json']:
        (folder / name).write_text('{}')
    (folder / 'model.safetensors').write_bytes(b'weights')
    (folder / '.env').write_text('NEVER_UPLOAD=secret')
    (folder / 'raw.jsonl').write_text('private training data')
    (folder / 'checkpoint.pt').write_bytes(b'optimizer')
    return PublishConfig(export_dir=str(folder), repo_id='example/test-model', license='apache-2.0',
                         summary='Test model', training_data='Authored fixtures',
                         evaluation='Pipeline tests only', limitations='Random test weights')


def fake_hub(monkeypatch, private=True, token='mock-token'):
    calls = []
    api = SimpleNamespace(
        create_repo=lambda **kw: calls.append(('create', kw)),
        repo_info=lambda **kw: SimpleNamespace(private=private),
        create_commit=lambda **kw: (calls.append(('commit', kw)) or
            SimpleNamespace(commit_url='https://huggingface.co/example/test-model/commit/test', oid='test')))
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda **kw: api, get_token=lambda: token,
        CommitOperationAdd=lambda **kw: SimpleNamespace(**kw)))
    return calls


def test_preview_is_offline_allowlisted_and_requires_license(tmp_path, monkeypatch):
    config = export(tmp_path)
    calls = fake_hub(monkeypatch)
    result = publish(config, tmp_path)
    assert not calls and not result['uploaded'] and result['private']
    assert {item['name'] for item in result['files']} == {
        'config.json', 'tokenizer_config.json', 'tokenizer.json', 'model.safetensors', 'README.md'}
    assert 'Authored fixtures' in result['model_card']
    assert 'mock-token' not in json.dumps(result)
    with pytest.raises(ValueError):
        PublishConfig.model_validate({**config.model_dump(), 'license': ''})


def test_explicit_upload_and_visibility_match(tmp_path, monkeypatch):
    config = export(tmp_path)
    calls = fake_hub(monkeypatch)
    result = publish(config, tmp_path, run=True)
    assert result['uploaded'] and result['commit_sha'] == 'test'
    assert calls[0][1]['private'] is True
    names = {item.path_in_repo for item in calls[1][1]['operations']}
    assert '.env' not in names and 'raw.jsonl' not in names and 'README.md' in names
    assert 'mock-token' not in json.dumps(result)
    calls = fake_hub(monkeypatch, private=False)
    with pytest.raises(ValueError, match='visibility differs'):
        publish(config, tmp_path, run=True)
    assert len(calls) == 1
    calls = fake_hub(monkeypatch, token=None)
    with pytest.raises(ValueError, match='hf auth login'):
        publish(config, tmp_path, run=True)
    assert not calls


def test_publication_bounds_credentials_and_shard_integrity(tmp_path):
    config = export(tmp_path)
    config.max_upload_bytes = 1
    with pytest.raises(ValueError, match='max_upload_bytes'):
        publish(config, tmp_path)
    config.max_upload_bytes = 10000
    folder = tmp_path / 'export'
    (folder / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'layer': '../missing'}}))
    with pytest.raises(ValueError, match='missing or excluded'):
        publish(config, tmp_path)
    (folder / 'model.safetensors.index.json').unlink()
    (folder / 'config.json').write_text(json.dumps({'token': 'hf_' + 'x' * 25}))
    with pytest.raises(ValueError, match='Possible credential'):
        publish(config, tmp_path)
