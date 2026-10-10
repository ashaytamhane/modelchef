import io
import json
import sys
import tarfile
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('pydantic')

from ashayllm.aws_training import build_bundle, training_user_data
from ashayllm.data_tools import convert_data
from ashayllm.pretrained import download_model
from ashayllm.run_config import RunConfig


def test_converter_is_bounded_and_does_not_overwrite(tmp_path):
    source, target = tmp_path / 'input.csv', tmp_path / 'output.jsonl'
    source.write_text('prompt,response\nQuestion,Answer\nOther,Second\n')
    result = convert_data(source, target, 'csv', max_documents=1)
    assert result['documents'] == 1
    assert json.loads(target.read_text()) == {'prompt': 'Question', 'response': 'Answer'}
    with pytest.raises(FileExistsError):
        convert_data(source, target, 'csv')
    with pytest.raises(ValueError, match='byte cap'):
        convert_data(source, tmp_path / 'small.jsonl', 'csv', max_bytes=2)


def test_download_plan_pins_revision_and_enforces_cap_before_download(tmp_path, monkeypatch):
    calls = []
    info = SimpleNamespace(sha='a' * 40, siblings=[
        SimpleNamespace(rfilename='model.safetensors', size=20),
        SimpleNamespace(rfilename='config.json', size=10),
        SimpleNamespace(rfilename='unsafe.py', size=5)])
    monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=lambda *a, **kw: info),
        snapshot_download=lambda *a, **kw: calls.append(kw)))
    plan = download_model('example/model', tmp_path, max_bytes=30)
    assert plan['revision'] == 'a' * 40 and not calls
    assert 'unsafe.py' not in plan['files']
    with pytest.raises(ValueError, match='cap'):
        download_model('example/model', tmp_path, max_bytes=29, run=True)
    assert not calls


def test_aws_bundle_allowlists_code_and_declared_data(tmp_path):
    (tmp_path / 'src/ashayllm').mkdir(parents=True)
    (tmp_path / 'src/ashayllm/__init__.py').write_text('')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname="test"')
    (tmp_path / '.env').write_text('DO_NOT_UPLOAD=secret')
    (tmp_path / 'data.jsonl').write_text('{"text":"Example"}\n')
    config = RunConfig(data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'})
    (tmp_path / 'recipe.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    bundle = build_bundle(tmp_path, {'pipeline_config': 'recipe.yaml'})
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        assert '.env' not in archive.getnames()
        assert 'inputs/data.jsonl' in archive.getnames()
        recipe = yaml.safe_load(archive.extractfile('configs/job.yaml').read())
        assert recipe['data']['local_path'] == 'inputs/data.jsonl'
    with pytest.raises(ValueError, match='bundle'):
        build_bundle(tmp_path, {'pipeline_config': 'recipe.yaml', 'max_bundle_bytes': 10})
    config.model.custom_swiglu_path = 'kernels/swiglu'
    (tmp_path / 'recipe.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    with pytest.raises(ValueError, match='Custom CUDA'):
        build_bundle(tmp_path, {'pipeline_config': 'recipe.yaml'})


def test_training_guest_arms_shutdown_before_transfer_and_verifies_bundle():
    settings = {'artifact_bucket': 'example-bucket', 'instance_profile_name': 'example-profile',
                'max_runtime_minutes': 12}
    script = training_user_data(settings)
    assert script.index('systemd-run') < script.index('aws s3 cp')
    assert 'trap finish EXIT' in script and 'sha256sum -c -' in script
    assert '--sse AES256' in script and 'pipeline configs/job.yaml --run' in script
    assert len(script.encode()) < 16384
    with pytest.raises(ValueError, match='instance_profile_name'):
        training_user_data({'max_runtime_minutes': 12})


def test_stage_configuration_requires_parent_and_teacher():
    with pytest.raises(ValueError, match='parent_run'):
        RunConfig(stage={'kind': 'sft'})
    with pytest.raises(ValueError, match='teacher_run'):
        RunConfig(stage={'kind': 'distill', 'parent_run': 'parent'},
                  data={'provider': 'local_jsonl', 'local_path': 'input.jsonl'})
