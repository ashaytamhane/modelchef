"""Regression checks for user-facing validation and the shipped offline workflow."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip('pydantic')

from modelchef.cli import main
from modelchef.run_config import RunConfig
from modelchef.workflow import cook


def test_recipe_check_and_concise_failure(tmp_path, monkeypatch, capsys):
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump({'training': {'steps': 2, 'warmup_steps': 0}}))
    monkeypatch.setattr(sys, 'argv', ['modelchef', 'check-config', str(path), '--quiet'])
    main()
    assert json.loads(capsys.readouterr().out)['training']['steps'] == 2
    path.write_text('training:\n  learning_rate: .inf\n')
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert 'learning_rate' in output and 'Traceback' not in output


def test_missing_dependency_hint_and_debug(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'tokenizers'", name='tokenizers')
    monkeypatch.setattr('modelchef.cli._settings', missing)
    monkeypatch.setattr(sys, 'argv', ['modelchef', 'aws-test', '--quiet'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "uv pip install -e '.[pipeline]'" in capsys.readouterr().err
    monkeypatch.setattr(sys, 'argv', ['modelchef', 'aws-test', '--quiet', '--debug'])
    with pytest.raises(ModuleNotFoundError):
        main()


def test_invalid_workflow_section_is_actionable(tmp_path):
    stage = tmp_path / 'stage.yaml'
    stage.write_text('training: null\n')
    workflow = tmp_path / 'workflow.yaml'
    workflow.write_text(yaml.safe_dump({'stages': [{'name': 'first', 'recipe': str(stage)}]}))
    with pytest.raises(ValueError, match='training must be a mapping'):
        cook(workflow, tmp_path)
    assert not (tmp_path / 'artifacts').exists()


@pytest.mark.parametrize('recipe', [
    {'tokenizer': {'algorithm': 'pretrained'}},
    {'model': {'backend': 'huggingface', 'pretrained_path': 'local'}},
])
def test_stored_metadata_is_valid_but_not_scratch_execution(recipe):
    metadata = RunConfig.model_validate(recipe)
    with pytest.raises(ValueError):
        metadata.validate_execution()


def test_tied_output_adapter_and_ignored_kernel_are_rejected():
    with pytest.raises(ValueError, match='tied output head'):
        RunConfig(model={'peft': {'method': 'lora', 'target_modules': ['head']}},
                  stage={'kind': 'midtrain', 'parent_run': 'parent'})
    with pytest.raises(ValueError, match='native dense'):
        RunConfig(model={'type': 'decoder_moe', 'custom_swiglu_path': 'kernels/swiglu'})


def test_standard_evaluation_uses_cpu_without_loading_torch(tmp_path, monkeypatch):
    from modelchef.comparison import standard_plan
    (tmp_path / 'config.json').write_text('{}')
    with pytest.raises(ValueError, match='weights'):
        standard_plan(tmp_path)
    (tmp_path / 'model.safetensors').write_bytes(b'local fixture')
    (tmp_path / 'tokenizer.json').write_text('{}')
    monkeypatch.setattr('modelchef.comparison.detect_hardware',
                        lambda: {'backend': 'cpu', 'bf16_supported': False})
    result = standard_plan(tmp_path)
    command = result['command']
    assert command[command.index('--device') + 1] == 'cpu'
    assert 'local_files_only=True' in command[command.index('--model_args') + 1]
    assert result['executed'] is False


def test_pipeline_timeout_is_not_cli_success(tmp_path, monkeypatch):
    path = tmp_path / 'recipe.yaml'
    path.write_text('{}')
    monkeypatch.setitem(sys.modules, 'modelchef.pipeline', SimpleNamespace(
        run_pipeline=lambda *a: {'status': 'timed-out', 'run_dir': str(tmp_path)}))
    monkeypatch.setattr(sys, 'argv', ['modelchef', 'pipeline', str(path), '--run', '--quiet'])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1


def test_offline_starter_workflow_trains_and_exports(tmp_path, monkeypatch, capsys):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tokenizers')
    from modelchef import benchmarks, corpus
    from modelchef.chat import chat

    def network_forbidden(*args, **kwargs):
        pytest.fail('The offline starter made a dataset network request')
    monkeypatch.setattr(corpus, 'urlopen', network_forbidden)
    monkeypatch.setattr(benchmarks, 'urlopen', network_forbidden)
    root = Path(__file__).resolve().parents[1]
    workflow = yaml.safe_load((root / 'configs/workflow-demo.yaml').read_text())
    workflow['output_dir'] = str(tmp_path / 'workflows')
    for stage in workflow['stages']:
        recipe = yaml.safe_load((root / stage['recipe']).read_text())
        recipe['output_dir'] = str(tmp_path / 'training')
        recipe.setdefault('tokenizer', {})['packing'] = (
            'document' if recipe.get('stage', {}).get('kind', 'pretrain') == 'pretrain'
            else 'concatenate')
        if recipe['tokenizer']['packing'] == 'document':
            recipe['data']['storage'] = 'disk'
            recipe.setdefault('model', {})['context_length'] = 1024
        target = tmp_path / (stage['name'] + '.yaml')
        target.write_text(yaml.safe_dump(recipe))
        stage['recipe'] = str(target)
    path = tmp_path / 'workflow.yaml'
    path.write_text(yaml.safe_dump(workflow))
    torch.set_num_threads(1)
    result = cook(path, root, execute=True)
    assert result['status'] == 'completed'
    assert all(stage['checkpoint_recovery'] == 'passed' for stage in result['stages'].values())
    assert (Path(result['export_dir']) / 'model.safetensors').is_file()
    final_config = json.loads((Path(result['final_run']) / 'config.json').read_text())
    assert final_config['tokenizer']['packing'] == 'concatenate'
    chat(Path(result['final_run']), prompt='A model', max_tokens=2)
    assert 'Loaded checkpoint' in capsys.readouterr().out


def test_intel_mac_setup_message_preserves_existing_runtime(monkeypatch):
    from modelchef.hardware import resolve_runtime, setup_plan
    monkeypatch.setattr('modelchef.hardware.platform.system', lambda: 'Darwin')
    monkeypatch.setattr('modelchef.hardware.platform.machine', lambda: 'x86_64')
    with pytest.raises(ValueError, match='Intel Macs'):
        setup_plan()
    assert resolve_runtime({'accelerator': 'cpu'},
                           {'backend': 'cpu', 'bf16_supported': False})['device'] == 'cpu'


def test_provenance_fingerprints_code_without_environment_secrets(tmp_path, monkeypatch):
    from modelchef import provenance
    source = tmp_path / 'module.py'
    source.write_text('version = 1')
    monkeypatch.setattr(provenance, '__file__', str(source))
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'private-test-sentinel')
    first = provenance.environment_provenance(tmp_path)
    source.write_text('version = 2')
    second = provenance.environment_provenance(tmp_path)
    assert first['code']['source_sha256'] != second['code']['source_sha256']
    assert first['code']['git_revision'] is None
    assert 'private-test-sentinel' not in json.dumps(first)
