import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from modelchef.aws_training import build_bundle
from modelchef.cli import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('command', ['chat', 'export'])
def test_missing_run_path_is_actionable_without_loading_torch(tmp_path, command):
    from modelchef.chat import chat, export_model
    with pytest.raises(ValueError, match='RUN_DIR is a placeholder'):
        (chat if command == 'chat' else export_model)(tmp_path / 'RUN_DIR')


@pytest.mark.parametrize('command', ['pipeline', 'cook'])
@pytest.mark.parametrize('execute', [False, True])
def test_cli_cloud_target_routes_recipe_without_local_execution(monkeypatch, capsys, command, execute):
    calls = []
    monkeypatch.setattr('modelchef.cli.cloud_job', lambda *args: calls.append(args) or
                        {'target': 'aws', 'test_passed': True})
    monkeypatch.setattr('sys.argv', ['modelchef', command, 'recipe.yaml', '--target', 'aws',
                                   '--aws-config', 'cloud.yaml', *(['--run'] if execute else [])])
    main()
    assert calls == [(Path('recipe.yaml'), Path('cloud.yaml'), execute, command, False)]
    assert json.loads(capsys.readouterr().out)['target'] == 'aws'


@pytest.mark.parametrize('args, message', [
    (['--target', 'aws'], '--aws-config'),
    (['--aws-config', 'cloud.yaml'], '--target aws'),
    (['--target', 'aws', '--aws-config', 'cloud.yaml', '--resume', 'checkpoint'], 'Cloud --resume'),
])
def test_cloud_options_fail_before_aws_calls(monkeypatch, capsys, args, message):
    monkeypatch.setattr('sys.argv', ['modelchef', 'pipeline', 'missing.yaml', *args])
    with pytest.raises(SystemExit):
        main()
    assert message in capsys.readouterr().err


def test_cloud_recipe_overrides_settings_and_preview_never_launches(tmp_path, monkeypatch):
    from modelchef.cli import cloud_job

    recipe = tmp_path / 'stage.yaml'
    recipe.write_text('{}')
    config = tmp_path / 'aws.yaml'
    config.write_text(yaml.safe_dump({'aws_test': {'pipeline_config': 'old.yaml', 'total_budget_usd': '10'}}))
    calls = []

    class Runner:
        def __init__(self, settings, root):
            calls.append(settings)

        def plan(self):
            return {'user_data': 'bootstrap', 'reserved': '0.50'}

        def run(self):
            raise AssertionError('Preview must not launch compute')

    monkeypatch.setattr('modelchef.aws_runner.SpotRunner', Runner)
    monkeypatch.setattr('modelchef.aws_training.build_bundle', lambda root, settings: b'bundle')
    result = cloud_job(recipe, config)
    assert calls[0]['pipeline_config'] == str(recipe.resolve())
    assert calls[0]['total_budget_usd'] == '10'
    assert result['target'] == 'aws'
    assert 'user_data' not in result


@pytest.mark.parametrize('answer, approved', [('yes', True), ('no', False)])
def test_budget_review_requires_explicit_confirmation(monkeypatch, capsys, answer, approved):
    from modelchef.cli import confirm_cloud

    monkeypatch.setattr('builtins.input', lambda: answer)
    plan = {'instance_type': 'g6.xlarge', 'spot_price_usd_hour': '0.40', 'reserved_usd': '8.00'}
    settings = {'max_runtime_minutes': 360, 'total_budget_usd': '10.00'}
    if approved:
        confirm_cloud(plan, settings)
    else:
        with pytest.raises(ValueError, match='cancelled'):
            confirm_cloud(plan, settings)
    assert 'budget reservation $8.00' in capsys.readouterr().err


def test_budget_review_handles_noninteractive_execution(monkeypatch):
    from modelchef.cli import confirm_cloud

    def no_input():
        raise EOFError

    monkeypatch.setattr('builtins.input', no_input)
    plan = {'instance_type': 'g6.xlarge', 'spot_price_usd_hour': '0.40', 'reserved_usd': '8.00'}
    settings = {'max_runtime_minutes': 360, 'total_budget_usd': '10.00'}
    with pytest.raises(ValueError, match='requires confirmation'):
        confirm_cloud(plan, settings)
    confirm_cloud(plan, settings, yes=True)


def test_declined_cloud_review_never_calls_runner_run(tmp_path, monkeypatch):
    from modelchef.cli import cloud_job

    recipe = tmp_path / 'stage.yaml'
    recipe.write_text('{}')
    config = tmp_path / 'aws.yaml'
    config.write_text(yaml.safe_dump({'aws_test': {'max_runtime_minutes': 360, 'total_budget_usd': '10'}}))

    class Runner:
        def __init__(self, *args):
            pass

        def plan(self):
            return {'instance_type': 'g6.xlarge', 'spot_price_usd_hour': '0.40', 'reserved_usd': '8.00'}

        def run(self):
            raise AssertionError('A declined review must not execute the billable job')

    monkeypatch.setattr('modelchef.aws_runner.SpotRunner', Runner)
    monkeypatch.setattr('modelchef.aws_training.build_bundle', lambda *args: b'bundle')
    monkeypatch.setattr('builtins.input', lambda: 'no')
    with pytest.raises(ValueError, match='cancelled'):
        cloud_job(recipe, config, execute=True)


def unpack_demo(tmp_path):
    bundle = build_bundle(ROOT, {'pipeline_config': 'configs/workflow-demo.yaml'})
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        for entry in archive.getmembers():
            path = tmp_path / entry.name
            assert path.resolve().is_relative_to(tmp_path.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(entry).read())
    return tmp_path / 'configs/job.yaml'


def test_workflow_bundle_relocates_each_stage_and_keeps_links(tmp_path):
    path = unpack_demo(tmp_path)
    job = yaml.safe_load(path.read_text())
    assert job['export']
    assert job['stages'][1]['parent_stage'] == job['stages'][0]['name']
    data_paths = []
    for stage in job['stages']:
        recipe = yaml.safe_load((tmp_path / stage['recipe']).read_text())
        data_paths.append(recipe['data']['local_path'])
        assert (tmp_path / recipe['data']['local_path']).is_file()
    assert len(set(data_paths)) == len(data_paths)
    with pytest.raises(ValueError, match='max_bundle_bytes'):
        build_bundle(ROOT, {'pipeline_config': 'configs/workflow-demo.yaml', 'max_bundle_bytes': 1})


def test_packaged_workflow_runs_to_export_and_preserves_lineage(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from modelchef.workflow import cook

    path = unpack_demo(tmp_path)
    monkeypatch.chdir(tmp_path)
    result = cook(path, tmp_path, True)
    assert result['status'] == 'completed'
    assert (Path(result['export_dir']) / 'model.safetensors').is_file()
    runs = list(result['stages'].values())
    config = json.loads((Path(runs[1]['run_dir']) / 'config.json').read_text())
    assert config['stage']['parent_run'] == runs[0]['run_dir']


def test_guest_launches_full_workflow_once(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace

    from modelchef.aws_guest import prepare_and_run

    path = unpack_demo(tmp_path)
    monkeypatch.chdir(tmp_path)
    calls = []
    monkeypatch.setattr('subprocess.run', lambda command, **kwargs: calls.append(command))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    prepare_and_run(path)
    assert calls[-1][2:] == ['modelchef.cli', 'cook', str(path), '--run']
    assert sum('cook' in command for command in calls) == 1


def test_workflow_local_cpu_preflight(tmp_path):
    pytest.importorskip('torch')
    from modelchef.aws_workflow import workflow_preflight

    # Use the real project root for the subprocess package and declared starter inputs.
    result = workflow_preflight(ROOT, {'pipeline_config': 'configs/workflow-demo.yaml'})
    assert result['passed']
    assert result['steps_per_stage'] == 2
