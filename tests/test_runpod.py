import io
import json
import tarfile
from pathlib import Path

import pytest
import yaml

from modelchef.cli import main, runpod_job
from modelchef.runpod_runner import (
    RunpodAPI,
    RunpodConfig,
    RunpodRunner,
    extract_results,
    find_owned_pods,
)


def settings():
    return {'gpu_type_id': 'NVIDIA GeForce RTX 4090', 'network_volume_id': 'volume',
            'max_compute_hourly_usd': 1, 'storage_hourly_usd': 0.05,
            'safety_reserve_usd': 0.1, 'per_run_budget_usd': 1, 'total_budget_usd': 2,
            'max_runtime_minutes': 30, 'local_preflight': False}


def test_settlement_keeps_conservative_spend_and_requires_verified_cleanup(tmp_path):
    from decimal import Decimal

    from modelchef.aws_runner import Ledger
    ledger = Ledger(tmp_path / 'budget.sqlite')
    ledger.reserve('owned', 'runpod:test', Decimal(2), 3)
    report = {'termination_confirmed': True, 'cleanup_verified_by_read': True,
              'elapsed_seconds': 3600, 'estimated_spend_bound_usd': '1.2',
              'settings': {'max_compute_hourly_usd': 1, 'storage_hourly_usd': .1,
                           'safety_reserve_usd': .1}}
    with pytest.raises(ValueError, match='Settlement'):
        ledger.settle_terminated('owned', {**report, 'cleanup_verified_by_read': False})
    with pytest.raises(ValueError, match='Settlement'):
        ledger.settle_terminated('owned', {**report, 'estimated_spend_bound_usd': '0'})
    ledger.settle_terminated('owned', report)
    assert ledger.db.execute('SELECT reserved,state FROM runs').fetchone() == ('1.2', 'settled')
    ledger.reserve('next', 'runpod:test', Decimal('1.8'), 3)
    ledger.db.close()


@pytest.mark.parametrize('field,value', [('max_compute_hourly_usd', float('nan')),
    ('max_runtime_minutes', 0), ('gpu_count', 0), ('per_run_budget_usd', 0.1),
    ('total_budget_usd', 0.5), ('api_key', 'never-in-recipes')])
def test_invalid_budget_and_embedded_secrets_rejected(field, value):
    with pytest.raises(ValueError):
        RunpodConfig.model_validate({**settings(), field: value})


def test_preview_needs_no_key_and_packages_external_recipe(tmp_path, monkeypatch):
    monkeypatch.delenv('RUNPOD_API_KEY', raising=False)
    recipe = tmp_path/'stage.yaml'
    recipe.write_text('{}')
    runner = RunpodRunner(settings(), tmp_path, recipe)
    plan = runner.plan()
    assert plan['target'] == 'runpod'
    assert plan['bundle_bytes'] > 0
    assert not (tmp_path/'artifacts').exists()
    with pytest.raises(ValueError, match='RUNPOD_API_KEY'):
        RunpodAPI(runner.settings)


@pytest.mark.parametrize('command', ['pipeline', 'cook'])
def test_cli_routes_runpod_without_local_or_aws_execution(command, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr('modelchef.cli.runpod_job', lambda *args: calls.append(args) or {'target': 'runpod'})
    monkeypatch.setattr('sys.argv', ['modelchef', command, 'stage.yaml', '--target', 'runpod',
                                   '--runpod-config', 'host.yaml'])
    main()
    assert calls == [(Path('stage.yaml'), Path('host.yaml'), False, command, False)]
    assert json.loads(capsys.readouterr().out)['target'] == 'runpod'


def test_declined_launch_creates_nothing(tmp_path, monkeypatch):
    recipe, host = tmp_path/'stage.yaml', tmp_path/'host.yaml'
    recipe.write_text('{}')
    host.write_text(yaml.safe_dump({'runpod': settings()}))
    monkeypatch.setattr('builtins.input', lambda _: 'no')
    monkeypatch.setattr(RunpodRunner, 'run', lambda _: pytest.fail('Declined job launched'))
    with pytest.raises(ValueError, match='confirmation'):
        runpod_job(recipe, host, execute=True)


def test_pod_payload_does_not_include_control_plane_key(tmp_path, monkeypatch):
    monkeypatch.setenv('RUNPOD_API_KEY', 'sentinel-private-key')
    runner = RunpodRunner(settings(), tmp_path, tmp_path/'stage.yaml')
    payload = runner.payload('modelchef-test', 'ssh-ed25519 AAAA')
    assert 'sentinel-private-key' not in json.dumps(payload)
    assert payload['ports'] == ['22/tcp']
    assert payload['mounts']['network'][0]['volumeId'] == 'volume'
    assert payload['gpu']['minCudaVersion'] == '13.0'
    bootstrap = payload['cmd'][0]
    assert bootstrap.index('ssh-keygen -A') < bootstrap.index('exec /usr/sbin/sshd')


def test_reconciliation_reads_all_pages():
    class API:
        def request(self, method, path):
            if '?' not in path:
                return {'pods': [], 'pagination': {'hasNextPage': True, 'nextCursor': 'next'}}
            return {'pods': [{'name': 'owned', 'id': 'p'}], 'pagination': {'hasNextPage': False}}
    assert find_owned_pods(API(), 'owned') == [{'name': 'owned', 'id': 'p'}]


def test_reuses_cli_credentials_with_environment_override(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.delenv('RUNPOD_API_KEY', raising=False)
    directory = tmp_path/'.runpod'
    directory.mkdir()
    (directory/'config.toml').write_text('apikey = "saved-credential"\n')
    assert RunpodAPI(RunpodConfig.model_validate(settings())).key == 'saved-credential'
    monkeypatch.setenv('RUNPOD_API_KEY', 'environment-credential')
    assert RunpodAPI(RunpodConfig.model_validate(settings())).key == 'environment-credential'
    monkeypatch.setenv('RUNPOD_API_KEY', 'malformed\nsecret')
    with pytest.raises(ValueError, match='Invalid Runpod API key format') as error:
        RunpodAPI(RunpodConfig.model_validate(settings()))
    assert 'secret' not in str(error.value)


def test_result_extraction_rejects_traversal_and_links(tmp_path):
    for name, kind in [('../outside', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)]:
        archive = tmp_path/(kind.decode()+'.tar.gz')
        with tarfile.open(archive, 'w:gz') as output:
            info = tarfile.TarInfo(name)
            info.type = kind
            info.size = 1 if kind == tarfile.REGTYPE else 0
            info.linkname = '../outside'
            output.addfile(info, io.BytesIO(b'x') if info.size else None)
        with pytest.raises(ValueError, match='Unsafe'):
            extract_results(archive, tmp_path/'results', 1000)
    assert not (tmp_path/'results').exists()


@pytest.mark.parametrize('failure', [None, 'download', 'guest', 'ambiguous', 'cleanup'])
def test_lifecycle_always_attempts_cleanup_and_preserves_recovery(tmp_path, monkeypatch, failure):
    private, public = tmp_path/'key', tmp_path/'key.pub'
    private.write_text('private-not-uploaded')
    public.write_text('ssh-ed25519 AAAA')
    configuration = {**settings(), 'ssh_private_key': str(private), 'ssh_public_key': str(public)}
    calls = []

    class API:
        def __init__(self):
            self.pod = {'id': 'owned', 'status': 'RUNNING', 'cost': 0.2,
                        'ssh': {'direct': {'host': '127.0.0.1', 'port': 22, 'username': 'root'}}}

        def request(self, method, path, payload=None):
            calls.append((method, path))
            if '/catalog/' in path:
                return {'price': {'secure': 0.2}}
            if method == 'POST':
                self.pod['name'] = payload['name']
                if failure == 'ambiguous':
                    raise TimeoutError('Outcome unknown')
            if path == '/pods' and method == 'GET':
                return {'pods': [self.pod]}
            if method == 'DELETE' and failure == 'cleanup':
                raise TimeoutError('Cleanup unknown')
            return self.pod

    runner = RunpodRunner(configuration, tmp_path, tmp_path/'recipe.yaml', api=API())
    monkeypatch.setattr(runner, 'bundle', lambda: b'allowlisted-source')
    monkeypatch.setattr('modelchef.runpod_runner.shutil.which', lambda _: 'tool')
    ssh_commands = []
    def ssh(*args, **kw):
        ssh_commands.append(args[2])
        return (
        json.dumps({'state': 'failed' if failure == 'guest' else 'completed'})
        if args[2].startswith('cat ') else '')
    monkeypatch.setattr(runner, 'ssh', ssh)
    monkeypatch.setattr(runner, 'copy', lambda *args, **kw: None)

    def fetch(*args):
        if failure == 'download':
            raise OSError('Disk full')
    monkeypatch.setattr(runner, 'fetch', fetch)
    report = runner.run()
    launch = next(command for command in ssh_commands if 'nohup' in command)
    assert '&& (nohup ' in launch and launch.endswith('&)')
    assert ('DELETE', '/pods/owned') in calls
    assert report['termination_confirmed'] == (failure != 'cleanup')
    assert report['test_passed'] == (failure not in {'download', 'guest'})
    assert report['network_volume_id'] == 'volume'
    assert 'private-not-uploaded' not in json.dumps(report)
    assert len([call for call in calls if call[0] == 'POST']) == 1


def test_guest_deadline_preserves_outputs_and_stops_child_group(tmp_path, monkeypatch):
    import subprocess

    from modelchef import runpod_guest
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv('MODELCHEF_LOCAL_CHECKPOINT_KEEP', '2')
    monkeypatch.setenv('MODELCHEF_RECOVERY_S3_URI', 'unused-test-value')
    (tmp_path/'artifacts').mkdir()
    checkpoint = tmp_path/'artifacts/saved-checkpoint'
    checkpoint.write_text('preserve-me')
    killed = []

    class Process:
        pid = 123
        waits = 0

        def wait(self, timeout=None):
            self.waits += 1
            if self.waits == 1:
                raise subprocess.TimeoutExpired('guest-command', timeout)
            return -15
    monkeypatch.setattr(runpod_guest.subprocess, 'Popen', lambda *a, **kw: Process())
    monkeypatch.setattr(runpod_guest.os, 'killpg', lambda pid, signal: killed.append(pid), raising=False)
    with pytest.raises(TimeoutError):
        runpod_guest.execute(30)
    assert killed == [123]
    assert json.loads((tmp_path/'status.json').read_text())['state'] == 'failed'
    assert checkpoint.read_text() == 'preserve-me'


def test_shared_guest_dependency_policy_is_provider_independent():
    from modelchef.aws_guest import guest_plan
    from modelchef.run_config import RunConfig
    config = RunConfig(model={'custom_swiglu_path': 'kernels/example'})
    plan = guest_plan(config, cloud_extra=None)
    assert 'aws' not in plan['extras']
    assert 'cuda13-toolchain' in plan['extras']
    assert 'pipeline' in plan['extras']


def test_guest_system_python_creates_venv_before_package_install(tmp_path, monkeypatch):
    from modelchef import runpod_guest
    monkeypatch.setattr(runpod_guest.sys, 'prefix', '/system')
    monkeypatch.setattr(runpod_guest.sys, 'base_prefix', '/system')
    commands, python = runpod_guest.runtime_commands(tmp_path / 'runtime')
    assert commands[0][1:5] == ['-m', 'venv', '--without-pip', '--system-site-packages']
    assert str(tmp_path / 'runtime') in python
    assert python != runpod_guest.sys.executable
    monkeypatch.setattr(runpod_guest.sys, 'prefix', '/isolated-venv')
    assert runpod_guest.runtime_commands(tmp_path) == ([], runpod_guest.sys.executable)


def test_result_transfer_contains_training_data_and_tokenizer(tmp_path, monkeypatch):
    names = ['raw.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl',
             'tokens.pt', 'tokens-train.bin', 'tokens-validation.bin', 'tokens-test.bin',
             'tokenizer.json', 'config.json', 'checkpoints/step-000001.pt']
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as archive:
        for name in names:
            content = ('saved-' + name).encode()
            entry = tarfile.TarInfo('artifacts/training/run/' + name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    payload = buffer.getvalue()
    runner = RunpodRunner(settings(), tmp_path, tmp_path / 'recipe.yaml')
    monkeypatch.setattr(runner, 'ssh', lambda *a, **kw: str(len(payload)))
    monkeypatch.setattr(runner, 'copy', lambda pod, directory, local, remote, **kw:
                        Path(local).write_bytes(payload))
    runner.fetch({}, tmp_path, '/workspace/owned-run')
    for name in names:
        restored = tmp_path / 'results/artifacts/training/run' / name
        assert restored.read_bytes() == ('saved-' + name).encode()
