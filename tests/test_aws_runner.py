from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from modelchef.aws_runner import Ledger, SpotRunner, budget_reservation, user_data


@pytest.fixture
def settings():
    return yaml.safe_load((Path(__file__).parents[1] / "configs/aws-test.yaml").read_text())[
        "aws_test"]


def test_budget_reservation(settings):
    assert budget_reservation(settings) == Decimal("0.50")


def test_reconcile_old_unaccepted_launch_releases_only_after_confirmation(settings, tmp_path):
    from datetime import UTC, datetime, timedelta
    from types import SimpleNamespace

    ledger = Ledger(tmp_path / settings['ledger_path'])
    ledger.reserve('absent-run', 'test-account:us-east-1', Decimal('0.50'), Decimal(1))
    report = {'started_at': (datetime.now(UTC) - timedelta(minutes=20)).isoformat()}
    ledger.record('absent-run', 'cleanup-pending', report)
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, 'test-account'
    deleted = []
    runner.ec2 = SimpleNamespace(describe_instances=lambda **kw: {'Reservations': []},
        describe_security_groups=lambda **kw: {'SecurityGroups': [{'GroupId': 'sg-old'}]},
        delete_security_group=lambda **kw: deleted.append(kw['GroupId']))
    result = runner.reconcile('absent-run')
    assert result['launch_error_code'] == 'ConfirmedNoInstance'
    assert deleted == ['sg-old']
    assert ledger.db.execute('SELECT reserved,state FROM runs').fetchone() == ('0', 'no-instance')


def test_on_demand_budget_uses_declared_rate(settings):
    settings.update(market_type='on_demand', on_demand_hourly_usd='1.25',
                    per_run_budget_usd='0.40', total_budget_usd='1.00')
    with pytest.raises(ValueError, match='budgets'):
        budget_reservation(settings)
    settings['per_run_budget_usd'] = settings['total_budget_usd'] = '5.00'
    assert budget_reservation(settings) == Decimal('5.00')


@pytest.mark.parametrize("with_recipe", [False, True])
def test_on_demand_plan_does_not_require_spot_offering(settings, with_recipe):
    from types import SimpleNamespace

    runner = SpotRunner.__new__(SpotRunner)
    runner.root = Path(__file__).parents[1]
    runner.settings = {**settings, 'market_type': 'on_demand',
                       'on_demand_hourly_usd': '1.25', 'ami_id': 'ami-test',
                       'subnet_id': 'subnet-test'}
    if with_recipe:
        runner.settings.update(pipeline_config='configs/pretrain-demo.yaml',
                               artifact_bucket='test-bucket', instance_profile_name='test-profile',
                               per_run_budget_usd=10, total_budget_usd=10)
    runner.account = '123456789012'
    runner.ec2 = SimpleNamespace(
        describe_instance_types=lambda **kw: {'InstanceTypes': [{'InstanceType': settings['instance_type'],
            'GpuInfo': {'Gpus': [{'Count': 4, 'MemoryInfo': {'SizeInMiB': 24576}}]}}]},
        describe_subnets=lambda **kw: {'Subnets': [{'SubnetId': 'subnet-test',
            'VpcId': 'vpc-test', 'AvailabilityZone': 'us-east-1a', 'State': 'available'}]},
        describe_images=lambda **kw: {'Images': [{'Name': 'GPU AMI', 'Architecture': 'x86_64',
            'RootDeviceName': '/dev/xvda', 'BlockDeviceMappings': [{'DeviceName': '/dev/xvda',
            'Ebs': {'VolumeSize': 100}}]}]},
        get_paginator=lambda _: pytest.fail('On-Demand plan must not query Spot prices'))
    plan = runner.plan()
    assert plan['market_type'] == 'on_demand'
    assert plan['spot_price_usd_hour'] is None
    assert plan['hourly_budget_rate_usd'] == '1.25'
    if with_recipe:
        assert plan['memory_plan']['gpu_count'] == 4
        assert plan['memory_plan']['memory_bytes_per_gpu'] == 24576 * 1024**2
        assert plan['memory_plan']['capacity_verified']


def test_result_download_error_preserves_success_and_shutdown_evidence(tmp_path):
    import json
    from types import SimpleNamespace

    from botocore.exceptions import ClientError

    def missing_bucket(**kwargs):
        raise ClientError({'Error': {'Code': 'NoSuchBucket', 'Message': 'Deleted bucket'}},
                          'GetObject')
    runner = SpotRunner.__new__(SpotRunner)
    runner.settings = {}
    runner.s3 = SimpleNamespace(get_object=missing_bucket,
                               exceptions=SimpleNamespace(NoSuchKey=type('NoSuchKey', (Exception,), {})))
    report = {'artifact_bucket': 'deleted', 'artifact_prefix': 'run',
              'terminated': True, 'test_passed': True}
    with pytest.raises(RuntimeError, match='result retrieval failed'):
        runner.fetch_training_results(report, tmp_path)
    saved = json.loads((tmp_path / 'report.json').read_text())
    assert saved['terminated'] and saved['test_passed']
    assert 'NoSuchBucket' in saved['results_error']


@pytest.mark.parametrize("key,value", [
    ("total_budget_usd", "0.10"), ("per_run_budget_usd", "0.01"),
    ("max_runtime_minutes", -1), ("max_runtime_minutes", 0.5),
    ("max_spot_hourly_usd", "NaN"), ("instance_type", "p5.48xlarge"),
])
def test_reject_unsafe_budget(settings, key, value):
    settings[key] = value
    with pytest.raises(ValueError):
        budget_reservation(settings)


def test_total_budget_survives_new_ledger_connection(tmp_path):
    path = tmp_path / "ledger.sqlite"
    ledger = Ledger(path)
    ledger.reserve("one", "account:region", Decimal("0.5"), "1")
    Ledger(path).reserve("two", "account:region", Decimal("0.5"), "1")
    with pytest.raises(ValueError, match="Overall budget"):
        Ledger(path).reserve("three", "account:region", Decimal("0.5"), "1")


def test_unlaunched_budget_is_released_only_with_confirmed_no_instance(tmp_path):
    ledger = Ledger(tmp_path / 'ledger.sqlite')
    ledger.reserve('one', 'account:region', Decimal('0.5'), '0.5')
    with pytest.raises(ValueError, match='unconfirmed launch'):
        ledger.release_unlaunched('one')
    ledger.record('one', 'no-instance', {'run_id': 'one',
                                        'launch_error_code': 'InsufficientInstanceCapacity'})
    ledger.release_unlaunched('one')
    assert ledger.db.execute("SELECT reserved FROM runs WHERE id='one'").fetchone()[0] == '0'
    ledger.reserve('two', 'account:region', Decimal('0.5'), '0.5')
    ledger.record('two', 'no-instance', {'instance_id': 'i-accepted',
                                        'launch_error_code': 'InsufficientInstanceCapacity'})
    with pytest.raises(ValueError, match='unconfirmed launch'):
        ledger.release_unlaunched('two')
    ledger.reserve('three', 'account:region', Decimal('0.5'), '1.0')
    ledger.record('three', 'no-instance', {'launch_error_code': 'VcpuLimitExceeded'})
    ledger.release_unlaunched('three')
    assert ledger.db.execute("SELECT reserved FROM runs WHERE id='three'").fetchone()[0] == '0'
    ledger.reserve('quota', 'account:region', Decimal('0.5'), '1.0')
    ledger.record('quota', 'no-instance', {'launch_error_code': 'MaxSpotInstanceCountExceeded'})
    ledger.release_unlaunched('quota')
    assert ledger.db.execute("SELECT reserved FROM runs WHERE id='quota'").fetchone()[0] == '0'


def test_userdata_contains_guest_deadline_and_termination(settings):
    root = Path(__file__).parents[1]
    script = user_data(root, settings)
    assert "trap finish EXIT" in script
    assert "/sbin/shutdown -h now" in script
    assert "--on-active=12m" in script
    assert len(script.encode()) <= 16384
    assert "vllm==" not in script
    assert "set +u\nsource /opt/pytorch/bin/activate\nset -u" in script


@pytest.mark.parametrize("scheduler_fails", [False, True])
@pytest.mark.parametrize("waiter_fails", [False, True])
@pytest.mark.parametrize('console_fails', [False, True])
def test_run_terminates_after_test_or_watchdog_failure(
        settings, tmp_path, scheduler_fails, waiter_fails, console_fails, monkeypatch):
    from types import SimpleNamespace

    from botocore.exceptions import EndpointConnectionError, WaiterError

    calls = []
    monkeypatch.setattr("modelchef.aws_runner.local_preflight", lambda _: {"passed": True})
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, "test-account"
    runner.plan = lambda: {"user_data": "test", "vpc_id": "vpc-test", "ami_id": "ami-test",
                           "subnet_id": "subnet-test", "root_device": "/dev/xvda"}
    runner.prepare_deadline_role = lambda: "role-test"
    def create_schedule(**kwargs):
        if scheduler_fails:
            raise RuntimeError("schedule failed")
    def wait(**kwargs):
        calls.append("wait-terminated")
        if waiter_fails and calls.count("wait-terminated") == 1:
            raise WaiterError(name="instance_terminated", reason="timeout", last_response={})
    def console(**kwargs):
        if console_fails:
            raise EndpointConnectionError(endpoint_url='https://ec2.example.invalid')
        return {'Output': 'MODELCHEF_RESULT=0'}
    runner.scheduler = SimpleNamespace(create_schedule=create_schedule,
        delete_schedule=lambda **kw: calls.append("delete-schedule"))
    runner.ec2 = SimpleNamespace(
        create_security_group=lambda **kw: {"GroupId": "sg-test"},
        run_instances=lambda **kw: {"Instances": [{"InstanceId": "i-test"}]},
        describe_instances=lambda **kw: {"Reservations": [{"Instances": [
            {"InstanceId": "i-test", "State": {"Name": "terminated"}}]}]},
        terminate_instances=lambda **kw: calls.append("force" if kw.get("Force") else "terminate"),
        get_waiter=lambda _: SimpleNamespace(wait=wait),
        get_console_output=console,
        delete_security_group=lambda **kw: calls.append("delete-sg"))
    if scheduler_fails:
        with pytest.raises(RuntimeError, match="schedule failed"):
            runner.run()
        assert calls[:2] == ["terminate", "wait-terminated"]
        if waiter_fails:
            assert "force" in calls
    else:
        assert runner.run()["terminated"]
    assert "delete-sg" in calls


def test_run_preserves_console_before_ec2_discards_it(settings, tmp_path, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr('modelchef.aws_runner.local_preflight', lambda _: {'passed': True})
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, 'test-account'
    runner.plan = lambda: {'user_data': 'test', 'vpc_id': 'vpc-test', 'ami_id': 'ami-test',
                           'subnet_id': 'subnet-test', 'root_device': '/dev/xvda'}
    runner.prepare_deadline_role = lambda: 'role-test'
    outputs = iter(['MODELCHEF_RESULT=1: unsupported Python', ''])
    runner.ec2 = SimpleNamespace(
        create_security_group=lambda **kw: {'GroupId': 'sg-test'},
        run_instances=lambda **kw: {'Instances': [{'InstanceId': 'i-test'}]},
        describe_instances=lambda **kw: {'Reservations': [{'Instances': [
            {'InstanceId': 'i-test', 'State': {'Name': 'shutting-down'}}]}]},
        get_console_output=lambda **kw: {'Output': next(outputs)},
        terminate_instances=lambda **kw: None,
        get_waiter=lambda _: SimpleNamespace(wait=lambda **kw: None),
        delete_security_group=lambda **kw: None)
    runner.scheduler = SimpleNamespace(create_schedule=lambda **kw: None,
                                       delete_schedule=lambda **kw: None)
    report = runner.run()
    assert report['terminated'] is True
    assert report['test_passed'] is False
    assert report['console_output'] == 'MODELCHEF_RESULT=1: unsupported Python'


def test_lost_launch_response_remains_recoverable_when_cleanup_is_offline(settings, tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    monkeypatch.setattr('modelchef.aws_runner.local_preflight', lambda _: {'passed': True})
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, 'test-account'
    runner.plan = lambda: {'user_data': 'test', 'vpc_id': 'vpc-test', 'ami_id': 'ami-test',
                          'subnet_id': 'subnet-test', 'root_device': '/dev/xvda'}
    runner.prepare_deadline_role = lambda: 'role-test'
    def offline(**kwargs):
        raise RuntimeError('Network unavailable')
    runner.ec2 = SimpleNamespace(create_security_group=lambda **kw: {'GroupId': 'sg-test'},
                                 run_instances=offline, describe_instances=offline)
    with pytest.raises(RuntimeError, match='Network unavailable'):
        runner.run()
    path = next((tmp_path / 'artifacts/aws-runs').glob('*/report.json'))
    report = json.loads(path.read_text())
    assert report['security_group_id'] == 'sg-test'
    assert report['cleanup_error'] == 'Network unavailable'
    run_id = report['run_id']
    recovered_calls = []
    def recovered(**kwargs):
        recovered_calls.append(kwargs)
        return {'Reservations': [{'Instances': [{'InstanceId': 'i-accepted',
                                                 'State': {'Name': 'running'}}]}]}
    runner.ec2 = SimpleNamespace(describe_instances=recovered,
                                 get_console_output=lambda **kw: {'Output': ''})
    result = runner.reconcile(run_id)
    assert result['instance_id'] == 'i-accepted' and not result['terminated']
    assert {'Name': 'client-token', 'Values': [run_id]} in recovered_calls[0]['Filters']
    ledger = Ledger(tmp_path / settings['ledger_path'])
    assert ledger.db.execute('SELECT state FROM runs WHERE id=?', (run_id,)).fetchone()[0] == 'cleanup-pending'


def test_failed_local_preflight_prevents_cloud_plan(settings, tmp_path, monkeypatch):
    from types import SimpleNamespace

    from modelchef.aws_runner import local_preflight

    monkeypatch.setattr("modelchef.aws_runner.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="test failed", stderr=""))
    runner = SpotRunner.__new__(SpotRunner)
    runner.settings, runner.root = settings, tmp_path
    runner.plan = lambda: pytest.fail("AWS plan must not run after failed local tests")
    with pytest.raises(RuntimeError, match="no instance launched"):
        runner.run()
    assert '"passed": false' in (tmp_path / "artifacts/aws-preflight.json").read_text()
    with pytest.raises(RuntimeError):
        local_preflight(tmp_path)


@pytest.mark.parametrize('download', [False, True])
def test_training_uploads_bundle_and_terminates_before_result_fetch(settings, tmp_path, monkeypatch, download):
    import io
    from types import SimpleNamespace

    from modelchef.aws_runner import SpotRunner

    calls = []
    settings.update(pipeline_config='job.yaml', artifact_bucket='test-bucket',
                    instance_profile_name='test-profile', download_results=download)
    monkeypatch.setattr('modelchef.aws_runner.local_preflight', lambda _: {'passed': True})
    monkeypatch.setattr('modelchef.aws_training.build_bundle', lambda *a: b'allowed-source')
    monkeypatch.setattr('modelchef.aws_training.training_preflight', lambda *a: {'passed': True})
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, 'test-account'
    runner.plan = lambda: {'user_data': 'preview', 'vpc_id': 'vpc-test', 'ami_id': 'ami-test',
                           'subnet_id': 'subnet-test', 'root_device': '/dev/xvda'}
    runner.prepare_deadline_role = lambda: 'role-test'
    runner.s3 = SimpleNamespace(
        put_object=lambda **kw: calls.append(('upload', kw)),
        get_object=lambda **kw: (calls.append(('download', kw)) or {'Body': io.BytesIO(b'results')}))
    def launch(**kw):
        calls.append(('launch', kw))
        return {'Instances': [{'InstanceId': 'i-test'}]}
    runner.ec2 = SimpleNamespace(
        create_security_group=lambda **kw: {'GroupId': 'sg-test'}, run_instances=launch,
        describe_instances=lambda **kw: {'Reservations': [{'Instances': [
            {'InstanceId': 'i-test', 'State': {'Name': 'terminated'}}]}]},
        get_console_output=lambda **kw: {'Output': 'MODELCHEF_RESULT=0'},
        delete_security_group=lambda **kw: calls.append(('cleanup', kw)))
    runner.scheduler = SimpleNamespace(create_schedule=lambda **kw: None,
                                      delete_schedule=lambda **kw: None)
    result = runner.run()
    assert result['terminated'] and result['test_passed']
    kinds = [kind for kind, _ in calls]
    assert kinds.index('upload') < kinds.index('launch') < kinds.index('cleanup')
    assert ('download' in kinds) == download
    if download:
        assert kinds.index('cleanup') < kinds.index('download')
    arguments = next(value for kind, value in calls if kind == 'launch')
    assert arguments['InstanceMarketOptions']['MarketType'] == 'spot'
    assert arguments['IamInstanceProfile'] == {'Name': 'test-profile'}
    assert arguments['BlockDeviceMappings'][0]['Ebs']['DeleteOnTermination'] is False
    if download:
        assert Path(result['results_path']).read_bytes() == b'results'


@pytest.mark.parametrize("state", ["shutting-down", "terminated"])
@pytest.mark.parametrize("console", ["MODELCHEF_RESULT=0", "", None])
def test_reconcile_preserves_watchdog_until_terminated(settings, tmp_path, state, console):
    from types import SimpleNamespace

    calls = []
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, "test-account"
    ledger = Ledger(tmp_path / settings["ledger_path"])
    ledger.reserve("run-test", "test-account:us-east-1", Decimal("0.5"), "10")
    ledger.record("run-test", "launched", {"instance_id": "i-test", "console_output": "MODELCHEF_RESULT=0"})
    runner.ec2 = SimpleNamespace(
        describe_instances=lambda **kw: {"Reservations": [{"Instances": [
            {"State": {"Name": state}}]}]},
        get_console_output=lambda **kw: {"Output": console},
        describe_security_groups=lambda **kw: {"SecurityGroups": [{"GroupId": "sg-test"}]},
        delete_security_group=lambda **kw: calls.append("delete-sg"))
    runner.scheduler = SimpleNamespace(delete_schedule=lambda **kw: calls.append("delete-schedule"))
    result = runner.reconcile("run-test")
    assert result["test_passed"]
    assert result["terminated"] == (state == "terminated")
    assert calls == (["delete-schedule", "delete-sg"] if state == "terminated" else [])


@pytest.fixture
def training_source(tmp_path):
    from modelchef.run_config import RunConfig
    (tmp_path / 'src/modelchef').mkdir(parents=True)
    (tmp_path / 'src/modelchef/__init__.py').write_text('')
    (tmp_path / 'pyproject.toml').write_text('[project]\nname="example"\nreadme="README.md"')
    (tmp_path / 'README.md').write_text('ModelChef tutorial')
    (tmp_path / 'data.jsonl').write_text('{"text":"Example"}\n')
    config = RunConfig(data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'})
    (tmp_path / 'job.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    return tmp_path, config


@pytest.mark.parametrize('feature', ['ddp', 'fsdp2', 'nvfp4'])
def test_managed_training_preserves_gpu_optimization_policy(training_source, feature):
    from modelchef.aws_training import build_bundle
    root, config = training_source
    if feature in {'ddp', 'fsdp2'}:
        config.training.distributed = feature
    else:
        config.model.quantization = 'nvfp4'
    (root / 'job.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    import io
    import tarfile

    from modelchef.aws_guest import guest_plan
    with tarfile.open(fileobj=io.BytesIO(build_bundle(root, {'pipeline_config': 'job.yaml'}))) as archive:
        loaded = yaml.safe_load(archive.extractfile('configs/job.yaml').read())
    assert loaded['model']['quantization'] == config.model.quantization
    assert loaded['training']['distributed'] == config.training.distributed
    plan = guest_plan(config)
    if feature in {'ddp', 'fsdp2'}:
        assert '--nproc-per-node=gpu' in plan['command']
        assert '--' in plan['command']
    else:
        assert 'nvfp4' in plan['extras']


def test_training_bundle_includes_build_readme_and_bounds_single_input_line(training_source):
    import io
    import tarfile

    from modelchef.aws_training import build_bundle
    root, config = training_source
    with tarfile.open(fileobj=io.BytesIO(build_bundle(root, {'pipeline_config': 'job.yaml'}))) as archive:
        assert archive.extractfile('README.md').read() == b'ModelChef tutorial'
    config.data.max_download_bytes = 64
    (root / 'job.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    (root / 'data.jsonl').write_text('x' * 1024)
    with pytest.raises(ValueError, match='byte budget'):
        build_bundle(root, {'pipeline_config': 'job.yaml'})


def test_training_preflight_requires_completed_report_even_if_process_exits_zero(training_source, monkeypatch):
    import json

    from modelchef.aws_training import training_preflight
    root, _ = training_source
    def fake_pipeline(*args, **kwargs):
        config = yaml.safe_load((root / 'artifacts/aws-training-preflight.yaml').read_text())
        folder = root / config['output_dir'] / 'aws-local-preflight-test'
        folder.mkdir(parents=True)
        (folder / 'report.json').write_text(json.dumps({'status': 'timed-out'}))
    monkeypatch.setattr('subprocess.run', fake_pipeline)
    with pytest.raises(RuntimeError, match='completed run'):
        training_preflight(root, {'pipeline_config': 'job.yaml'})


def test_pretrained_bundle_relocates_parent_and_preserves_token_roles(training_source):
    import io
    import json
    import tarfile

    from modelchef.aws_training import build_bundle
    root, config = training_source
    model = root / 'hf-model'
    model.mkdir()
    (model / 'config.json').write_text('{}')
    (model / 'model.safetensors').write_bytes(b'weights')
    (model / '.env').write_text('secret')
    parent = root / 'parent'
    (parent / 'checkpoints').mkdir(parents=True)
    config.model.backend = 'huggingface'
    config.model.pretrained_path = str(model)
    config.tokenizer.algorithm = 'pretrained'
    (parent / 'config.json').write_text(config.model_dump_json())
    (parent / 'report.json').write_text(json.dumps({'status': 'completed', 'optimizer_steps': 0}))
    (parent / 'tokenizer.json').write_text('{}')
    (parent / 'tokenizer-roles.json').write_text('{"<eos>":3}')
    (parent / 'checkpoints/step-000000.pt').write_bytes(b'checkpoint')
    config.stage.kind, config.stage.parent_run = 'sft', str(parent)
    config.model.peft.method = 'qlora'
    (root / 'job.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    with tarfile.open(fileobj=io.BytesIO(build_bundle(root, {'pipeline_config': 'job.yaml'}))) as archive:
        job = yaml.safe_load(archive.extractfile('configs/job.yaml').read())
        saved = json.loads(archive.extractfile('inputs/parent_run/config.json').read())
        assert job['model']['pretrained_path'] == saved['model']['pretrained_path']
        assert job['model']['pretrained_path'] == 'inputs/pretrained-0'
        assert 'inputs/pretrained-0/model.safetensors' in archive.getnames()
        assert 'inputs/pretrained-0/.env' not in archive.getnames()
        assert archive.extractfile('inputs/parent_run/tokenizer-roles.json').read() == b'{"<eos>":3}'
