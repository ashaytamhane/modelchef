import io
import json
import tarfile
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from modelchef.aws_serving import ServingRunner, bundle, guest_script


@pytest.fixture
def hosting(tmp_path):
    (tmp_path / 'src/modelchef').mkdir(parents=True)
    (tmp_path / 'src/modelchef/__init__.py').write_text('')
    (tmp_path / 'pyproject.toml').write_text('')
    model = tmp_path / 'model'
    model.mkdir()
    for name in ['config.json', 'tokenizer.json', 'model.safetensors']:
        (model / name).write_text('{}')
    (tmp_path / 'private.key').write_text('DO_NOT_PACKAGE')
    (tmp_path / 'inference.yaml').write_text(yaml.safe_dump({'inference': {'model_path': 'model', 'port': 8443}}))
    settings = {'profile': 'example', 'region': 'us-east-1', 'instance_type': 'g4dn.xlarge',
        'ami_id': 'ami-example', 'subnet_id': 'subnet-example', 'security_group_ids': ['sg-example'],
        'instance_profile_name': 'example', 'artifact_bucket': 'example', 'secret_id': 'secret-example',
        'inference_config': 'inference.yaml', 'hourly_estimate_usd': '1', 'alert_after_usd': '10'}
    clients = {name: Mock() for name in ['ec2', 's3', 'ssm']}
    clients['ec2'].describe_images.return_value = {'Images': [{'Architecture': 'x86_64',
        'State': 'available', 'RootDeviceName': '/dev/xvda', 'BlockDeviceMappings': [
            {'DeviceName': '/dev/xvda', 'Ebs': {'VolumeSize': 50}}]}]}
    session = SimpleNamespace(client=lambda name, **kw: clients[name])
    return ServingRunner(settings, tmp_path, session), clients


def test_plan_bundle_and_service_have_no_deadline(hosting):
    runner, clients = hosting
    assert runner.plan()['automatic_shutdown'] is False
    assert not clients['ec2'].mock_calls and not clients['s3'].mock_calls
    with tarfile.open(fileobj=io.BytesIO(bundle(runner.root, runner.settings))) as archive:
        assert 'private.key' not in archive.getnames()
        config = yaml.safe_load(archive.extractfile('configs/serving.yaml').read())
        assert config['inference']['security']['api_key_env'] == 'MODELCHEF_API_KEY'
    script = guest_script(runner.settings, 'source.tar.gz', 'a' * 64)
    assert 'Restart=on-failure' in script and 'systemctl enable --now' in script
    assert 'shutdown' not in script and 'terminate' not in script
    assert 'secretsmanager get-secret-value' in script
    assert len(script.encode()) < 16384


def test_deploy_is_on_demand_and_does_not_cleanup_after_launch(hosting, monkeypatch):
    runner, clients = hosting
    monkeypatch.setattr('modelchef.aws_runner.local_preflight', lambda root: {'passed': True})
    clients['ec2'].run_instances.return_value = {'Instances': [{'InstanceId': 'i-example'}]}
    record = runner.deploy()
    args = clients['ec2'].run_instances.call_args.kwargs
    assert 'InstanceMarketOptions' not in args
    assert args['InstanceInitiatedShutdownBehavior'] == 'stop'
    assert args['MetadataOptions']['HttpTokens'] == 'required'
    assert not clients['ec2'].terminate_instances.called
    assert record['instance_id'] == 'i-example'


def test_lifecycle_checks_ownership_and_status_never_stops(hosting):
    runner, clients = hosting
    deployment = 'hosting-example'
    path = runner._path(deployment)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'region': runner.settings['region']}))
    instance = {'InstanceId': 'i-example', 'State': {'Name': 'running'},
        'LaunchTime': datetime(2020, 1, 1, tzinfo=UTC), 'Tags': [
            {'Key': 'Project', 'Value': 'ModelChef'},
            {'Key': 'Lifecycle', 'Value': 'persistent-serving'},
            {'Key': 'DeploymentId', 'Value': deployment}]}
    clients['ec2'].describe_instances.return_value = {'Reservations': [{'Instances': [instance]}]}
    assert runner.action(deployment, 'status')['budget_alert']
    assert not clients['ec2'].stop_instances.called
    runner.action(deployment, 'stop')
    clients['ec2'].stop_instances.assert_called_once_with(InstanceIds=['i-example'])
    runner.action(deployment, 'start')
    runner.action(deployment, 'delete')
    clients['ec2'].terminate_instances.assert_called_once_with(InstanceIds=['i-example'])
    instance['Tags'] = []
    with pytest.raises(ValueError, match='ownership'):
        runner.action(deployment, 'delete')


def test_bundle_cap_prevents_upload(hosting):
    runner, clients = hosting
    runner.settings['max_bundle_bytes'] = 1
    with pytest.raises(ValueError, match='cap'):
        runner.deploy()
    assert not clients['s3'].put_object.called
