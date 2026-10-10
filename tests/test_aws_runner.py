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
def test_run_terminates_after_test_or_watchdog_failure(
        settings, tmp_path, scheduler_fails, waiter_fails, monkeypatch):
    from types import SimpleNamespace

    from botocore.exceptions import WaiterError

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
    runner.scheduler = SimpleNamespace(create_schedule=create_schedule,
        delete_schedule=lambda **kw: calls.append("delete-schedule"))
    runner.ec2 = SimpleNamespace(
        create_security_group=lambda **kw: {"GroupId": "sg-test"},
        run_instances=lambda **kw: {"Instances": [{"InstanceId": "i-test"}]},
        describe_instances=lambda **kw: {"Reservations": [{"Instances": [
            {"InstanceId": "i-test", "State": {"Name": "terminated"}}]}]},
        terminate_instances=lambda **kw: calls.append("force" if kw.get("Force") else "terminate"),
        get_waiter=lambda _: SimpleNamespace(wait=wait),
        get_console_output=lambda **kw: {"Output": "MODELCHEF_RESULT=0"},
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


def test_training_uploads_bundle_and_terminates_before_result_fetch(settings, tmp_path, monkeypatch):
    import io
    from types import SimpleNamespace

    from modelchef.aws_runner import SpotRunner

    calls = []
    settings.update(pipeline_config='job.yaml', artifact_bucket='test-bucket',
                    instance_profile_name='test-profile')
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
    assert kinds.index('upload') < kinds.index('launch') < kinds.index('cleanup') < kinds.index('download')
    arguments = next(value for kind, value in calls if kind == 'launch')
    assert arguments['InstanceMarketOptions']['MarketType'] == 'spot'
    assert arguments['IamInstanceProfile'] == {'Name': 'test-profile'}
    assert Path(result['results_path']).read_bytes() == b'results'


@pytest.mark.parametrize("state", ["shutting-down", "terminated"])
def test_reconcile_preserves_watchdog_until_terminated(settings, tmp_path, state):
    from types import SimpleNamespace

    calls = []
    runner = SpotRunner.__new__(SpotRunner)
    runner.root, runner.settings, runner.account = tmp_path, settings, "test-account"
    ledger = Ledger(tmp_path / settings["ledger_path"])
    ledger.reserve("run-test", "test-account:us-east-1", Decimal("0.5"), "10")
    ledger.record("run-test", "launched", {"instance_id": "i-test"})
    runner.ec2 = SimpleNamespace(
        describe_instances=lambda **kw: {"Reservations": [{"Instances": [
            {"State": {"Name": state}}]}]},
        get_console_output=lambda **kw: {"Output": "MODELCHEF_RESULT=0"},
        describe_security_groups=lambda **kw: {"SecurityGroups": [{"GroupId": "sg-test"}]},
        delete_security_group=lambda **kw: calls.append("delete-sg"))
    runner.scheduler = SimpleNamespace(delete_schedule=lambda **kw: calls.append("delete-schedule"))
    result = runner.reconcile("run-test")
    assert result["test_passed"]
    assert result["terminated"] == (state == "terminated")
    assert calls == (["delete-schedule", "delete-sg"] if state == "terminated" else [])
