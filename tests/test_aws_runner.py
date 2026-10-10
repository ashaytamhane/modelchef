from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from ashayllm.aws_runner import Ledger, SpotRunner, budget_reservation, user_data


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
    monkeypatch.setattr("ashayllm.aws_runner.local_preflight", lambda _: {"passed": True})
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
        get_console_output=lambda **kw: {"Output": "ASHAYLLM_RESULT=0"},
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

    from ashayllm.aws_runner import local_preflight

    monkeypatch.setattr("ashayllm.aws_runner.subprocess.run", lambda *a, **kw:
                        SimpleNamespace(returncode=1, stdout="test failed", stderr=""))
    runner = SpotRunner.__new__(SpotRunner)
    runner.settings, runner.root = settings, tmp_path
    runner.plan = lambda: pytest.fail("AWS plan must not run after failed local tests")
    with pytest.raises(RuntimeError, match="no instance launched"):
        runner.run()
    assert '"passed": false' in (tmp_path / "artifacts/aws-preflight.json").read_text()
    with pytest.raises(RuntimeError):
        local_preflight(tmp_path)


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
        get_console_output=lambda **kw: {"Output": "ASHAYLLM_RESULT=0"},
        describe_security_groups=lambda **kw: {"SecurityGroups": [{"GroupId": "sg-test"}]},
        delete_security_group=lambda **kw: calls.append("delete-sg"))
    runner.scheduler = SimpleNamespace(delete_schedule=lambda **kw: calls.append("delete-schedule"))
    result = runner.reconcile("run-test")
    assert result["test_passed"]
    assert result["terminated"] == (state == "terminated")
    assert calls == (["delete-schedule", "delete-sg"] if state == "terminated" else [])
