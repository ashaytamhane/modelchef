"""Budget-reserved, one-shot Spot tests. No embedded AWS credentials."""

import base64
import hashlib
import io
import json
import shlex
import sqlite3
import subprocess
import sys
import tarfile
import time
import uuid
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path


def local_preflight(root):
    """Fail closed before provisioning; persist evidence for the next run/reviewer."""
    root = Path(root).resolve()
    report = {"passed": False, "checks": []}
    output = root / "artifacts/aws-preflight.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    for name, arguments in [("pytest", ["pytest", "-q"]),
                            ("ruff", ["ruff", "check", "src", "tests"])]:
        result = subprocess.run([sys.executable, "-m", *arguments], cwd=root,
                                capture_output=True, text=True, timeout=300, check=False)
        report["checks"].append({"name": name, "exit_code": result.returncode,
                                 "output": result.stdout + result.stderr})
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"Local {name} preflight failed; no instance launched. See {output}")
    report["passed"] = True
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def budget_reservation(settings):
    def amount(key):
        value = Decimal(str(settings[key]))
        if not value.is_finite() or value <= 0:
            raise ValueError(f"{key} must be positive and finite")
        return value
    total = amount("total_budget_usd")
    per_run = amount("per_run_budget_usd")
    minutes = amount("max_runtime_minutes")
    hourly = amount("max_spot_hourly_usd")
    storage = amount("storage_gb")
    reserve = amount("safety_reserve_usd")
    # Include 3 minutes for scheduler granularity and termination, conservatively.
    hours = (minutes + 3) / 60
    estimate = hours * (hourly + amount("ipv4_hourly_usd") +
                        storage * amount("ebs_gb_month_usd") / 730) + reserve
    if estimate > per_run or per_run > total:
        raise ValueError("Configured runtime/rates cannot fit the per-run and total budgets")
    if minutes != int(minutes) or storage != int(storage):
        raise ValueError("Runtime minutes and storage GB must be integers")
    if not settings["instance_type"].startswith(("g4dn.", "g5.", "g6.", "g6e.")):
        raise ValueError("Only x86 NVIDIA G-series types are supported by this test recipe")
    return per_run


class Ledger:
    """Atomic reservations; charged reserves are retained even after a crash."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, timeout=10)
        self.db.execute("CREATE TABLE IF NOT EXISTS runs "
                        "(id TEXT PRIMARY KEY, scope TEXT, reserved TEXT, state TEXT, report TEXT)")
        self.db.commit()

    def reserve(self, run_id, scope, reserve, total):
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            used = sum((Decimal(row[0]) for row in self.db.execute(
                "SELECT reserved FROM runs WHERE scope=?", (scope,))), Decimal(0))
            if used + reserve > Decimal(str(total)):
                raise ValueError("Overall budget exhausted or reserved by another run")
            self.db.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?)",
                            (run_id, scope, str(reserve), "reserved", "{}"))

    def record(self, run_id, state, report):
        with self.db:
            self.db.execute("UPDATE runs SET state=?,report=? WHERE id=?",
                            (state, json.dumps(report, default=str), run_id))


def package_source(root):
    """Allowlist code files only: never package .aws, .env, keys, or Git history."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for folder in ["src/ashayllm", "tests"]:
            for path in sorted((root / folder).glob("*.py")):
                if path.name in {"aws_runner.py", "test_aws_runner.py"}:
                    continue
                archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
    return buffer.getvalue()


def user_data(root, settings):
    payload = base64.b64encode(package_source(root)).decode()
    minutes = int(settings["max_runtime_minutes"])
    inference = ("python -m pip install vllm==0.31.0\n" if settings.get("install_vllm") else "")
    # Arm a guest timer before extraction or installation; poweroff terminates the instance.
    script = f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/ashayllm-test.log /dev/console) 2>&1
systemd-run --unit=ashayllm-deadline --on-active={minutes}m /sbin/shutdown -h now
finish() {{ code=$?; echo "ASHAYLLM_RESULT=$code"; /sbin/shutdown -h now; }}
trap finish EXIT
mkdir -p /opt/ashayllm-test
cd /opt/ashayllm-test
echo {shlex.quote(payload)} | base64 -d | tar -xz
# The DLAMI activation script references optional unset environment variables.
set +u
source /opt/pytorch/bin/activate
set -u
python -m pip install 'pyyaml>=6,<7' 'pytest>=8,<10'
export PYTHONPATH=/opt/ashayllm-test/src
{inference}python -m ashayllm.cli doctor
python -m pytest -q
python -m ashayllm.cli smoke
python -m ashayllm.cli smoke --precision fp16
'''
    if len(script.encode()) > 16384:
        raise ValueError("Allowlisted source exceeds EC2 user-data limit")
    return script


class SpotRunner:
    def __init__(self, settings, root, session=None):
        import boto3
        from botocore.config import Config

        self.settings = settings
        self.root = Path(root).resolve()
        budget_reservation(settings)
        session = session or boto3.Session(profile_name=settings["profile"],
                                           region_name=settings["region"])
        cfg = Config(connect_timeout=5, read_timeout=10,
                     retries={"total_max_attempts": 3, "mode": "standard"})
        self.ec2 = session.client("ec2", config=cfg)
        self.iam = session.client("iam", config=cfg)
        self.scheduler = session.client("scheduler", config=cfg)
        self.ssm = session.client("ssm", config=cfg)
        self.account = session.client("sts", config=cfg).get_caller_identity()["Account"]

    def reconcile(self, run_id):
        """Recover a journaled run after interruption without provisioning anything."""
        ledger = Ledger(self.root / self.settings["ledger_path"])
        row = ledger.db.execute("SELECT scope,report FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row or row[0] != f"{self.account}:{self.settings['region']}":
            raise ValueError("Run not found in this account/region ledger")
        report = json.loads(row[1])
        instance_id = report.get("instance_id")
        if not instance_id:
            raise ValueError("This run has no journaled instance to reconcile")
        instance = self.ec2.describe_instances(InstanceIds=[instance_id])[
            "Reservations"][0]["Instances"][0]
        report["terminated"] = instance["State"]["Name"] == "terminated"
        report["console_output"] = self.ec2.get_console_output(
            InstanceId=instance_id, Latest=True).get("Output", "")
        report["test_passed"] = "ASHAYLLM_RESULT=0" in report["console_output"]
        ledger.record(run_id, "terminated" if report["terminated"] else "cleanup-pending", report)
        out = self.root / "artifacts/aws-runs" / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["terminated"]:
            try:
                self.scheduler.delete_schedule(Name=run_id, GroupName="ashayllm")
            except self.scheduler.exceptions.ResourceNotFoundException:
                pass
            groups = self.ec2.describe_security_groups(Filters=[
                {"Name": "group-name", "Values": [run_id]}])["SecurityGroups"]
            for group in groups:
                self.ec2.delete_security_group(GroupId=group["GroupId"])
        return report

    def plan(self):
        s = self.settings
        subnet_args = ({"SubnetIds": [s["subnet_id"]]} if s.get("subnet_id") else
                       {"Filters": [{"Name": "default-for-az", "Values": ["true"]}]})
        subnets = self.ec2.describe_subnets(**subnet_args)["Subnets"]
        prices = {}
        for page in self.ec2.get_paginator("describe_spot_price_history").paginate(
            InstanceTypes=[s["instance_type"]], ProductDescriptions=["Linux/UNIX"],
            StartTime=datetime.now(UTC)):
            for entry in page["SpotPriceHistory"]:
                prices.setdefault(entry["AvailabilityZone"], Decimal(entry["SpotPrice"]))
        choices = [subnet for subnet in subnets if subnet["State"] == "available" and
                   subnet["AvailabilityZone"] in prices and
                   prices[subnet["AvailabilityZone"]] <= Decimal(str(s["max_spot_hourly_usd"]))]
        if not choices:
            raise ValueError("No default subnet/Spot offering fits the configured hourly ceiling")
        subnet = min(choices, key=lambda sub: prices[sub["AvailabilityZone"]])
        ami = s.get("ami_id")
        if not ami:
            try:
                ami = self.ssm.get_parameter(Name=s["ami_parameter"])["Parameter"]["Value"]
            except (self.ssm.exceptions.AccessDeniedException,
                    self.ssm.exceptions.ParameterNotFound):
                images = self.ec2.describe_images(Owners=["amazon"], Filters=[
                    {"Name": "name", "Values": [s["ami_name"]]},
                    {"Name": "state", "Values": ["available"]}])["Images"]
                if not images:
                    raise ValueError("No Amazon-owned DLAMI matches the configured recipe") from None
                ami = max(images, key=lambda image: image["CreationDate"])["ImageId"]
        image = self.ec2.describe_images(ImageIds=[ami], Owners=["amazon"])["Images"][0]
        if image["Architecture"] != "x86_64":
            raise ValueError("AMI architecture does not match this x86 GPU recipe")
        root_mapping = next(m for m in image["BlockDeviceMappings"]
                            if m["DeviceName"] == image["RootDeviceName"])
        if root_mapping["Ebs"]["VolumeSize"] > s["storage_gb"]:
            raise ValueError("AMI root volume exceeds configured storage budget")
        script = user_data(self.root, s)
        return {"account": self.account, "region": s["region"], "ami_id": ami,
                "ami_name": image["Name"], "instance_type": s["instance_type"],
                "subnet_id": subnet["SubnetId"], "vpc_id": subnet["VpcId"],
                "availability_zone": subnet["AvailabilityZone"],
                "spot_price_usd_hour": str(prices[subnet["AvailabilityZone"]]),
                "root_device": image["RootDeviceName"],
                "reserved_usd": str(budget_reservation(s)),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "user_data": script}

    def prepare_deadline_role(self):
        region = self.settings["region"]
        try:
            self.scheduler.get_schedule_group(Name="ashayllm")
        except self.scheduler.exceptions.ResourceNotFoundException:
            self.scheduler.create_schedule_group(Name="ashayllm")
        name = "ashayllm-spot-deadline"
        try:
            return self.iam.get_role(RoleName=name)["Role"]["Arn"]
        except self.iam.exceptions.NoSuchEntityException:
            trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                "Principal": {"Service": "scheduler.amazonaws.com"},
                "Action": "sts:AssumeRole", "Condition": {
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnEquals": {"aws:SourceArn":
                        f"arn:aws:scheduler:{region}:{self.account}:schedule-group/ashayllm"}}}]}
            role = self.iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                                       Tags=[{"Key": "Project", "Value": "AshayLLM"}])["Role"]
            policy = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": f"arn:aws:ec2:{region}:{self.account}:instance/*",
                "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "AshayLLM"}}}]}
            self.iam.put_role_policy(RoleName=name, PolicyName="terminate-ashayllm-only",
                                     PolicyDocument=json.dumps(policy))
            self.iam.get_waiter("role_exists").wait(RoleName=name,
                                                  WaiterConfig={"Delay": 2, "MaxAttempts": 10})
            time.sleep(10)
            return role["Arn"]

    def run(self):
        s = self.settings
        preflight = local_preflight(self.root)
        plan = self.plan()
        run_id = "ashayllm-" + uuid.uuid4().hex
        ledger = Ledger(self.root / s["ledger_path"])
        scope = f"{self.account}:{s['region']}"
        ledger.reserve(run_id, scope, budget_reservation(s), s["total_budget_usd"])
        report = {k: v for k, v in plan.items() if k != "user_data"}
        report["run_id"] = run_id
        report["local_preflight"] = preflight
        started = time.monotonic()
        instance_id = None
        security_group = None
        schedule_created = False
        terminated = False
        try:
            role = self.prepare_deadline_role()
            security_group = self.ec2.create_security_group(
                GroupName=run_id, Description="AshayLLM outbound-only test",
                VpcId=plan["vpc_id"])["GroupId"]
            deadline = datetime.now(UTC) + timedelta(minutes=s["max_runtime_minutes"])
            launch = self.ec2.run_instances(
                ClientToken=run_id, MinCount=1, MaxCount=1,
                ImageId=plan["ami_id"], InstanceType=s["instance_type"],
                UserData=plan["user_data"], InstanceInitiatedShutdownBehavior="terminate",
                InstanceMarketOptions={"MarketType": "spot", "SpotOptions": {
                    "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate",
                    "MaxPrice": str(s["max_spot_hourly_usd"])}},
                MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": plan["subnet_id"],
                    "Groups": [security_group], "AssociatePublicIpAddress": True}],
                BlockDeviceMappings=[{"DeviceName": plan["root_device"], "Ebs": {
                    "VolumeSize": s["storage_gb"], "VolumeType": "gp3", "Encrypted": True,
                    "DeleteOnTermination": True}}],
                TagSpecifications=[{"ResourceType": kind, "Tags": [
                    {"Key": "Project", "Value": "AshayLLM"},
                    {"Key": "RunId", "Value": run_id}]} for kind in ["instance", "volume"]])
            instance_id = launch["Instances"][0]["InstanceId"]
            report["instance_id"] = instance_id
            ledger.record(run_id, "launched", report)
            self.scheduler.create_schedule(Name=run_id, GroupName="ashayllm",
                ScheduleExpression="at(" + deadline.strftime("%Y-%m-%dT%H:%M:%S") + ")",
                ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"},
                ActionAfterCompletion="DELETE", Target={
                    "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                    "RoleArn": role, "Input": json.dumps({"InstanceIds": [instance_id]}),
                    "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 3}})
            schedule_created = True
            while time.monotonic() - started < s["max_runtime_minutes"] * 60:
                instance = self.ec2.describe_instances(InstanceIds=[instance_id])[
                    "Reservations"][0]["Instances"][0]
                if instance["State"]["Name"] == "terminated":
                    terminated = True
                    break
                if instance["State"]["Name"] == "shutting-down":
                    break  # Confirm completion through the termination waiter below.
                time.sleep(15)
        finally:
            # Recover an instance ID if the launch response was lost after AWS accepted it.
            if instance_id is None and security_group:
                recovered = self.ec2.describe_instances(Filters=[
                    {"Name": "client-token", "Values": [run_id]},
                    {"Name": "instance-state-name", "Values": ["pending", "running",
                                                              "shutting-down", "stopped"]}])
                instances = [i for r in recovered["Reservations"] for i in r["Instances"]]
                if instances:
                    instance_id = instances[0]["InstanceId"]
                    report["instance_id"] = instance_id
            # Do not remove the AWS watchdog or release reservations before confirmed termination.
            if instance_id and not terminated:
                from botocore.exceptions import WaiterError

                self.ec2.terminate_instances(InstanceIds=[instance_id])
                waiter = self.ec2.get_waiter("instance_terminated")
                try:
                    waiter.wait(InstanceIds=[instance_id],
                                WaiterConfig={"Delay": 5, "MaxAttempts": 12})
                except WaiterError:
                    # Disposable test: no persistent data to preserve after a stalled shutdown.
                    self.ec2.terminate_instances(InstanceIds=[instance_id],
                                                 Force=True, SkipOsShutdown=True)
                    waiter.wait(InstanceIds=[instance_id],
                                WaiterConfig={"Delay": 5, "MaxAttempts": 60})
                terminated = True
            report.update(terminated=terminated, elapsed_seconds=time.monotonic() - started)
            if instance_id:
                console = self.ec2.get_console_output(InstanceId=instance_id, Latest=True)
                report["console_output"] = console.get("Output", "")
                report["test_passed"] = "ASHAYLLM_RESULT=0" in report["console_output"]
            ledger.record(run_id, "terminated" if terminated else "no-instance", report)
            out = self.root / "artifacts/aws-runs" / run_id
            out.mkdir(parents=True, exist_ok=True)
            (out / "report.json").write_text(json.dumps(report, indent=2, default=str))
            if terminated and schedule_created:
                try:
                    self.scheduler.delete_schedule(Name=run_id, GroupName="ashayllm")
                except self.scheduler.exceptions.ResourceNotFoundException:
                    pass
            if security_group and (terminated or not instance_id):
                self.ec2.delete_security_group(GroupId=security_group)
        return report
