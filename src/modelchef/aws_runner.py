"""Budget-reserved, one-shot EC2 tests. No embedded AWS credentials."""

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
    market = settings.get('market_type', 'spot')
    if market not in {'spot', 'on_demand'}:
        raise ValueError('market_type must be spot or on_demand')
    hourly = amount('max_spot_hourly_usd' if market == 'spot' else 'on_demand_hourly_usd')
    storage = amount("storage_gb")
    reserve = amount("safety_reserve_usd")
    # Include 3 minutes for scheduler granularity and termination, conservatively.
    hours = (minutes + 3) / 60
    estimate = hours * (hourly + amount("ipv4_hourly_usd") +
                        storage * amount("ebs_gb_month_usd") / 730) + reserve
    if settings.get('pipeline_config'):
        artifact_reserve = Decimal(str(settings.get('artifact_cost_reserve_usd', '0.05')))
        if not artifact_reserve.is_finite() or artifact_reserve <= 0:
            raise ValueError('artifact_cost_reserve_usd must be positive and finite')
        estimate += artifact_reserve
    if estimate > per_run or per_run > total:
        raise ValueError("Configured runtime/rates cannot fit the per-run and total budgets")
    if minutes != int(minutes) or storage != int(storage):
        raise ValueError("Runtime minutes and storage GB must be integers")
    neuron_probe = settings.get('job_backend') == 'neuron_probe'
    if neuron_probe and (settings['instance_type'] != 'trn1.2xlarge'
                         or not settings.get('pipeline_config')
                         or minutes > 60 or storage > 100):
        raise ValueError('Neuron probes require trn1.2xlarge, a recipe, <=60 minutes and <=100 GB')
    if not neuron_probe and not settings["instance_type"].startswith(("g4dn.", "g5.", "g6.", "g6e.")):
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

    def release_unlaunched(self, run_id):
        """Return budget only after the report proves EC2 never created an instance."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            row = self.db.execute(
                "SELECT state,report FROM runs WHERE id=?", (run_id,)).fetchone()
            report = json.loads(row[1]) if row else {}
            rejected = {'InsufficientInstanceCapacity', 'VcpuLimitExceeded', 'MaxSpotInstanceCountExceeded',
                        'ConfirmedNoInstance'}
            if (not row or row[0] != 'no-instance' or report.get('instance_id') or
                    report.get('launch_error_code') not in rejected):
                raise ValueError('Cannot release a reservation for an unconfirmed launch')
            self.db.execute("UPDATE runs SET reserved='0' WHERE id=?", (run_id,))

    def settle_terminated(self, run_id, report):
        """Replace a reservation with its conservative bound after confirmed cleanup."""
        from decimal import Decimal
        with self.db:
            self.db.execute('BEGIN IMMEDIATE')
            row = self.db.execute('SELECT reserved FROM runs WHERE id=?', (run_id,)).fetchone()
            bound = Decimal(str(report.get('estimated_spend_bound_usd', 'NaN')))
            settings = report.get('settings', {})
            elapsed = Decimal(str(report.get('elapsed_seconds', 'NaN')))
            rate = Decimal(str(settings.get('max_compute_hourly_usd', 'NaN')))
            storage = Decimal(str(settings.get('storage_hourly_usd', 'NaN')))
            reserve = Decimal(str(settings.get('safety_reserve_usd', 'NaN')))
            valid_components = all(value.is_finite() and value >= 0
                                   for value in [elapsed, rate, storage, reserve])
            minimum = elapsed / 3600 * (rate + storage) + reserve if valid_components else None
            if (not row or not report.get('termination_confirmed') or
                    not report.get('cleanup_verified_by_read') or
                    not bound.is_finite() or not valid_components or bound < minimum or
                    bound > Decimal(row[0])):
                raise ValueError('Settlement requires verified termination and a valid spend bound')
            self.db.execute('UPDATE runs SET reserved=?,state=?,report=? WHERE id=?',
                            (str(bound), 'settled', json.dumps(report, default=str), run_id))


def package_source(root):
    """Allowlist code files only: never package .aws, .env, keys, or Git history."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        files = ['src/modelchef/' + name for name in
                 ['__init__.py', 'cli.py', 'hardware.py', 'inference.py', 'smoke.py',
                  'progress.py', 'diagnostics.py']]
        files += ['tests/' + name for name in ['test_cli.py', 'test_hardware.py', 'test_inference.py']]
        files += ['configs/local.yaml']
        for filename in files:
            path = root / ('src/modelchef/guest_cli.py' if filename == 'src/modelchef/cli.py' else filename)
            content = path.read_bytes()
            entry = tarfile.TarInfo(filename)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return buffer.getvalue()


def user_data(root, settings):
    if settings.get('pipeline_config'):
        from modelchef.aws_training import training_user_data
        return training_user_data(settings)
    payload = base64.b64encode(package_source(root)).decode()
    minutes = int(settings["max_runtime_minutes"])
    inference = ("python -m pip install vllm==0.31.0\n" if settings.get("install_vllm") else "")
    # Arm a guest timer before extraction or installation; poweroff terminates the instance.
    script = f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/modelchef-test.log /dev/console) 2>&1
systemd-run --unit=modelchef-deadline --on-active={minutes}m /sbin/shutdown -h now
finish() {{ code=$?; echo "MODELCHEF_RESULT=$code"; /sbin/shutdown -h now; }}
trap finish EXIT
mkdir -p /opt/modelchef-test
cd /opt/modelchef-test
echo {shlex.quote(payload)} | base64 -d | tar -xz
# The DLAMI activation script references optional unset environment variables.
set +u
source /opt/pytorch/bin/activate
set -u
python -m pip install 'pyyaml>=6,<7' 'pytest>=8,<10'
export PYTHONPATH=/opt/modelchef-test/src
{inference}python -m modelchef.cli doctor
python -m pytest -q
python -m modelchef.cli smoke
python -m modelchef.cli smoke --precision fp16
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
        cfg = Config(connect_timeout=15, read_timeout=30,
                     retries={"total_max_attempts": 3, "mode": "standard"})
        self.ec2 = session.client("ec2", config=cfg)
        self.iam = session.client("iam", config=cfg)
        self.scheduler = session.client("scheduler", config=cfg)
        self.ssm = session.client("ssm", config=cfg)
        if settings.get('pipeline_config'):
            self.s3 = session.client('s3', config=cfg)
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
            response = self.ec2.describe_instances(Filters=[
                *([] if self.settings.get('fleet_instance_types') else
                  [{'Name': 'client-token', 'Values': [run_id]}]),
                {'Name': 'tag:Project', 'Values': ['ModelChef']},
                {'Name': 'tag:RunId', 'Values': [run_id]}])
            found = [i for reservation in response['Reservations'] for i in reservation['Instances']]
            if not found and report.get('started_at') and datetime.now(UTC) - datetime.fromisoformat(
                    report['started_at']) >= timedelta(minutes=15):
                # Avoid treating a briefly inconsistent EC2 listing as a rejected launch.
                for group in self.ec2.describe_security_groups(Filters=[
                        {'Name': 'group-name', 'Values': [run_id]}])['SecurityGroups']:
                    self.ec2.delete_security_group(GroupId=group['GroupId'])
                report.update(launch_error_code='ConfirmedNoInstance',
                              no_instance_checked_at=datetime.now(UTC).isoformat())
                report.pop('cleanup_error', None)
                ledger.record(run_id, 'no-instance', report)
                ledger.release_unlaunched(run_id)
                out = self.root / 'artifacts/aws-runs' / run_id
                out.mkdir(parents=True, exist_ok=True)
                (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                return report
            if len(found) != 1:
                raise ValueError('Expected one instance for this run; inspect AWS before retrying')
            instance_id = found[0]['InstanceId']
            report['instance_id'] = instance_id
            ledger.record(run_id, 'recovered', report)
        instance = self.ec2.describe_instances(InstanceIds=[instance_id])[
            "Reservations"][0]["Instances"][0]
        report["terminated"] = instance["State"]["Name"] == "terminated"
        report["console_output"] = (self.ec2.get_console_output(
            InstanceId=instance_id, Latest=True).get("Output")
            or report.get("console_output", ""))
        report["test_passed"] = "MODELCHEF_RESULT=0" in report["console_output"]
        ledger.record(run_id, "terminated" if report["terminated"] else "cleanup-pending", report)
        out = self.root / "artifacts/aws-runs" / run_id
        out.mkdir(parents=True, exist_ok=True)
        (out / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        if report["terminated"]:
            try:
                self.scheduler.delete_schedule(Name=run_id, GroupName="modelchef")
            except self.scheduler.exceptions.ResourceNotFoundException:
                pass
            groups = self.ec2.describe_security_groups(Filters=[
                {"Name": "group-name", "Values": [run_id]}])["SecurityGroups"]
            for group in groups:
                self.ec2.delete_security_group(GroupId=group["GroupId"])
            if report.get('artifact_bucket'):
                self.fetch_training_results(report, out)
        return report

    def plan(self):
        s = self.settings
        market = s.get('market_type', 'spot')
        subnet_args = ({"SubnetIds": [s["subnet_id"]]} if s.get("subnet_id") else
                       {"Filters": [{"Name": "default-for-az", "Values": ["true"]}]})
        subnets = self.ec2.describe_subnets(**subnet_args)["Subnets"]
        prices = {}
        if market == 'spot':
            for page in self.ec2.get_paginator("describe_spot_price_history").paginate(
                InstanceTypes=[s["instance_type"]], ProductDescriptions=["Linux/UNIX"],
                StartTime=datetime.now(UTC)):
                for entry in page["SpotPriceHistory"]:
                    prices.setdefault(entry["AvailabilityZone"], Decimal(entry["SpotPrice"]))
        choices = [subnet for subnet in subnets if subnet["State"] == "available" and
                   (market == 'on_demand' or (subnet["AvailabilityZone"] in prices and
                    prices[subnet["AvailabilityZone"]] <= Decimal(str(s["max_spot_hourly_usd"]))))]
        if not choices:
            raise ValueError('No available subnet fits the configured market and hourly ceiling')
        subnet = min(choices, key=(lambda sub: prices[sub['AvailabilityZone']])
                     if market == 'spot' else (lambda sub: sub['AvailabilityZone']))
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
                "market_type": market,
                "hourly_budget_rate_usd": str(s['max_spot_hourly_usd'] if market == 'spot'
                                             else s['on_demand_hourly_usd']),
                "spot_price_usd_hour": (str(prices[subnet["AvailabilityZone"]])
                                         if market == 'spot' else None),
                "root_device": image["RootDeviceName"],
                "reserved_usd": str(budget_reservation(s)),
                "source_sha256": hashlib.sha256(script.encode()).hexdigest(),
                "user_data": script}

    def prepare_deadline_role(self):
        region = self.settings["region"]
        try:
            self.scheduler.get_schedule_group(Name="modelchef")
        except self.scheduler.exceptions.ResourceNotFoundException:
            self.scheduler.create_schedule_group(Name="modelchef")
        name = "modelchef-spot-deadline"
        try:
            return self.iam.get_role(RoleName=name)["Role"]["Arn"]
        except self.iam.exceptions.NoSuchEntityException:
            trust = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                "Principal": {"Service": "scheduler.amazonaws.com"},
                "Action": "sts:AssumeRole", "Condition": {
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnEquals": {"aws:SourceArn":
                        f"arn:aws:scheduler:{region}:{self.account}:schedule-group/modelchef"}}}]}
            role = self.iam.create_role(RoleName=name, AssumeRolePolicyDocument=json.dumps(trust),
                                       Tags=[{"Key": "Project", "Value": "ModelChef"}])["Role"]
            policy = {"Version": "2012-10-17", "Statement": [{"Effect": "Allow",
                "Action": "ec2:TerminateInstances",
                "Resource": f"arn:aws:ec2:{region}:{self.account}:instance/*",
                "Condition": {"StringEquals": {"ec2:ResourceTag/Project": "ModelChef"}}}]}
            self.iam.put_role_policy(RoleName=name, PolicyName="terminate-modelchef-only",
                                     PolicyDocument=json.dumps(policy))
            self.iam.get_waiter("role_exists").wait(RoleName=name,
                                                  WaiterConfig={"Delay": 2, "MaxAttempts": 10})
            time.sleep(10)
            return role["Arn"]

    def run(self):
        from modelchef.progress import report as progress

        s = self.settings
        market = s.get('market_type', 'spot')
        progress('Running local tests and lint before requesting any cloud compute.')
        preflight = local_preflight(self.root)
        bundle = None
        if s.get('pipeline_config'):
            from modelchef.aws_training import build_bundle, training_preflight
            bundle = build_bundle(self.root, s)
            progress('Running the bounded local CPU training preflight before cloud launch.')
            preflight['training'] = training_preflight(self.root, s)
        progress('Resolving the AWS image, subnet, market rate and budget reservation.')
        plan = self.plan()
        run_id = "modelchef-" + uuid.uuid4().hex
        ledger = Ledger(self.root / s["ledger_path"])
        scope = f"{self.account}:{s['region']}"
        ledger.reserve(run_id, scope, budget_reservation(s), s["total_budget_usd"])
        report = {k: v for k, v in plan.items() if k != "user_data"}
        report["run_id"] = run_id
        report["local_preflight"] = preflight
        report['started_at'] = datetime.now(UTC).isoformat()
        out = self.root / 'artifacts/aws-runs' / run_id
        out.mkdir(parents=True, exist_ok=True)

        def persist(state):
            ledger.record(run_id, state, report)
            temporary = out / 'report.json.tmp'
            temporary.write_text(json.dumps(report, indent=2, default=str), encoding='utf-8')
            temporary.replace(out / 'report.json')

        persist('reserved')
        started = time.monotonic()
        instance_id = None
        security_group = None
        schedule_created = False
        terminated = False
        try:
            if bundle is not None:
                progress('Uploading the approved training bundle to encrypted S3 storage.')
                prefix = s.get('artifact_prefix', 'modelchef-training').strip('/') + '/' + run_id
                self.s3.put_object(Bucket=s['artifact_bucket'], Key=prefix + '/source.tar.gz',
                    Body=bundle, ServerSideEncryption='AES256')
                report.update(artifact_bucket=s['artifact_bucket'], artifact_prefix=prefix,
                              bundle_sha256=hashlib.sha256(bundle).hexdigest())
                report['recovery_uri'] = f"s3://{s['artifact_bucket']}/{prefix}/recovery"
                plan['user_data'] = user_data(self.root, {**s, '_artifact_prefix': prefix,
                                                         '_bundle_sha256': report['bundle_sha256']})
            progress('Preparing the shutdown deadline and outbound-only instance network.')
            role = self.prepare_deadline_role()
            security_group = self.ec2.create_security_group(
                GroupName=run_id, Description="ModelChef outbound-only test",
                VpcId=plan["vpc_id"])["GroupId"]
            report['security_group_id'] = security_group
            persist('launching')
            deadline = datetime.now(UTC) + timedelta(minutes=s["max_runtime_minutes"])
            profile = ({'IamInstanceProfile': {'Name': s['instance_profile_name']}}
                       if bundle is not None else {})
            progress(f'Requesting the configured {market} instance.')
            from botocore.exceptions import ClientError

            try:
                def request_instance(**request):
                    if s.get('fleet_instance_types'):
                        from modelchef.aws_fleet import launch as fleet_launch
                        return fleet_launch(self.ec2, s, plan, request)
                    return self.ec2.run_instances(**request)

                launch = request_instance(
                    **profile,
                    ClientToken=run_id, MinCount=1, MaxCount=1,
                    ImageId=plan["ami_id"], InstanceType=s["instance_type"],
                    UserData=plan["user_data"], InstanceInitiatedShutdownBehavior="terminate",
                    **({'InstanceMarketOptions': {"MarketType": "spot", "SpotOptions": {
                        "SpotInstanceType": "one-time", "InstanceInterruptionBehavior": "terminate",
                        "MaxPrice": str(s["max_spot_hourly_usd"])}}}
                       if market == 'spot' else {}),
                    MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
                    NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": plan["subnet_id"],
                        "Groups": [security_group], "AssociatePublicIpAddress": True}],
                    BlockDeviceMappings=[{"DeviceName": plan["root_device"], "Ebs": {
                        "VolumeSize": s["storage_gb"], "VolumeType": "gp3", "Encrypted": True,
                        "DeleteOnTermination": True}}],
                    TagSpecifications=[{"ResourceType": kind, "Tags": [
                        {"Key": "Project", "Value": "ModelChef"},
                        {"Key": "RunId", "Value": run_id}]} for kind in ["instance", "volume"]])
            except ClientError as error:
                report['launch_error_code'] = error.response['Error']['Code']
                persist('launch-rejected')
                raise
            instance_id = launch["Instances"][0]["InstanceId"]
            report["instance_id"] = instance_id
            persist('launched')
            self.scheduler.create_schedule(Name=run_id, GroupName="modelchef",
                ScheduleExpression="at(" + deadline.strftime("%Y-%m-%dT%H:%M:%S") + ")",
                ScheduleExpressionTimezone="UTC", FlexibleTimeWindow={"Mode": "OFF"},
                ActionAfterCompletion="DELETE", Target={
                    "Arn": "arn:aws:scheduler:::aws-sdk:ec2:terminateInstances",
                    "RoleArn": role, "Input": json.dumps({"InstanceIds": [instance_id]}),
                    "RetryPolicy": {"MaximumEventAgeInSeconds": 300, "MaximumRetryAttempts": 3}})
            schedule_created = True
            progress('Monitoring EC2 state until guest completion or the configured deadline. Guest training logs are collected after termination; unchanged state is not repeated.')
            last_state = None
            while time.monotonic() - started < s["max_runtime_minutes"] * 60:
                instance = self.ec2.describe_instances(InstanceIds=[instance_id])[
                    "Reservations"][0]["Instances"][0]
                state = instance["State"]["Name"]
                if s.get('fleet_instance_types'):
                    report['actual_instance_type'] = instance.get('InstanceType')
                    report['actual_availability_zone'] = instance.get('Placement', {}).get('AvailabilityZone')
                if state != last_state:
                    progress(f'EC2 state changed to {state}.')
                    last_state = state
                if state == "terminated":
                    terminated = True
                    break
                if instance["State"]["Name"] == "shutting-down":
                    # EC2 may discard the serial console after termination.
                    # Keep the last available guest result before the waiter runs.
                    from botocore.exceptions import BotoCoreError, ClientError

                    try:
                        output = self.ec2.get_console_output(
                            InstanceId=instance_id, Latest=True).get("Output")
                        if output:
                            report["console_output"] = output
                            persist('guest-shutting-down')
                    except (BotoCoreError, ClientError) as error:
                        report['early_console_error'] = str(error)
                    break  # Confirm completion through the termination waiter below.
                time.sleep(15)
        finally:
            try:
                # Recover an instance ID if AWS accepted a launch whose response was lost.
                if instance_id is None and security_group:
                    recovered = self.ec2.describe_instances(Filters=[
                        *([] if s.get('fleet_instance_types') else
                          [{"Name": "client-token", "Values": [run_id]}]),
                        {"Name": "instance-state-name", "Values": ["pending", "running",
                                                                  "shutting-down", "stopped"]}])
                    instances = [i for r in recovered["Reservations"] for i in r["Instances"]]
                    if instances:
                        instance_id = instances[0]["InstanceId"]
                        report["instance_id"] = instance_id
                        persist('recovered')
                # Preserve the watchdog and reservation until termination is confirmed.
                if instance_id and not terminated:
                    progress('Requesting instance termination and waiting for AWS confirmation.')
                    from botocore.exceptions import WaiterError

                    self.ec2.terminate_instances(InstanceIds=[instance_id])
                    waiter = self.ec2.get_waiter("instance_terminated")
                    try:
                        waiter.wait(InstanceIds=[instance_id],
                                    WaiterConfig={"Delay": 5, "MaxAttempts": 12})
                    except WaiterError:
                        # Disposable test: no persistent data after a stalled shutdown.
                        self.ec2.terminate_instances(InstanceIds=[instance_id],
                                                     Force=True, SkipOsShutdown=True)
                        waiter.wait(InstanceIds=[instance_id],
                                    WaiterConfig={"Delay": 5, "MaxAttempts": 60})
                    terminated = True
                report.update(terminated=terminated, elapsed_seconds=time.monotonic() - started)
                persist('terminated' if terminated else 'no-instance')
                if instance_id:
                    progress('Collecting guest console results and cleaning up temporary cloud resources.')
                    from botocore.exceptions import BotoCoreError, ClientError

                    try:
                        console = self.ec2.get_console_output(InstanceId=instance_id, Latest=True)
                        report["console_output"] = console.get("Output") or report.get(
                            "console_output", "")
                        report["test_passed"] = "MODELCHEF_RESULT=0" in report["console_output"]
                    except (BotoCoreError, ClientError) as error:
                        # Console collection must never strand otherwise disposable resources.
                        report['console_error'] = str(error)
                        report['test_passed'] = False
                if terminated and schedule_created:
                    try:
                        self.scheduler.delete_schedule(Name=run_id, GroupName="modelchef")
                    except self.scheduler.exceptions.ResourceNotFoundException:
                        pass
                if security_group and (terminated or not instance_id):
                    self.ec2.delete_security_group(GroupId=security_group)
                persist('terminated' if terminated else 'no-instance')
                if (not instance_id and report.get('launch_error_code') in
                        {'InsufficientInstanceCapacity', 'VcpuLimitExceeded', 'MaxSpotInstanceCountExceeded'}):
                    ledger.release_unlaunched(run_id)
            except BaseException as error:
                report.update(terminated=terminated, elapsed_seconds=time.monotonic() - started,
                              cleanup_error=str(error))
                persist('cleanup-pending')
                raise
        if bundle is not None and terminated:
            progress('Downloading the completed training result bundle from S3.')
            self.fetch_training_results(report, out)
        return report

    def fetch_training_results(self, report, out):
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            response = self.s3.get_object(Bucket=report['artifact_bucket'],
                                         Key=report['artifact_prefix'] + '/results.tar.gz')
        except self.s3.exceptions.NoSuchKey:
            report['results_missing'] = True
            (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            return
        except (BotoCoreError, ClientError) as error:
            report['results_error'] = str(error)
            (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            raise RuntimeError('Training instance is terminated, but result retrieval failed. '
                               f'Inspect {out / "report.json"} and the S3 artifact bucket; '
                               'do not rerun training solely to retry the download.') from error
        cap = int(self.settings.get('max_result_bytes', 512_000_000))
        body = response['Body']
        try:
            content = body.read(cap + 1)
        finally:
            body.close()
        if len(content) > cap:
            raise ValueError('Cloud result exceeds max_result_bytes; instance already terminated')
        path = out / 'results.tar.gz'
        path.write_bytes(content)
        report['results_sha256'] = hashlib.sha256(content).hexdigest()
        report['results_path'] = str(path)
        (out / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
