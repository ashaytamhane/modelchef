"""Persistent on-demand hosting, separate from the disposable Spot runner."""
import hashlib
import io
import json
import shlex
import tarfile
import uuid
from datetime import UTC, datetime
from pathlib import Path

from modelchef.inference import serve_plan


def validate(settings):
    required = ['profile', 'region', 'instance_type', 'ami_id', 'subnet_id',
                'security_group_ids', 'instance_profile_name', 'artifact_bucket',
                'secret_id', 'inference_config', 'hourly_estimate_usd', 'alert_after_usd']
    if any(not settings.get(k) for k in required):
        raise ValueError('Persistent serving requires explicit identity, network, image, secrets and cost settings')
    if not settings['instance_type'].startswith(('g4dn.', 'g5.', 'g6.', 'g6e.')):
        raise ValueError('Serving recipe requires an x86 NVIDIA G-series instance')
    if not isinstance(settings['security_group_ids'], list):
        raise TypeError('security_group_ids must be a list of existing restricted groups')
    for name in ['hourly_estimate_usd', 'alert_after_usd']:
        import math
        if not math.isfinite(float(settings[name])) or float(settings[name]) <= 0:
            raise ValueError(f'{name} must be positive and finite')


def bundle(root, settings):
    import yaml
    root = Path(root).resolve()
    config = yaml.safe_load((root / settings['inference_config']).read_text())
    inference = config['inference']
    inference = dict(inference)
    # Validate the local export without requiring production secrets on the controller.
    local_security = dict(inference.get('security', {}))
    local_security.pop('tls_certfile', None)
    local_security.pop('tls_keyfile', None)
    local = {**inference, 'host': '127.0.0.1', 'security': local_security}
    serve_plan({'inference': local}, root, check_hardware=False)
    model = (root / inference['model_path']).resolve()
    cap = int(settings.get('max_bundle_bytes', 1_000_000_000))
    if cap <= 0:
        raise ValueError('max_bundle_bytes must be positive')
    files = {p.relative_to(root).as_posix(): p for p in (root / 'src/modelchef').glob('*.py')}
    files['pyproject.toml'] = root / 'pyproject.toml'
    if (root / 'README.md').is_file():
        files['README.md'] = root / 'README.md'
    for p in model.iterdir():
        if p.is_file() and (p.suffix in {'.json', '.safetensors', '.model', '.bin'} or
                            p.name in {'merges.txt', 'vocab.txt', 'chat_template.jinja'}):
            files['model/' + p.name] = p
    if sum(p.stat().st_size for p in files.values()) > cap:
        raise ValueError('Serving source/model exceeds bundle cap')
    inference['model_path'] = '/opt/modelchef-serving/model'
    inference['host'] = '0.0.0.0'
    security = dict(inference.get('security', {}))
    security.update(api_key_env='MODELCHEF_API_KEY',
        tls_certfile='/etc/modelchef-serving/fullchain.pem',
        tls_keyfile='/etc/modelchef-serving/server.key')
    inference['security'] = security
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz', dereference=True) as archive:
        for name, path in sorted(files.items()):
            content = path.read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
        content = yaml.safe_dump({'inference': inference}).encode()
        info = tarfile.TarInfo('configs/serving.yaml')
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))
    return output.getvalue()


def guest_script(settings, object_key, checksum):
    source = shlex.quote(f"s3://{settings['artifact_bucket']}/{object_key}")
    secret = shlex.quote(settings['secret_id'])
    region = shlex.quote(settings['region'])
    # No shutdown trap or scheduled deadline: deployment failures remain inspectable.
    return f'''#!/bin/bash
set -euo pipefail
umask 077
mkdir -p /opt/modelchef-serving /etc/modelchef-serving
cd /opt/modelchef-serving
set +u
source /opt/pytorch/bin/activate
set -u
command -v aws >/dev/null || python -m pip install awscli
aws s3 cp {source} /tmp/modelchef-serving.tar.gz --region {region}
echo '{checksum}  /tmp/modelchef-serving.tar.gz' | sha256sum -c -
tar -xzf /tmp/modelchef-serving.tar.gz
python -m venv --system-site-packages .venv
.venv/bin/python -m pip install '.[inference]'
id modelchef-serving >/dev/null 2>&1 || useradd --system --home /opt/modelchef-serving --shell /usr/sbin/nologin modelchef-serving
aws secretsmanager get-secret-value --secret-id {secret} --region {region} --query SecretString --output text > /etc/modelchef-serving/secret.json
.venv/bin/python - <<'PY'
import json
from pathlib import Path
directory = Path('/etc/modelchef-serving')
secret = json.loads((directory / 'secret.json').read_text())
token = secret['api_key']
if len(token) < 32 or any(c.isspace() for c in token) or not token.isascii() or not all(c.isalnum() or c in '-_' for c in token):
    raise ValueError('Secret API key must be a random URL-safe token of at least 32 characters')
for filename, field in [('fullchain.pem','tls_cert'),('server.key','tls_key')]:
    (directory / filename).write_text(secret[field])
(directory / 'service.env').write_text('MODELCHEF_API_KEY=' + token + '\\n')
(directory / 'secret.json').unlink()
PY
chown -R modelchef-serving:modelchef-serving /opt/modelchef-serving /etc/modelchef-serving
chmod 700 /etc/modelchef-serving
cat > /etc/systemd/system/modelchef-serving.service <<'UNIT'
[Unit]
Description=ModelChef persistent vLLM API
After=network-online.target
Wants=network-online.target
[Service]
User=modelchef-serving
WorkingDirectory=/opt/modelchef-serving
EnvironmentFile=/etc/modelchef-serving/service.env
Environment=HF_HUB_OFFLINE=1
Environment=TRANSFORMERS_OFFLINE=1
Environment=VLLM_NO_USAGE_STATS=1
Environment=HOME=/opt/modelchef-serving
ExecStart=/opt/modelchef-serving/.venv/bin/python -m modelchef.cli serve-vllm configs/serving.yaml --run
Restart=on-failure
RestartSec=10
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=full
UMask=0077
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now modelchef-serving
'''


class ServingRunner:
    def __init__(self, settings, root, session=None):
        import boto3
        from botocore.config import Config
        validate(settings)
        self.settings, self.root = settings, Path(root).resolve()
        session = session or boto3.Session(profile_name=settings['profile'], region_name=settings['region'])
        cfg = Config(connect_timeout=5, read_timeout=10,
                     retries={'total_max_attempts': 3, 'mode': 'standard'})
        self.ec2, self.s3, self.ssm = (session.client(n, config=cfg) for n in ['ec2', 's3', 'ssm'])

    def plan(self):
        payload = bundle(self.root, self.settings)
        return {'market': 'on-demand', 'lifecycle': 'persistent', 'automatic_shutdown': False,
                'instance_type': self.settings['instance_type'], 'bytes': len(payload),
                'source_sha256': hashlib.sha256(payload).hexdigest(),
                'hourly_estimate_usd': self.settings['hourly_estimate_usd'],
                'note': 'No AWS calls made by plan; estimates are caller supplied, not billing limits.'}

    def _path(self, deployment):
        if not deployment.startswith('hosting-') or not deployment[8:].isalnum():
            raise ValueError('Invalid deployment ID')
        return self.root / 'artifacts/aws-serving' / (deployment + '.json')

    def deploy(self):
        from modelchef.aws_runner import local_preflight
        payload = bundle(self.root, self.settings)
        local_preflight(self.root)
        image = self.ec2.describe_images(ImageIds=[self.settings['ami_id']])['Images'][0]
        if image['Architecture'] != 'x86_64' or image['State'] != 'available':
            raise ValueError('Serving AMI must be available and x86_64')
        root_device = self.settings.get('root_device', image['RootDeviceName'])
        if root_device != image['RootDeviceName']:
            raise ValueError('root_device must match the selected AMI')
        volume = next(m['Ebs'] for m in image['BlockDeviceMappings'] if m['DeviceName'] == root_device)
        if int(self.settings.get('storage_gb', 100)) < volume['VolumeSize']:
            raise ValueError('storage_gb is smaller than the AMI root volume')
        deployment = 'hosting-' + uuid.uuid4().hex
        record = {'deployment_id': deployment, 'region': self.settings['region'],
                  'state': 'launching', 'object_key': f'modelchef-serving/{deployment}/source.tar.gz'}
        path = self._path(deployment)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2))
        self.s3.put_object(Bucket=self.settings['artifact_bucket'], Key=record['object_key'],
                           Body=payload, ServerSideEncryption='AES256')
        s = self.settings
        response = self.ec2.run_instances(ClientToken=deployment, MinCount=1, MaxCount=1,
            ImageId=s['ami_id'], InstanceType=s['instance_type'],
            IamInstanceProfile={'Name': s['instance_profile_name']},
            InstanceInitiatedShutdownBehavior='stop',
            MetadataOptions={'HttpTokens': 'required'},
            NetworkInterfaces=[{'DeviceIndex': 0, 'SubnetId': s['subnet_id'],
                'Groups': s['security_group_ids'], 'AssociatePublicIpAddress': True}],
            BlockDeviceMappings=[{'DeviceName': root_device, 'Ebs': {
                'VolumeSize': int(s.get('storage_gb', 100)), 'VolumeType': 'gp3',
                'Encrypted': True,
                'DeleteOnTermination': s.get('delete_storage_on_termination', False)}}],
            TagSpecifications=[{'ResourceType': kind, 'Tags': [
                {'Key': 'Project', 'Value': 'ModelChef'}, {'Key': 'DeploymentId', 'Value': deployment},
                {'Key': 'Lifecycle', 'Value': 'persistent-serving'}]} for kind in ['instance', 'volume']],
            UserData=guest_script(s, record['object_key'], hashlib.sha256(payload).hexdigest()))
        record['instance_id'] = response['Instances'][0]['InstanceId']
        record['state'] = 'provisioned'
        path.write_text(json.dumps(record, indent=2))
        return record

    def instance(self, deployment):
        record = json.loads(self._path(deployment).read_text())
        if record['region'] != self.settings['region']:
            raise ValueError('Deployment region differs from config')
        filters = [{'Name': 'client-token', 'Values': [deployment]}]
        response = self.ec2.describe_instances(Filters=filters)
        found = [i for r in response['Reservations'] for i in r['Instances']]
        if len(found) != 1:
            raise ValueError('Expected one journaled hosting instance; inspect AWS before retrying deploy')
        instance = found[0]
        tags = {t['Key']: t['Value'] for t in instance.get('Tags', [])}
        if tags.get('Project') != 'ModelChef' or tags.get('Lifecycle') != 'persistent-serving' or tags.get('DeploymentId') != deployment:
            raise ValueError('Instance ownership tags do not match deployment')
        return instance

    def action(self, deployment, action):
        instance = self.instance(deployment)
        identifier = instance['InstanceId']
        if action in {'stop', 'start', 'delete'}:
            method = {'stop': 'stop_instances', 'start': 'start_instances', 'delete': 'terminate_instances'}[action]
            getattr(self.ec2, method)(InstanceIds=[identifier])
            return {'instance_id': identifier, 'requested': action, 'confirmed': False}
        if action == 'logs':
            return self.ssm.send_command(InstanceIds=[identifier], DocumentName='AWS-RunShellScript',
                Parameters={'commands': ['systemctl status modelchef-serving --no-pager; journalctl -u modelchef-serving -n 100 --no-pager']})['Command']
        age = max(0, (datetime.now(UTC) - instance['LaunchTime']).total_seconds() / 3600)
        estimate = age * float(self.settings['hourly_estimate_usd'])
        return {'instance_id': identifier, 'state': instance['State']['Name'],
                'public_ip': instance.get('PublicIpAddress'), 'estimated_since_launch_usd': estimate,
                'budget_alert': estimate >= float(self.settings['alert_after_usd']),
                'automatic_shutdown': False,
                'note': 'Estimate includes stopped time, resets after start; not actual billing. Configure AWS Budgets alerts separately.'}
