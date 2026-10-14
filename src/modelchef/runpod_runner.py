"""Run unchanged ModelChef recipes on budget-reviewed Runpod REST v2 GPU Pods."""
import hashlib
import ipaddress
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import time
import tomllib
import uuid
from decimal import Decimal
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field, model_validator

from modelchef.progress import report as progress


class RunpodConfig(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    gpu_type_id: str = Field(pattern=r'^NVIDIA .+')
    gpu_count: int = Field(default=1, ge=1)
    cloud: str = Field(default='SECURE', pattern='^(SECURE|COMMUNITY)$')
    image: str = 'runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404'
    network_volume_id: str = Field(min_length=1)
    data_center_ids: list[str] = Field(default_factory=list)
    container_disk_gb: int = Field(default=50, ge=10)
    min_cuda_version: str = Field(default='13.0', pattern=r'^\d+\.\d+$')
    ssh_private_key: str = '~/.ssh/id_ed25519'
    ssh_public_key: str = '~/.ssh/id_ed25519.pub'
    api_key_env: str = Field(default='RUNPOD_API_KEY', pattern=r'^[A-Z][A-Z0-9_]*$')
    max_compute_hourly_usd: float = Field(gt=0)
    storage_hourly_usd: float = Field(gt=0)
    safety_reserve_usd: float = Field(default=0.5, gt=0)
    per_run_budget_usd: float = Field(gt=0)
    total_budget_usd: float = Field(gt=0)
    max_runtime_minutes: int = Field(default=30, ge=1)
    poll_seconds: int = Field(default=15, ge=5, le=60)
    api_timeout_seconds: int = Field(default=20, ge=1, le=60)
    max_bundle_bytes: int = Field(default=64_000_000, ge=1)
    max_result_bytes: int = Field(default=20_000_000_000, ge=1)
    budget_scope: str = 'default'
    local_preflight: bool = True
    download_results: bool = False

    @model_validator(mode='after')
    def budget(self):
        if self.estimated_maximum > Decimal(str(self.per_run_budget_usd)):
            raise ValueError('Configured runtime/rates exceed per_run_budget_usd')
        if self.per_run_budget_usd > self.total_budget_usd:
            raise ValueError('per_run_budget_usd exceeds total_budget_usd')
        return self

    @property
    def estimated_maximum(self):
        return ((Decimal(str(self.max_compute_hourly_usd))+Decimal(str(self.storage_hourly_usd)))
                * Decimal(self.max_runtime_minutes+3)/60+Decimal(str(self.safety_reserve_usd)))


class RunpodHTTPError(RuntimeError):
    def __init__(self, method, status):
        self.status = status
        super().__init__(f'Runpod API {method} returned HTTP {status}')


class RunpodAPI:
    """No embedded keys, no secret-bearing URLs, and no automatic POST retries."""
    def __init__(self, settings):
        self.key = os.environ.get(settings.api_key_env)
        if not self.key and settings.api_key_env == 'RUNPOD_API_KEY':
            configuration = Path.home()/'.runpod/config.toml'
            legacy = Path.home()/'.runpod.yaml'
            try:
                if configuration.is_file():
                    with configuration.open('rb') as source:
                        saved = tomllib.load(source)
                    self.key = next((value for key, value in saved.items() if key.lower() == 'apikey'), None)
                elif legacy.is_file():
                    import yaml
                    try:
                        saved = yaml.safe_load(legacy.read_text(encoding='utf-8')) or {}
                    except yaml.YAMLError:
                        raise ValueError('Invalid saved Runpod CLI credential file') from None
                    if not isinstance(saved, dict):
                        raise ValueError('Invalid saved Runpod CLI credential file')
                    self.key = next((value for key, value in saved.items() if key.lower() == 'apikey'), None)
            except (OSError, ValueError):
                raise ValueError('Cannot read the saved Runpod CLI credentials; configure RUNPOD_API_KEY') from None
        if not self.key:
            raise ValueError(f'Set {settings.api_key_env} locally; do not put API keys in recipes')
        if not isinstance(self.key, str) or any(ord(character) < 33 or ord(character) > 126 for character in self.key):
            raise ValueError('Invalid Runpod API key format; configure credentials locally')
        self.timeout = settings.api_timeout_seconds

    def request(self, method, path, payload=None):
        request = Request('https://api.runpod.io/v2'+path,
                          data=json.dumps(payload).encode() if payload is not None else None,
                          headers={'Authorization': 'Bearer '+self.key, 'Content-Type': 'application/json'},
                          method=method)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content = response.read(8_000_001)
                if len(content) > 8_000_000:
                    raise ValueError('Runpod API response exceeded its size limit')
                return json.loads(content) if content else None
        except HTTPError as error:
            # Response bodies can echo environment values. Never print or persist them.
            raise RunpodHTTPError(method, error.code) from None


def find_owned_pods(api, name):
    matches, cursor, seen = [], None, set()
    while True:
        page = api.request('GET', '/pods'+('?' + urlencode({'cursor': cursor}) if cursor else ''))
        matches.extend(pod for pod in page['pods'] if pod['name'] == name)
        pagination = page.get('pagination', {})
        if not pagination.get('hasNextPage'):
            return matches
        cursor = pagination.get('nextCursor')
        if not cursor or cursor in seen:
            raise ValueError('Invalid Runpod pagination response')
        seen.add(cursor)


def manage_run(directory, action, execute=False):
    directory = Path(directory).resolve()
    path = directory/'report.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    if report.get('target') != 'runpod' or not re.fullmatch(r'modelchef-[a-f0-9]{32}', report.get('name', '')):
        raise ValueError('Expected an owned ModelChef Runpod run report')
    if action == 'terminate' and not execute:
        return {'action': action, 'name': report['name'], 'note': 'Preview; --run deletes this owned Pod'}
    api = RunpodAPI(RunpodConfig.model_validate(report['settings']))
    pods = find_owned_pods(api, report['name'])
    if len(pods) > 1:
        raise ValueError('Multiple Pods match this run; refusing ambiguous cleanup')
    if not pods:
        return {'name': report['name'], 'status': 'absent', 'network_volume_id': report['network_volume_id']}
    pod = pods[0]
    if report.get('pod_id') and report['pod_id'] != pod['id']:
        raise ValueError('Pod ownership differs from saved run')
    if action == 'terminate':
        api.request('DELETE', '/pods/'+quote(pod['id'], safe=''))
        report.update(pod_id=pod['id'], termination_confirmed=True, cleanup_required=False)
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        return {'pod_id': pod['id'], 'termination_confirmed': True}
    if action in {'logs', 'fetch'}:
        runner = RunpodRunner(report['settings'], directory, directory/'unused-recipe.yaml', api=api)
        if action == 'logs':
            return {'pod_id': pod['id'], 'logs': runner.ssh(pod, directory,
                'tail -n 100 '+shlex.quote(report['recovery_path']+'/guest.log'))}
        runner.fetch(pod, directory, report['recovery_path'])
        return {'pod_id': pod['id'], 'results': str(directory/'results')}
    return {key: pod.get(key) for key in ['id', 'name', 'status', 'cost', 'createdAt']}


def extract_results(archive, target, max_bytes):
    """Extract regular files only, into a new directory, within the declared cap."""
    target = Path(target).resolve()
    with tarfile.open(archive, 'r:gz') as source:
        entries, total = [], 0
        for entry in source:
            total += entry.size
            if total > max_bytes or len(entries) >= 100_000:
                raise ValueError('Runpod results exceed their size or file-count cap')
            destination = (target/entry.name).resolve()
            if not destination.is_relative_to(target) or not (entry.isfile() or entry.isdir()):
                raise ValueError('Unsafe path or link in Runpod result archive')
            entries.append(entry)
        if target.exists():
            raise FileExistsError('Result destination already exists')
        target.mkdir(parents=True)
        source.extractall(target, members=entries, filter='data')


class RunpodRunner:
    def __init__(self, settings, root, recipe, *, api=None):
        self.settings = RunpodConfig.model_validate(settings)
        self.root, self.recipe = Path(root).resolve(), Path(recipe).resolve()
        self.api = api
        self.code_root = Path(__file__).resolve().parents[2]

    def bundle(self):
        from modelchef.aws_training import build_bundle
        if not (self.code_root/'pyproject.toml').is_file():
            raise ValueError('Runpod packaging requires an editable ModelChef source checkout')
        return build_bundle(self.root, {'pipeline_config': str(self.recipe),
                            '_source_root': str(self.code_root),
                            'max_bundle_bytes': self.settings.max_bundle_bytes})

    def plan(self):
        bundle = self.bundle()
        return {'target': 'runpod', 'gpu_type_id': self.settings.gpu_type_id,
                'gpu_count': self.settings.gpu_count, 'recipe': str(self.recipe),
                'max_runtime_minutes': self.settings.max_runtime_minutes,
                'estimated_maximum_usd': str(self.settings.estimated_maximum),
                'reserved_usd': str(self.settings.per_run_budget_usd),
                'total_budget_usd': str(self.settings.total_budget_usd),
                'bundle_bytes': len(bundle), 'bundle_sha256': hashlib.sha256(bundle).hexdigest(),
                'network_volume_id': self.settings.network_volume_id,
                'note': 'Offline estimate using configured rate ceilings, not a live quote. '
                        'Controller must remain running to terminate the Pod. '
                        'Network-volume charges continue after Pod deletion.'}

    def payload(self, name, public_key):
        s = self.settings
        # Explicit SSH bootstrap avoids relying on account-wide key injection or image entrypoints.
        boot = ('set -eu; command -v sshd >/dev/null || '
                '(apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server); '
                'mkdir -p /root/.ssh /run/sshd; chmod 700 /root/.ssh; '
                'printf "%s\\n" "$PUBLIC_KEY" > /root/.ssh/authorized_keys; '
                'chmod 600 /root/.ssh/authorized_keys; ssh-keygen -A; '
                'exec /usr/sbin/sshd -D -e')
        return {'name': name, 'cloud': s.cloud, 'image': s.image,
                'gpu': {'id': s.gpu_type_id, 'count': s.gpu_count, 'minCudaVersion': s.min_cuda_version},
                'disk': s.container_disk_gb, 'ports': ['22/tcp'],
                'mounts': {'network': [{'volumeId': s.network_volume_id, 'path': '/workspace'}]},
                'dataCenterIds': s.data_center_ids,
                'env': {'PUBLIC_KEY': public_key}, 'entrypoint': ['/bin/bash', '-lc'], 'cmd': [boot]}

    def transport(self, pod, directory):
        endpoint = pod.get('ssh', {}).get('direct')
        if not endpoint:
            raise ValueError('Pod has no direct SSH connection yet')
        host = str(ipaddress.ip_address(endpoint['host']))
        port = int(endpoint['port'])
        if not 1 <= port <= 65535:
            raise ValueError('Invalid SSH port')
        username = endpoint['username']
        if not re.fullmatch(r'[a-zA-Z0-9_-]+', username):
            raise ValueError('Invalid SSH username')
        options = ['-i', str(Path(self.settings.ssh_private_key).expanduser()),
                   '-o', 'BatchMode=yes', '-o', 'IdentitiesOnly=yes', '-o', 'ConnectTimeout=10',
                   '-o', 'StrictHostKeyChecking=accept-new',
                   '-o', 'UserKnownHostsFile='+str(directory/'known_hosts')]
        return host, port, username, options

    def ssh(self, pod, directory, command, timeout=30):
        timeout = min(timeout, max(1, getattr(self, 'deadline', time.monotonic()+timeout)-time.monotonic()))
        host, port, user, options = self.transport(pod, directory)
        return subprocess.run(['ssh', *options, '-p', str(port), user+'@'+host, command],
                              capture_output=True, text=True, timeout=timeout, check=True).stdout

    def copy(self, pod, directory, local, remote, *, download=False, timeout=30):
        timeout = min(timeout, max(1, getattr(self, 'deadline', time.monotonic()+timeout)-time.monotonic()))
        host, port, user, options = self.transport(pod, directory)
        scp_host = '['+host+']' if ':' in host else host
        endpoint = f'{user}@{scp_host}:{remote}'
        arguments = [endpoint, str(local)] if download else [str(local), endpoint]
        subprocess.run(['scp', *options, '-P', str(port), *arguments], capture_output=True,
                       timeout=timeout, check=True)

    def fetch(self, pod, directory, remote):
        size = int(self.ssh(pod, directory,
            f'cd {shlex.quote(remote)} && tar -czf results.tar.gz artifacts configs status.json guest.log '
            '&& stat -c %s results.tar.gz', timeout=180).strip())
        if size > self.settings.max_result_bytes:
            raise ValueError('Compressed results exceed max_result_bytes; retained on network volume')
        if shutil.disk_usage(directory).free < size*2+1_000_000_000:
            raise OSError('Insufficient local disk for result download; retained on network volume')
        archive = directory/'results.tar.gz'
        self.copy(pod, directory, archive, remote+'/results.tar.gz', download=True, timeout=300)
        extract_results(archive, directory/'results', self.settings.max_result_bytes)
        archive.unlink()

    def run(self):
        from modelchef.aws_runner import Ledger, local_preflight
        s = self.settings
        bundle = self.bundle()
        for tool in ['ssh', 'scp']:
            if not shutil.which(tool):
                raise ValueError(f'Install OpenSSH client ({tool}) before launching')
        private, public = Path(s.ssh_private_key).expanduser(), Path(s.ssh_public_key).expanduser()
        if not private.is_file() or not public.is_file():
            raise ValueError('Configure an SSH private/public key pair before launching')
        public_key = public.read_text().strip()
        if not re.fullmatch(r'(ssh-ed25519|ssh-rsa|ecdsa-sha2-\S+) [A-Za-z0-9+/=]+(?: .*?)?', public_key):
            raise ValueError('Invalid SSH public key format')
        if s.local_preflight:
            progress('Running local source tests and lint before any Runpod provisioning.')
            local_preflight(self.code_root)
        api = self.api or RunpodAPI(s)
        quote_result = api.request('GET', '/catalog/gpus/'+quote(s.gpu_type_id, safe=''))
        rate = Decimal(str(quote_result['price'][s.cloud.lower()]))*s.gpu_count
        if rate <= 0 or rate > Decimal(str(s.max_compute_hourly_usd)):
            raise ValueError('Live Runpod catalog rate exceeds configured ceiling or is unavailable')
        run_id = 'modelchef-'+uuid.uuid4().hex
        directory = self.root/'artifacts/runpod-runs'/run_id
        directory.mkdir(parents=True)
        (directory/'bundle.tar.gz').write_bytes(bundle)
        ledger = Ledger(self.root/'artifacts/runpod-budget.sqlite')
        ledger.reserve(run_id, 'runpod:'+s.budget_scope, Decimal(str(s.per_run_budget_usd)), s.total_budget_usd)
        report = {'run_id': run_id, 'name': run_id, 'target': 'runpod', 'test_passed': False,
                  'termination_confirmed': False, 'settings': s.model_dump(),
                  'network_volume_id': s.network_volume_id, 'recovery_path': '/workspace/'+run_id,
                  'quoted_compute_hourly_usd': str(rate), 'reserved_usd': str(s.per_run_budget_usd),
                  'bundle_sha256': hashlib.sha256(bundle).hexdigest()}
        remote, pod = report['recovery_path'], None
        started = time.monotonic()
        deadline = started+s.max_runtime_minutes*60
        self.deadline = deadline

        def save():
            report['elapsed_seconds'] = time.monotonic()-started
            report['estimated_spend_bound_usd'] = (report['elapsed_seconds']/3600
                * (s.max_compute_hourly_usd+s.storage_hourly_usd)+s.safety_reserve_usd)
            (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            ledger.record(run_id, report.get('state', 'running'), report)

        save()  # Persist ownership before the non-idempotent launch call.
        try:
            progress('Requesting one Runpod GPU Pod and waiting for its direct SSH connection.')
            try:
                pod = api.request('POST', '/pods', self.payload(run_id, public_key))
            except Exception as launch_error:  # noqa: BLE001 -- reconcile ambiguous launch failures
                # Never blindly retry creation: reconcile the unique saved name instead.
                report['state'] = 'launch-uncertain'
                report['launch_error_type'] = type(launch_error).__name__
                if isinstance(launch_error, RunpodHTTPError):
                    report['launch_http_status'] = launch_error.status
                save()
                candidates = find_owned_pods(api, run_id)
                if len(candidates) == 1:
                    pod = candidates[0]
                else:
                    raise RuntimeError('Launch outcome uncertain; reconcile saved run before retrying') from None
            report['pod_id'] = pod['id']
            save()
            while time.monotonic() < deadline:
                pod = api.request('GET', '/pods/'+quote(pod['id'], safe=''))
                if pod.get('status') in {'ERROR', 'EXITED', 'TERMINATED'}:
                    raise RuntimeError('Pod stopped before the guest started; inspect network-volume results')
                actual_rate = Decimal(str(pod.get('cost', 0)))
                if actual_rate > Decimal(str(s.max_compute_hourly_usd)):
                    raise ValueError('Actual Pod hourly rate exceeds configured ceiling')
                if pod.get('ssh', {}).get('direct'):
                    try:
                        self.ssh(pod, directory, 'true')
                        break
                    except (subprocess.SubprocessError, ValueError):
                        pass
                time.sleep(s.poll_seconds)
            else:
                raise TimeoutError('Provisioning exceeded the runtime deadline')
            self.ssh(pod, directory, 'mkdir -p '+shlex.quote(remote))
            self.copy(pod, directory, directory/'bundle.tar.gz', remote+'/bundle.tar.gz',
                      timeout=max(1, int(deadline-time.monotonic())))
            seconds = max(1, int(deadline-time.monotonic()))
            start = (f'cd {shlex.quote(remote)} && echo {report["bundle_sha256"]} " bundle.tar.gz" '
                     '| sha256sum -c - && tar -xzf bundle.tar.gz && '
                     f'(nohup env PYTHONPATH=src python -u -m modelchef.runpod_guest --seconds {seconds} '
                     '>guest.log 2>&1 </dev/null &)')
            self.ssh(pod, directory, start)
            report['state'] = 'running'
            save()
            progress(f'Runpod guest started; pod={pod["id"]}; recovery={remote}.')
            while time.monotonic() < deadline:
                cost_bound = ((time.monotonic()-started)/3600*(s.max_compute_hourly_usd+s.storage_hourly_usd)
                              +s.safety_reserve_usd)
                if cost_bound >= s.per_run_budget_usd:
                    raise TimeoutError('Per-run spending allowance reached')
                try:
                    status = json.loads(self.ssh(pod, directory, 'cat '+shlex.quote(remote+'/status.json')))
                except (subprocess.SubprocessError, json.JSONDecodeError):
                    status = {'state': 'starting'}
                if status['state'] in {'completed', 'failed'}:
                    report['guest'] = status
                    if s.download_results:
                        self.fetch(pod, directory, remote)
                    report['results_location'] = 'network-volume'
                    report['results_downloaded'] = s.download_results
                    report['test_passed'] = status['state'] == 'completed'
                    report['state'] = status['state']
                    break
                time.sleep(s.poll_seconds)
            else:
                raise TimeoutError('Runpod runtime deadline reached')
        except BaseException as error:  # noqa: BLE001 -- interrupts must also trigger cleanup
            report['state'] = report.get('state', 'launch-uncertain') if pod is None else 'failed'
            report['error'] = type(error).__name__  # Do not serialize request/SSH exception secrets.
            if isinstance(error, KeyboardInterrupt):
                report['error'] = 'Interrupted by user'
            progress('Runpod job stopped; inspect the saved report and network-volume recovery path.')
        finally:
            if pod is not None:
                progress('Requesting Runpod Pod deletion; the network-volume recovery data is retained.')
                for attempt in range(3):
                    try:
                        api.request('DELETE', '/pods/'+quote(pod['id'], safe=''))
                        report['termination_confirmed'] = True
                        break
                    except RunpodHTTPError as error:
                        if error.status == 404:
                            report['termination_confirmed'] = True
                            break
                    except Exception:  # noqa: BLE001 -- retry idempotent cleanup, preserve failure evidence
                        report['cleanup_attempts'] = attempt+1
                    if attempt < 2:
                        time.sleep(attempt+1)
                if not report['termination_confirmed']:
                    report['cleanup_required'] = True
            save()
            ledger.db.close()
        return report


def manage_volume(settings, volume_id, action='status', execute=False, api=None):
    """Network volumes are deleted only by an explicit command naming their ID."""
    if action not in {'status', 'delete'}:
        raise ValueError('Volume action must be status or delete')
    api = api or RunpodAPI(RunpodConfig.model_validate(settings))
    path = '/network-volumes/' + quote(volume_id, safe='')
    volume = api.request('GET', path)
    if volume['id'] != volume_id:
        raise ValueError('Volume read-back identity differs')
    result = {'volume': volume, 'action': action, 'deleted': False,
              'note': 'Stored data is permanently removed only with delete --run; storage bills until deletion.'}
    if action == 'delete' and execute:
        api.request('DELETE', path)
        try:
            api.request('GET', path)
        except RunpodHTTPError as error:
            if error.status != 404:
                raise
            result['deleted'] = True
        else:
            raise RuntimeError('Volume deletion unconfirmed by read-back')
    return result
