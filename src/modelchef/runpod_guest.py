"""Bounded CUDA guest execution; outputs stay on the mounted network volume."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def wait_tensorboard(server, seconds):
    deadline = time.monotonic() + max(0, seconds)
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError('TensorBoard stopped during startup; inspect tensorboard.log')
        try:
            with urlopen('http://127.0.0.1:6006/', timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(.2)
    raise TimeoutError('TensorBoard startup timed out; inspect tensorboard.log')


def runtime_commands(directory):
    """Use an existing venv or create an isolated one without altering system Python."""
    if sys.prefix != sys.base_prefix:
        return [], sys.executable
    directory = Path(directory).resolve()
    python = directory / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    # The image already supplies pip. Inherit it, avoiding a dependency on Debian's
    # optional ensurepip package and PEP 668's system-package installation restriction.
    create = [sys.executable, '-m', 'venv', '--without-pip', '--system-site-packages',
              str(directory)]
    return [create], str(python)


def execute(seconds):
    started = time.monotonic()
    Path('artifacts').mkdir(exist_ok=True)
    path = Path('status.json')

    def status(state, **fields):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'state': state, 'elapsed_seconds': time.monotonic()-started,
                                         'tensorboard_url': os.environ.get('MODELCHEF_TENSORBOARD_URL'),
                                         **fields}), encoding='utf-8')
        temporary.replace(path)

    status('running')
    guest_env = os.environ.copy()
    guest_env['MODELCHEF_LOCAL_CHECKPOINT_KEEP'] = '1'
    # Never carry AWS recovery settings into a different provider's guest.
    guest_env.pop('MODELCHEF_RECOVERY_S3_URI', None)
    initialization, python = runtime_commands(Path('.runtime'))
    setup = [python, '-m', 'pip', 'install', 'pyyaml>=6,<7', 'pydantic>=2.10,<3', 'tensorboard>=2.19,<3']
    train = [python, '-m', 'modelchef.aws_guest', 'configs/job.yaml', '--cloud-extra', 'none']
    server = None
    try:
        for command in [*initialization, setup, train]:
            if command is train:
                with Path('tensorboard.log').open('a') as server_log:
                    server = subprocess.Popen([python, '-m', 'tensorboard.main', '--logdir', 'artifacts',
                        '--host', '0.0.0.0', '--port', '6006', '--reload_interval', '5'],
                        stdout=server_log, stderr=server_log, env=guest_env)
                wait_tensorboard(server, min(20, seconds-(time.monotonic()-started)))
            remaining = seconds-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Guest runtime limit reached')
            process = subprocess.Popen(command, start_new_session=True, env=guest_env)
            try:
                code = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise TimeoutError('Guest runtime limit reached') from None
            if code:
                raise RuntimeError(f'Guest command failed with exit code {code}')
        status('completed', exit_code=0)
    except Exception as error:
        status('failed', error=str(error))
        raise
    finally:
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    execute(args.seconds)


if __name__ == '__main__':
    main()
