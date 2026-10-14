"""Bounded CUDA guest execution; outputs stay on the mounted network volume."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def execute(seconds):
    started = time.monotonic()
    Path('artifacts').mkdir(exist_ok=True)
    path = Path('status.json')

    def status(state, **fields):
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps({'state': state, 'elapsed_seconds': time.monotonic()-started,
                                         **fields}), encoding='utf-8')
        temporary.replace(path)

    status('running')
    os.environ['MODELCHEF_LOCAL_CHECKPOINT_KEEP'] = '1'
    # Never carry AWS recovery settings into a different provider's guest.
    os.environ.pop('MODELCHEF_RECOVERY_S3_URI', None)
    setup = [sys.executable, '-m', 'pip', 'install', 'pyyaml>=6,<7', 'pydantic>=2.10,<3']
    train = [sys.executable, '-m', 'modelchef.aws_guest', 'configs/job.yaml', '--cloud-extra', 'none']
    try:
        for command in [setup, train]:
            remaining = seconds-(time.monotonic()-started)
            if remaining <= 0:
                raise TimeoutError('Guest runtime limit reached')
            process = subprocess.Popen(command, start_new_session=True)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, required=True)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error('--seconds must be positive')
    execute(args.seconds)


if __name__ == '__main__':
    main()
