"""Bounded training bundles and guest scripts for the existing Spot lifecycle."""
import io
import json
import shlex
import tarfile
import uuid
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig


def build_bundle(root, settings):
    root = Path(root).resolve()
    config = RunConfig.model_validate(yaml.safe_load((root / settings['pipeline_config']).read_text()))
    if config.model.backend != 'native':
        raise ValueError('Managed AWS training currently packages native models only')
    if config.training.distributed == 'ddp':
        raise ValueError('Managed aws-train launches one process; DDP requires torchrun on a prepared host')
    if config.model.quantization == 'nvfp4':
        raise ValueError('Managed aws-train does not install Transformer Engine for native NVFP4; '
                         'use a prepared Blackwell host with the nvfp4 extra')
    cap = int(settings.get('max_bundle_bytes', 64_000_000))
    if cap <= 0:
        raise ValueError('max_bundle_bytes must be positive')
    files = {}

    def add_file(name, path):
        remaining = cap - sum(map(len, files.values()))
        if path.stat().st_size > remaining:
            raise ValueError('Uncompressed training bundle exceeds max_bundle_bytes')
        with path.open('rb') as source:
            content = source.read(remaining + 1)
        if len(content) > remaining:
            raise ValueError('Uncompressed training bundle exceeds max_bundle_bytes')
        files[name] = content

    for path in sorted((root / 'src/modelchef').glob('*.py')):
        add_file(path.relative_to(root).as_posix(), path)
    add_file('pyproject.toml', root / 'pyproject.toml')
    if (root / 'README.md').is_file():
        add_file('README.md', root / 'README.md')
    recipe = config.model_dump(mode='json')
    if config.model.custom_swiglu_path:
        raise ValueError('Custom CUDA managed AWS packaging is not supported yet; run on a prepared host')
    if config.data.provider == 'local_jsonl':
        path = root / config.data.local_path
        rows, read_bytes = [], 0
        with path.open('rb') as source:
            index = 0
            remaining = cap - sum(map(len, files.values()))
            while len(rows) < config.data.max_documents:
                line = source.readline(min(config.data.max_download_bytes, remaining) - read_bytes + 1)
                if not line:
                    break
                read_bytes += len(line)
                if read_bytes > config.data.max_download_bytes:
                    raise ValueError('Local training input exceeds byte budget')
                if read_bytes > remaining:
                    raise ValueError('Uncompressed training bundle exceeds max_bundle_bytes')
                if index >= config.data.offset:
                    rows.append(line)
                index += 1
        files['inputs/data.jsonl'] = b''.join(rows)
        recipe['data'].update(local_path='inputs/data.jsonl', offset=0)
    for field in ['parent_run', 'teacher_run']:
        if not getattr(config.stage, field):
            continue
        parent = (root / getattr(config.stage, field)).resolve()
        saved = RunConfig.model_validate_json((parent / 'config.json').read_text())
        if saved.model.backend != 'native':
            raise ValueError('AWS bundles cannot reference external pretrained paths')
        report = json.loads((parent / 'report.json').read_text())
        if report['status'] != 'completed':
            raise ValueError('AWS parent run must be complete')
        relative = f'inputs/{field}'
        checkpoint = f"checkpoints/step-{report['optimizer_steps']:06d}.pt"
        for name in ['config.json', 'report.json', 'tokenizer.json', checkpoint]:
            path = parent / name
            add_file(f'{relative}/{name}', path)
        recipe['stage'][field] = relative
    for index, filename in enumerate(config.processing.benchmark_files):
        add_file(f'inputs/decontamination-{index}.txt', root / filename)
    recipe['processing']['benchmark_files'] = [f'inputs/decontamination-{i}.txt'
        for i in range(len(config.processing.benchmark_files))]
    recipe['output_dir'] = 'artifacts/training'
    files['configs/job.yaml'] = yaml.safe_dump(recipe, sort_keys=False).encode()
    if sum(map(len, files.values())) > cap:
        raise ValueError('Uncompressed training bundle exceeds max_bundle_bytes')
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, content in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return output.getvalue()


def training_user_data(settings):
    if not settings.get('artifact_bucket') or not settings.get('instance_profile_name'):
        raise ValueError('AWS training requires artifact_bucket and instance_profile_name')
    bucket = settings['artifact_bucket']
    prefix = settings.get('_artifact_prefix', 'modelchef-training/PLAN')
    source = shlex.quote(f's3://{bucket}/{prefix}/source.tar.gz')
    result = shlex.quote(f's3://{bucket}/{prefix}/results.tar.gz')
    minutes = int(settings['max_runtime_minutes'])
    checksum = settings.get('_bundle_sha256', '0' * 64)
    if len(checksum) != 64 or any(char not in '0123456789abcdef' for char in checksum):
        raise ValueError('Invalid bundle checksum')
    max_result = int(settings.get('max_result_bytes', 512_000_000))
    if max_result <= 0:
        raise ValueError('max_result_bytes must be positive')
    return f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/modelchef-training.log /dev/console) 2>&1
systemd-run --unit=modelchef-deadline --on-active={minutes}m /sbin/shutdown -h now
finish() {{
  code=$?; trap - EXIT; set +e
  if [ -d /opt/modelchef-training/artifacts ]; then
    tar -czf /tmp/results.tar.gz -C /opt/modelchef-training artifacts
    size=$(stat -c %s /tmp/results.tar.gz)
    if [ "$size" -le {max_result} ]; then
      timeout 45 aws s3 cp /tmp/results.tar.gz {result} --sse AES256 --cli-connect-timeout 5 --cli-read-timeout 10 || code=1
    else code=1; fi
  fi
  echo "MODELCHEF_RESULT=$code"
  /sbin/shutdown -h now
}}
trap finish EXIT
set +u
source /opt/pytorch/bin/activate
set -u
command -v aws >/dev/null || python -m pip install awscli
mkdir -p /opt/modelchef-training
cd /opt/modelchef-training
timeout 60 aws s3 cp {source} /tmp/source.tar.gz --cli-connect-timeout 5 --cli-read-timeout 10
echo '{checksum}  /tmp/source.tar.gz' | sha256sum -c -
tar -xzf /tmp/source.tar.gz
python -m pip install '.[pipeline]'
python -m modelchef.cli doctor
python -m modelchef.cli pipeline configs/job.yaml --run
'''


def training_preflight(root, settings):
    """Actually exercise bounded data/model training before the AWS lifecycle begins."""
    import subprocess
    import sys
    root = Path(root)
    config = RunConfig.model_validate(yaml.safe_load((root / settings['pipeline_config']).read_text()))
    config.name = 'aws-local-preflight'
    config.output_dir = 'artifacts/aws-training-preflight/' + uuid.uuid4().hex
    config.data.max_documents = min(config.data.max_documents, 64)
    config.data.max_tokens = min(config.data.max_tokens, 16000)
    config.training.accelerator, config.training.precision = 'cpu', 'fp32'
    config.training.compile, config.training.allow_tf32 = False, False
    config.training.optimizer_kernel = 'default'
    config.model.attention_kernel, config.model.custom_swiglu_path = 'math', None
    config.training.steps, config.training.warmup_steps = 2, 0
    config.training.micro_batch_size = config.training.gradient_accumulation = 1
    config.training.max_runtime_seconds = 60
    config.evaluation.benchmark, config.evaluation.batches = 'none', 1
    config.evaluation.generate_tokens = 1
    if config.stage.kind == 'pretrain':
        config.model.layers, config.model.hidden_size = 1, 64
        config.model.attention_heads, config.model.kv_heads = 4, 2
        config.model.intermediate_size, config.model.context_length = 128, 64
    path = root / 'artifacts/aws-training-preflight.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelchef.cli', 'pipeline', str(path), '--run'],
                   cwd=root, check=True, timeout=300)
    reports = list((root / config.output_dir).glob('*/report.json'))
    if len(reports) != 1 or json.loads(reports[0].read_text())['status'] != 'completed':
        raise RuntimeError('Local AWS training preflight did not produce a completed run; no instance launched')
    return {'passed': True, 'recipe': str(path), 'steps': 2,
            'run_dir': str(reports[0].parent)}
