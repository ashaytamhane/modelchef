"""Bounded training bundles and guest scripts for the existing Spot lifecycle."""
import io
import json
import shlex
import tarfile
import uuid
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig


def build_bundle(root, settings, *, raw_override=None, linked_stage=False):
    if settings.get('job_backend') == 'neuron_probe':
        from modelchef.aws_neuron import build_bundle as neuron_bundle
        return neuron_bundle(root, settings)
    root = Path(root).resolve()
    raw = raw_override if raw_override is not None else yaml.safe_load((root / settings['pipeline_config']).read_text())
    if 'stages' in raw:
        from modelchef.aws_workflow import build_workflow_bundle
        return build_workflow_bundle(root, settings)
    if 'modality' in raw:
        return build_multimodal_bundle(root, settings, raw)
    config = RunConfig.model_validate(raw)
    from modelchef.model_references import resolve_references
    config, _ = resolve_references(config, root)
    config.validate_execution()
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

    source_root = Path(settings.get('_source_root', root)).resolve()
    for path in sorted((source_root / 'src/modelchef').glob('*.py')):
        add_file(path.relative_to(source_root).as_posix(), path)
    add_file('pyproject.toml', source_root / 'pyproject.toml')
    if (source_root / 'README.md').is_file():
        add_file('README.md', source_root / 'README.md')
    recipe = config.model_dump(mode='json')
    pretrained = {}
    kernels = {}

    def package_kernel(model):
        if not model['custom_swiglu_path']:
            return
        directory = (root / model['custom_swiglu_path']).resolve()
        if directory not in kernels:
            relative = 'inputs/custom-kernel' + (f'-{len(kernels)}' if kernels else '')
            if not (directory / 'kernel.py').is_file():
                raise ValueError('Custom CUDA directory requires kernel.py')
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in {'.py', '.cpp', '.cu', '.h', '.cuh'}:
                    add_file(relative + '/' + path.name, path)
            kernels[directory] = relative
        model['custom_swiglu_path'] = kernels[directory]

    def package_pretrained(model):
        if model['backend'] != 'huggingface':
            return
        directory = (root / model['pretrained_path']).resolve()
        if directory not in pretrained:
            relative = f'inputs/pretrained-{len(pretrained)}'
            if not (directory / 'config.json').is_file() or not any(directory.glob('*.safetensors')):
                raise ValueError('Managed pretrained bundles require local config and safetensors weights')
            for path in sorted(directory.iterdir()):
                if path.is_file() and path.suffix in {'.json', '.safetensors', '.txt', '.model', '.jinja'}:
                    add_file(relative + '/' + path.name, path)
            # Index references must stay in the allowlisted model directory.
            for path in directory.glob('*.index.json'):
                index = json.loads(path.read_text(encoding='utf-8'))
                for filename in set(index.get('weight_map', {}).values()):
                    if relative + '/' + filename not in files:
                        raise ValueError('Pretrained index references a missing or excluded shard')
            pretrained[directory] = relative
        model['pretrained_path'] = pretrained[directory]

    package_pretrained(recipe['model'])
    package_kernel(recipe['model'])
    local_inputs = [(root / filename).resolve() for filename in
                    config.data.local_paths or ([config.data.local_path] if config.data.local_path else [])]
    prepared = settings.get('_prepared_inputs', set())
    remote_prepared = bool(local_inputs) and all(str(path) in prepared for path in local_inputs)
    if config.data.provider == 'local_jsonl' and not remote_prepared:
        rows, read_bytes = [], 0
        index = 0
        remaining = cap - sum(map(len, files.values()))
        for filename in config.data.local_paths or [config.data.local_path]:
            with (root / filename).open('rb') as source:
                while len(rows) < config.data.max_documents:
                    line = source.readline(min(config.data.max_record_bytes,
                        config.data.max_download_bytes - read_bytes, remaining - read_bytes) + 1)
                    if not line:
                        break
                    read_bytes += len(line)
                    if read_bytes > config.data.max_download_bytes:
                        raise ValueError('Local training input exceeds byte budget')
                    if read_bytes > remaining:
                        raise ValueError('Uncompressed training bundle exceeds max_bundle_bytes')
                    if len(line) > config.data.max_record_bytes:
                        raise ValueError('Local training record exceeds max_record_bytes')
                    if not line.strip():
                        continue
                    if index >= config.data.offset:
                        rows.append(line if line.endswith(b'\n') else line + b'\n')
                    index += 1
            if len(rows) >= config.data.max_documents:
                break
        files['inputs/data.jsonl'] = b''.join(rows)
        recipe['data'].update(local_path='inputs/data.jsonl', local_paths=[], offset=0)
    for field in ['parent_run', 'teacher_run']:
        if not getattr(config.stage, field):
            continue
        if linked_stage and getattr(config.stage, field) == 'resolved-at-execution':
            continue
        parent = (root / getattr(config.stage, field)).resolve()
        saved = RunConfig.model_validate_json((parent / 'config.json').read_text())
        report = json.loads((parent / 'report.json').read_text())
        if report['status'] != 'completed':
            raise ValueError('AWS parent run must be complete')
        relative = f'inputs/{field}'
        checkpoint = f"checkpoints/step-{report['optimizer_steps']:06d}.pt"
        for name in ['report.json', 'tokenizer.json', checkpoint]:
            path = parent / name
            add_file(f'{relative}/{name}', path)
        if (parent / 'tokenizer-roles.json').is_file():
            add_file(f'{relative}/tokenizer-roles.json', parent / 'tokenizer-roles.json')
        metadata = saved.model_dump(mode='json')
        package_pretrained(metadata['model'])
        package_kernel(metadata['model'])
        files[f'{relative}/config.json'] = json.dumps(metadata).encode()
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


def build_multimodal_bundle(root, settings, raw):
    from modelchef.multimodal import model_manifest
    from modelchef.multimodal_config import MultimodalConfig
    from modelchef.multimodal_data import read_manifest
    config = MultimodalConfig.model_validate(raw)
    cap = int(settings.get('max_bundle_bytes', 64_000_000))
    if cap <= 0:
        raise ValueError('max_bundle_bytes must be positive')
    files = {}
    def add(name, path):
        if name in files:
            return
        remaining = cap - sum(map(len, files.values()))
        if path.stat().st_size > remaining:
            raise ValueError('Multimodal bundle exceeds max_bundle_bytes')
        with path.open('rb') as source:
            value = source.read(remaining + 1)
        if len(value) > remaining:
            raise ValueError('Multimodal bundle exceeds max_bundle_bytes')
        files[name] = value
    for path in sorted((root / 'src/modelchef').glob('*.py')):
        add(path.relative_to(root).as_posix(), path)
    for name in ['pyproject.toml', 'README.md']:
        if (root / name).is_file():
            add(name, root / name)
    records, _ = read_manifest(config, root)
    field = 'image' if config.modality == 'vision' else 'audio'
    rows = []
    for record in records:
        path = Path(record[field])
        name = f"media/{record['media_sha256']}{path.suffix}"
        add('inputs/' + name, path)
        row = {key: value for key, value in record.items() if key != 'media_sha256'}
        row[field] = name
        rows.append(row)
    model = (root / config.model.path).resolve()
    for name in model_manifest(model, config.model.max_model_bytes):
        add('inputs/model/' + name, model / name)
    recipe = config.model_dump(mode='json')
    recipe['data']['manifest'] = 'inputs/media.jsonl'
    recipe['model']['path'] = 'inputs/model'
    recipe['output_dir'] = 'artifacts/multimodal'
    files['inputs/media.jsonl'] = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    files['configs/job.yaml'] = yaml.safe_dump(recipe, sort_keys=False).encode()
    if sum(map(len, files.values())) > cap:
        raise ValueError('Multimodal bundle exceeds max_bundle_bytes')
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
    result_mode = settings.get('result_mode', 'full')
    if result_mode not in {'full', 'metadata'}:
        raise ValueError('result_mode must be full or metadata')
    archive_command = ('python -m modelchef.aws_results /opt/modelchef-training /tmp/results.tar.gz'
                       if result_mode == 'metadata' else
                       'tar -czf /tmp/results.tar.gz -C /opt/modelchef-training artifacts')
    if max_result <= 0:
        raise ValueError('max_result_bytes must be positive')
    script = f'''#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/modelchef-training.log /dev/console) 2>&1
export MODELCHEF_DEADLINE_EPOCH=$(($(date +%s) + {minutes} * 60 - 600))
systemd-run --unit=modelchef-deadline --on-active={minutes}m /sbin/shutdown -h now
finish() {{
  code=$?; trap - EXIT; set +e
  if [ -d /opt/modelchef-training/artifacts ]; then
    {archive_command}
    size=$(stat -c %s /tmp/results.tar.gz)
    if [ "$size" -le {max_result} ]; then
      timeout 300 aws s3 cp /tmp/results.tar.gz {result} --sse AES256 --cli-connect-timeout 5 --cli-read-timeout 30 || code=1
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
# Guest preparation preserves the original recipe's optimization policy.
export MODELCHEF_RECOVERY_S3_URI={shlex.quote(f's3://{bucket}/{prefix}/recovery')}
python -m modelchef.aws_guest configs/job.yaml
'''
    if settings.get('job_backend') == 'neuron_probe':
        script = script.replace('source /opt/pytorch/bin/activate',
                                'source /opt/aws_neuronx_venv_pytorch_2_9/bin/activate')
        script = script.replace('python -m modelchef.cli doctor',
                                "neuron-ls\npython -c 'import torch, torch_neuronx; print(torch.__version__)'")
        script = script.replace('python -m modelchef.aws_guest configs/job.yaml',
                                'python -m modelchef.aws_neuron configs/job.yaml')
    return script


def training_preflight(root, settings):
    """Actually exercise bounded data/model training before the AWS lifecycle begins."""
    import subprocess
    import sys
    if settings.get('job_backend') == 'neuron_probe':
        from modelchef.aws_neuron import preflight
        return preflight(root, settings)
    root = Path(root)
    raw = yaml.safe_load((root / settings['pipeline_config']).read_text())
    if 'stages' in raw:
        from modelchef.aws_workflow import workflow_preflight
        return workflow_preflight(root, settings)
    if 'modality' in raw:
        return multimodal_preflight(root, raw)
    config = RunConfig.model_validate(raw)
    config.name = 'aws-local-preflight'
    config.output_dir = 'artifacts/aws-training-preflight/' + uuid.uuid4().hex
    config.data.max_documents = min(config.data.max_documents, 64)
    config.data.max_tokens = min(config.data.max_tokens, 16000)
    config.training.accelerator, config.training.precision = 'cpu', 'fp32'
    config.training.distributed = 'none'
    if config.model.quantization == 'nvfp4':
        config.model.quantization = 'none'
    if config.model.peft.method == 'qlora':
        config.model.peft.method = 'lora'
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
            'scope': 'CPU data/objective/recovery probe; GPU optimization validation runs on guest',
            'run_dir': str(reports[0].parent)}


def multimodal_preflight(root, raw):
    import subprocess
    import sys

    from modelchef.multimodal_config import MultimodalConfig
    config = MultimodalConfig.model_validate(raw)
    config.name = 'aws-multimodal-preflight'
    config.output_dir = 'artifacts/aws-training-preflight/' + uuid.uuid4().hex
    config.data.max_records = min(config.data.max_records, 16)
    config.training.accelerator, config.training.precision = 'cpu', 'fp32'
    config.training.compile, config.training.allow_tf32 = False, False
    config.training.optimizer_kernel = 'default'
    config.model.attention_kernel = 'math'
    config.training.steps, config.training.warmup_steps = 2, 0
    config.training.gradient_accumulation = 1
    config.training.max_runtime_seconds = 120
    config.evaluation.max_examples, config.evaluation.max_new_tokens = 1, 1
    path = root / 'artifacts/aws-multimodal-preflight.yaml'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelchef.cli', 'multimodal', str(path), '--run'],
                   cwd=root, check=True, timeout=300)
    reports = list((root / config.output_dir).glob('*/report.json'))
    if len(reports) != 1 or json.loads(reports[0].read_text())['status'] != 'completed':
        raise RuntimeError('Multimodal local preflight did not complete; no instance launched')
    return {'passed': True, 'recipe': str(path), 'steps': 2, 'run_dir': str(reports[0].parent),
            'scope': 'CPU multimodal data/objective/recovery probe; actual GPU policy validated on guest'}
