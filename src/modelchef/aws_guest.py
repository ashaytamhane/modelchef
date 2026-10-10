"""Recipe-driven dependencies and launch policy inside a managed CUDA training guest."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig


def guest_plan(config, related_models=()):
    if hasattr(config, 'modality'):
        return {'extras': ['train', 'multimodal'],
                'command': [sys.executable, '-m', 'modelchef.cli', 'multimodal', 'configs/job.yaml', '--run'],
                'execution_policy': {'model': config.model.model_dump(mode='json'),
                                     'training': config.training.model_dump(mode='json')}}
    config.validate_execution()
    extras = ['train', 'pipeline']
    models = [config.model, *related_models]
    if any(model.backend == 'huggingface' for model in models):
        extras.append('pretrained')
    if any(model.peft.method == 'lora' for model in models):
        extras.append('peft')
    if any(model.peft.method == 'qlora' for model in models):
        extras.append('qlora')
    if any(model.quantization == 'nvfp4' for model in models):
        extras.append('nvfp4')
    if any(model.custom_swiglu_path for model in models):
        extras += ['cuda-kernels', 'cuda13-toolchain']
    command = [sys.executable, '-m', 'modelchef.cli', 'pipeline', 'configs/job.yaml', '--run']
    if config.training.distributed == 'ddp':
        command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
                   '--nproc-per-node=gpu', '--module', '--', *command[2:]]
    return {'extras': extras, 'command': command,
            'execution_policy': {'model': config.model.model_dump(mode='json'),
                                 'training': config.training.model_dump(mode='json')}}


def prepare_and_run(path):
    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if 'modality' in raw:
        from modelchef.multimodal_config import MultimodalConfig
        config, related = MultimodalConfig.model_validate(raw), []
    else:
        config = RunConfig.model_validate(raw)
        related = [RunConfig.model_validate_json((Path(folder) / 'config.json').read_text()).model
                   for folder in [config.stage.parent_run, config.stage.teacher_run] if folder]
    plan = guest_plan(config, related)
    print(json.dumps(plan, indent=2), flush=True)
    # Pin the training build instead of inheriting an arbitrary AMI PyTorch version.
    subprocess.run([sys.executable, '-m', 'pip', 'install', 'torch==2.13.0',
                    '--index-url', 'https://download.pytorch.org/whl/cu130'], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install',
                    '.[' + ','.join(plan['extras']) + ']'], check=True)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Managed GPU training requires a CUDA-compatible AMI and driver')
    if config.training.distributed == 'ddp' and torch.cuda.device_count() < 2:
        raise ValueError('DDP recipe requires an instance with at least two visible GPUs')
    if any(getattr(model, 'quantization', None) == 'nvfp4' for model in [config.model, *related]):
        from modelchef.low_precision import require_nvfp4
        require_nvfp4()
    if any(getattr(model, 'custom_swiglu_path', None) for model in [config.model, *related]):
        toolkit = Path(torch.__file__).resolve().parents[1] / 'nvidia/cu13'
        if not (toolkit / 'bin/nvcc').is_file():
            raise RuntimeError('Matching CUDA 13 compiler was not installed')
        os.environ['CUDA_HOME'] = str(toolkit)
        os.environ['PATH'] = str(toolkit / 'bin') + os.pathsep + os.environ.get('PATH', '')
    command = plan['command']
    command[command.index('configs/job.yaml')] = str(path)
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('recipe', type=Path)
    args = parser.parse_args()
    prepare_and_run(args.recipe)


if __name__ == '__main__':
    main()
