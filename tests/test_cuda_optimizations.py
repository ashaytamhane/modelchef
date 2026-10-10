"""Real-device tests; collected on CPU CI and exercised on compatible GPU hosts."""
import math
import os
import shutil
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA device required')


@pytest.mark.parametrize('policy', ['compiled_fused', 'flash_fp16', 'custom_cuda'])
def test_optimized_training_and_checkpoint_recovery(tmp_path, monkeypatch, policy):
    from modelchef.pipeline import run_pipeline
    from modelchef.run_config import RunConfig
    root = Path(__file__).resolve().parents[1]
    values = yaml.safe_load((root / 'configs/pretrain-demo.yaml').read_text())
    values['data']['local_path'] = str(root / values['data']['local_path'])
    values['data']['storage'] = 'disk'
    values['training'].update(accelerator='cuda', precision='fp16' if policy == 'flash_fp16' else 'bf16',
                              deterministic=False, steps=2, checkpoint_every=1,
                              max_runtime_seconds=300)
    if values['training']['precision'] == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 device required for this policy')
    if policy == 'compiled_fused':
        pytest.importorskip('triton', reason='CUDA Inductor requires the optional Triton runtime')
        values['training'].update(compile=True, optimizer_kernel='fused', allow_tf32=True)
    elif policy == 'flash_fp16':
        values['model']['attention_kernel'] = 'flash'
        values['training']['optimizer_kernel'] = 'foreach'
    else:
        pytest.importorskip('ninja', reason='Custom CUDA tests require the cuda-kernels extra')
        monkeypatch.setenv('PATH', str(Path(sys.executable).parent) + os.pathsep + os.environ['PATH'])
        toolkit = Path(torch.__file__).resolve().parents[1] / 'nvidia/cu13'
        if not (toolkit / 'bin/nvcc').is_file() and not shutil.which('nvcc'):
            pytest.skip('Custom CUDA tests require a matching local CUDA compiler')
        if (toolkit / 'bin/nvcc').is_file():
            monkeypatch.setenv('CUDA_HOME', str(toolkit))
        values['model']['custom_swiglu_path'] = str(root / 'kernels/swiglu')
    config = RunConfig.model_validate(values)
    monkeypatch.chdir(tmp_path)
    result = run_pipeline(config, tmp_path)
    assert result['status'] == 'completed'
    assert result['checkpoint_recovery'] == 'passed'
    assert math.isfinite(result['validation']['loss'])
    assert result['performance']['tokens_per_second'] > 0
    assert result['performance']['peak_cuda_allocated_bytes'] > 0
