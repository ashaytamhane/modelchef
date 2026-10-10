import sys
from types import SimpleNamespace

import pytest
import yaml

from modelchef.aws_guest import guest_plan, prepare_and_run
from modelchef.run_config import RunConfig


def test_guest_preserves_compiler_attention_and_optimizer_policy():
    config = RunConfig(model={'attention_kernel': 'flash', 'custom_swiglu_path': 'inputs/kernel'},
                       training={'compile': True, 'optimizer_kernel': 'fused', 'allow_tf32': True})
    plan = guest_plan(config)
    assert plan['execution_policy']['training']['compile']
    assert plan['execution_policy']['training']['optimizer_kernel'] == 'fused'
    assert plan['execution_policy']['model']['attention_kernel'] == 'flash'
    assert {'cuda-kernels', 'cuda13-toolchain'} <= set(plan['extras'])


def test_pretrained_qlora_selects_guest_dependencies(tmp_path):
    config = RunConfig(model={'backend': 'huggingface', 'pretrained_path': str(tmp_path),
                              'peft': {'method': 'qlora'}},
                       tokenizer={'algorithm': 'pretrained'},
                       data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'},
                       stage={'kind': 'sft', 'parent_run': 'parent'})
    assert {'pretrained', 'qlora'} <= set(guest_plan(config)['extras'])


@pytest.mark.parametrize('gpus', [1, 4])
def test_guest_ddp_launch_checks_hardware_before_training(tmp_path, monkeypatch, gpus):
    path = tmp_path / 'job.yaml'
    config = RunConfig(training={'distributed': 'ddp'})
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')))
    calls = []
    monkeypatch.setattr('subprocess.run', lambda command, **kw: calls.append(command))
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(cuda=SimpleNamespace(
        is_available=lambda: True, device_count=lambda: gpus)))
    if gpus == 1:
        with pytest.raises(ValueError, match='at least two'):
            prepare_and_run(path)
        assert not any('torch.distributed.run' in command for command in calls)
    else:
        prepare_and_run(path)
        assert calls[-1][2:8] == ['torch.distributed.run', '--standalone', '--nproc-per-node=gpu',
                                  '--module', '--', 'modelchef.cli']
        assert calls[-1][-2:] == [str(path), '--run']
    assert 'https://download.pytorch.org/whl/cu130' in calls[0]
