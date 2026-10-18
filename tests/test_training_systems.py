import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from modelchef.run_config import RunConfig
from modelchef.training_memory import memory_plan, parameter_layout


@pytest.mark.parametrize('tied,model_type,bias', [
    (True, 'decoder_rope', False), (False, 'decoder_rope', True),
    (True, 'decoder_moe', False)])
def test_parameter_count_matches_real_model(tied, model_type, bias):
    pytest.importorskip('torch')
    from modelchef.engines import create_model
    config = RunConfig(model={'layers': 2, 'hidden_size': 32, 'attention_heads': 4,
        'kv_heads': 2, 'intermediate_size': 64, 'tie_embeddings': tied,
        'type': model_type, 'bias': bias})
    model = create_model(config.model, 300)
    assert parameter_layout(config.model, 300)['parameters'] == sum(p.numel() for p in model.parameters())


def test_memory_accounting_and_cost_do_not_claim_fit():
    config = RunConfig(training={'precision': 'fp32'})
    plan = memory_plan(config, gpu_count=4, memory_bytes=10, hourly_usd=2,
                       measured_tokens_per_second=1000, remaining_tokens=3600000)
    p = plan['parameters']
    assert plan['strategies']['ddp']['persistent_state_bytes_per_gpu'] == p*16
    assert plan['strategies']['fsdp2']['persistent_state_bytes_per_gpu'] == ((p+3)//4)*16
    assert plan['strategies']['ddp']['persistent_state_exceeds_capacity']
    assert not plan['strategies']['fsdp2']['fit_verified']
    assert plan['cost_from_measurement']['remaining_compute_usd'] == 2
    assert plan['cost_from_measurement']['usd_per_million_tokens'] == pytest.approx(2/3.6)
    assert plan['global_batch_sequences'] == 16


@pytest.mark.parametrize('kwargs', [{'gpu_count': 0}, {'gpu_count': True}, {'hourly_usd': -1},
    {'memory_bytes': float('nan')}, {'remaining_tokens': -1}, {'measured_tokens_per_second': 0}])
def test_invalid_planning_input(kwargs):
    with pytest.raises(ValueError):
        memory_plan(RunConfig(), **kwargs)


@pytest.mark.parametrize('update', [
    {'model': {'backend': 'huggingface', 'pretrained_path': '/unused'}},
    {'model': {'type': 'decoder_moe'}}, {'training': {'precision': 'fp16'}},
    {'training': {'accelerator': 'cpu'}}, {'training': {'compile': True}},
    {'training': {'optimizer': 'adamw8bit'}}, {'model': {'quantization': 'fp4_fake'}}])
def test_fsdp_rejects_unsupported_before_execution(update):
    data = RunConfig().model_dump()
    data['training']['distributed'] = 'fsdp2'
    for section, changes in update.items():
        data[section].update(changes)
    with pytest.raises(ValueError):
        RunConfig.model_validate(data)


def test_phase_metrics_disabled_and_cpu():
    pytest.importorskip('torch')
    from modelchef.phase_metrics import PhaseMetrics
    disabled = PhaseMetrics('cuda')
    with disabled.measure('not_measured'):
        pass  # Must not touch CUDA on a CPU-only installation.
    assert disabled.rows == {}
    phases = PhaseMetrics('cpu', True)
    with pytest.raises(RuntimeError), phases.measure('failure'):
        raise RuntimeError('expected')
    assert phases.rows['failure']['calls'] == 1
    assert phases.rows['failure']['peak_allocated_bytes'] is None


def test_phase_metrics_cuda():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('CUDA phase measurements require a real GPU')
    from modelchef.phase_metrics import PhaseMetrics
    phases = PhaseMetrics('cuda', True)
    with phases.measure('allocation'):
        value = torch.ones(1024, 1024, device='cuda')
    assert phases.rows['allocation']['peak_allocated_bytes'] >= value.numel()*value.element_size()
    assert phases.rows['allocation']['peak_reserved_bytes'] >= phases.rows['allocation']['peak_allocated_bytes']
    del value


def test_retention_preserves_relative_resume_and_incomplete_directories(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from modelchef.sharded_checkpoint import _prune_sharded
    root = tmp_path / 'checkpoints'
    for step in range(1, 5):
        path = root / f'step-{step:06d}.dcp'
        path.mkdir(parents=True)
        (path / 'manifest.json').write_text('{}')
    incomplete = root / 'step-000005.dcp.incomplete-partial'
    incomplete.mkdir()
    monkeypatch.chdir(tmp_path)
    _prune_sharded(root, 2, protected=[Path('checkpoints/step-000001.dcp')])
    assert (root / 'step-000001.dcp').is_dir()
    assert not (root / 'step-000002.dcp').exists()
    assert (root / 'step-000003.dcp').is_dir() and (root / 'step-000004.dcp').is_dir()
    assert incomplete.is_dir()
    for step in [999999, 1000000]:
        path = root / f'step-{step:06d}.dcp'
        path.mkdir()
        (path / 'manifest.json').write_text('{}')
    _prune_sharded(root, 1, protected=[Path('checkpoints/step-000001.dcp')])
    assert (root / 'step-1000000.dcp').is_dir()
    assert not (root / 'step-999999.dcp').exists()


def test_host_io_group_uses_cpu_and_releases_on_failure(monkeypatch):
    pytest.importorskip('torch')
    from modelchef.sharded_checkpoint import checkpoint_io_group
    calls = []
    group = object()
    monkeypatch.setattr('modelchef.sharded_checkpoint.dist.new_group',
                        lambda **kwargs: calls.append(kwargs) or group)
    monkeypatch.setattr('modelchef.sharded_checkpoint.dist.destroy_process_group',
                        lambda value: calls.append(value))
    monkeypatch.setenv('MODELCHEF_DEADLINE_EPOCH', '0')
    with pytest.raises(RuntimeError), checkpoint_io_group() as value:
        assert value is group
        raise RuntimeError('host I/O failed')
    assert calls[0]['backend'] == 'gloo'
    assert calls[0]['timeout'].total_seconds() == 300
    assert calls[-1] is group


def test_topology_unavailable_and_driver_output(monkeypatch):
    from modelchef.hardware import gpu_topology
    monkeypatch.setattr('modelchef.hardware.shutil.which', lambda _: None)
    assert gpu_topology()['status'] == 'unavailable'
    monkeypatch.setattr('modelchef.hardware.shutil.which', lambda _: 'nvidia-smi')
    monkeypatch.setattr('modelchef.hardware.subprocess.run', lambda *a, **kw:
                        SimpleNamespace(returncode=0, stdout='GPU0 GPU1\nGPU0 X NV4'))
    assert 'NV4' in gpu_topology()['matrix']


def test_two_rank_fsdp_cpu_math_and_sharded_recovery(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip('Gloo is unavailable')
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1',
           'GLOO_SOCKET_IFNAME': 'lo' if os.name != 'nt' else '', 'USE_LIBUV': '0'}
    if os.name == 'nt':
        env.pop('GLOO_SOCKET_IFNAME')
    processes = [subprocess.Popen([sys.executable, str(root / 'tests/fsdp_checkpoint_worker.py'),
                                  str(rank), str(tmp_path)], env=env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for rank in range(2)]
    try:
        for process in processes:
            stdout, stderr = process.communicate(timeout=120)
            assert process.returncode == 0, stdout + stderr
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
    assert all(json.loads((tmp_path / 'result.json').read_text()).values())
    from modelchef.chat import checkpoint_model_state, export_model
    from modelchef.engines import load_tokenizer
    restored = tmp_path / 'restored/artifacts/training/run'
    config = RunConfig.model_validate_json((restored / 'config.json').read_text())
    state = checkpoint_model_state(restored, 'step-000001.dcp', config, load_tokenizer(restored))
    expected = torch.load(tmp_path / 'expected-model.pt', weights_only=True)
    assert state.keys() == expected.keys()
    assert all(torch.equal(state[key], value) for key, value in expected.items())
    exported = export_model(restored, 'step-000001.dcp', tmp_path / 'export')
    assert (exported / 'model.safetensors').is_file()
    original_config = (restored / 'config.json').read_bytes()
    (restored / 'config.json').write_bytes(original_config + b'\n')
    with pytest.raises(ValueError, match='identity changed'):
        checkpoint_model_state(restored, 'step-000001.dcp', config, load_tokenizer(restored))
    (restored / 'config.json').write_bytes(original_config)
    from modelchef.sharded_checkpoint import read_manifest
    path = tmp_path / 'guest/artifacts/training/run/checkpoints/step-000001.dcp'
    with pytest.raises(ValueError, match='world size'):
        read_manifest(path, 3)
    (path / 'rank-0.pt').write_bytes(b'corrupted')
    with pytest.raises(ValueError, match='changed'):
        read_manifest(path, 2)


def test_profiled_pipeline_preserves_test_isolation(tmp_path, monkeypatch):
    pytest.importorskip('torch')
    from modelchef.pipeline import run_pipeline
    source = tmp_path / 'corpus.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Example {i} has enough text to train. ' * 8})
                             + '\n' for i in range(40)))
    config = RunConfig(name='profile-check', output_dir=str(tmp_path / 'runs'), explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(source), 'max_documents': 40,
              'max_tokens': 6000}, processing={'min_chars': 1, 'near_dedup': False},
        tokenizer={'vocab_size': 300}, model={'layers': 1, 'hidden_size': 16,
            'intermediate_size': 32, 'attention_heads': 2, 'kv_heads': 1, 'context_length': 16},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'profile_phases': True,
            'steps': 2, 'warmup_steps': 0, 'micro_batch_size': 1, 'gradient_accumulation': 1},
        evaluation={'evaluate_test': False, 'benchmark': 'none', 'generate_tokens': 1, 'batches': 1})
    result = run_pipeline(config, tmp_path)
    assert result['status'] == 'completed'
    assert result['test'] is None and result['test_evaluation'] == 'withheld'
    phases = result['phase_metrics']
    assert {'model_initialization', 'optimizer_initialization', 'forward', 'backward',
            'optimizer_update', 'checkpoint', 'initial_validation', 'final_evaluation',
            'checkpoint_reload'} <= phases.keys()
    assert phases['optimizer_update']['calls'] == 2
    assert all(row['seconds'] >= 0 and row['peak_allocated_bytes'] is None for row in phases.values())

    # FSDP's preparation must not allocate the full model on GPU or execute GPU-only kernels.
    config.training.distributed = 'fsdp2'
    config.training.accelerator = 'cuda'
    config.training.optimizer_kernel = 'fused'
    config.model.attention_kernel = 'flash'
    monkeypatch.setenv('MODELCHEF_RECOVERY_S3_URI', 's3://bucket/test')
    monkeypatch.setattr('modelchef.recovery.checkpoint_to_s3',
                        lambda *args: pytest.fail('CPU preparation advertised a non-DCP recovery point'))
    prepared = run_pipeline(config, tmp_path, prepare_only=True)
    assert prepared['status'] == 'prepared'
    assert prepared['environment']['device'] == 'cpu'
    assert prepared['execution']['preparation_only']


def test_fsdp_guest_uses_torchrun():
    from modelchef.aws_guest import guest_plan
    config = RunConfig(training={'distributed': 'fsdp2', 'precision': 'bf16'})
    plan = guest_plan(config, cloud_extra='')
    assert 'torch.distributed.run' in plan['command']
    assert plan['execution_policy']['training']['distributed'] == 'fsdp2'


def test_memory_cli_is_allocation_free(tmp_path, monkeypatch, capsys):
    import yaml

    from modelchef.cli import main
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(RunConfig().model_dump(mode='json')))
    monkeypatch.setattr(sys, 'argv', ['modelchef', 'plan-memory', str(path), '--gpu-count', '4'])
    # Planning must not invoke training or GPU discovery unless topology was explicitly requested.
    monkeypatch.setattr('modelchef.cli.detect_hardware', lambda: pytest.fail('Allocated hardware'))
    main()
    result = json.loads(capsys.readouterr().out)
    assert result['gpu_count'] == 4 and not result['strategies']['fsdp2']['fit_verified']


def test_fsdp_two_cuda_training_and_resume(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2 or torch.version.hip:
        pytest.skip('Requires two NVIDIA GPUs; CPU checkpoint tests are not CUDA validation')
    import yaml
    source = tmp_path / 'corpus.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Example {i}: several words for training. ' * 12})
                             + '\n' for i in range(40)))
    config = RunConfig(name='fsdp-check', output_dir=str(tmp_path / 'runs'), explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(source), 'max_documents': 40,
              'max_tokens': 6000}, processing={'min_chars': 1, 'near_dedup': False},
        tokenizer={'vocab_size': 300}, model={'layers': 2, 'hidden_size': 16,
            'intermediate_size': 32, 'attention_heads': 2, 'kv_heads': 1, 'context_length': 16},
        training={'distributed': 'fsdp2', 'accelerator': 'cuda', 'precision': 'fp32',
            'profile_phases': True, 'steps': 2, 'warmup_steps': 0, 'micro_batch_size': 1,
            'gradient_accumulation': 2, 'checkpoint_every': 1},
        evaluation={'evaluate_test': False, 'benchmark': 'none', 'batches': 1})
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '--module', '--', 'modelchef.cli', 'pipeline', str(path), '--run', '--quiet']
    first = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                           timeout=180, check=False)
    assert first.returncode == 0, first.stdout + first.stderr
    folder = next((tmp_path / 'runs').iterdir())
    report = json.loads((folder / 'report.json').read_text())
    assert report['checkpoint_reload_verified'] and report['test'] is None
    resume = folder / 'checkpoints/step-000001.dcp'
    second = subprocess.run(command + ['--resume', str(resume)], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=180, check=False)
    assert second.returncode == 0, second.stdout + second.stderr
    assert (resume / 'manifest.json').exists()


def test_shared_preview_reports_unknown_and_unsupported_without_allocation(tmp_path, monkeypatch):
    from modelchef.planning import memory_preview
    monkeypatch.setattr('modelchef.engines.create_model', lambda *a, **kw: pytest.fail('Allocated a model'))
    result = memory_preview({}, tmp_path)
    assert result['status'] == 'estimated' and not result['gpu_count_verified']
    assert any('illustrative' in item for item in result['limitations'])
    assert memory_preview({'modality': 'vision'}, tmp_path)['status'] == 'unavailable'
    (tmp_path / 'stage.yaml').write_text('{}')
    stages = memory_preview({'stages': [{'name': 'base', 'recipe': 'stage.yaml'}]}, tmp_path, gpu_count=2)
    assert stages['stages']['base']['gpu_count'] == 2
    assert stages['stages']['base']['execution_world_size'] == 1
    assert stages['stages']['base']['global_batch_sequences'] == result['global_batch_sequences']
    distributed = memory_preview({'training': {'distributed': 'ddp'}}, tmp_path)
    assert distributed['execution_world_size'] is None
    assert distributed['global_batch_input_tokens'] is None
