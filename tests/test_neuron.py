import pytest

from modelchef.neuron import NeuronRecipe, create_decoder, plan


def recipe(**changes):
    return NeuronRecipe(tokens_path='unused.bin', vocab_size=260, sequence_length=8,
                        model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4,
                               'kv_heads': 2, 'intermediate_size': 64, 'context_length': 8},
                        **changes)


def test_neuron_plan_does_not_import_sdk_or_launch_cloud():
    result = plan(recipe(), 'train')
    assert result['launches_aws'] is False
    assert result['shape'] == [1, 8]


def test_portable_decoder_matches_native_logits_and_gradients():
    torch = pytest.importorskip('torch')
    from modelchef.model import Decoder
    config = recipe()
    native = Decoder(config.model, config.vocab_size)
    portable = create_decoder(config)
    portable.load_state_dict(native.state_dict())
    tokens = torch.randint(260, (2, 8))
    expected, actual = native(tokens), portable(tokens)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    for a, b in zip(native.parameters(), portable.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, atol=1e-6, rtol=1e-4)


@pytest.mark.parametrize('field,value', [('custom_swiglu_path', '/cuda'),
                                      ('attention_kernel', 'flash'),
                                      ('quantization', 'fp4_fake'),
                                      ('gradient_checkpointing', True)])
def test_incompatible_execution_is_explicit(field, value):
    base = recipe().model_dump()
    base['model'][field] = value
    with pytest.raises(ValueError, match='Neuron prototype'):
        NeuronRecipe.model_validate(base)


def test_cli_preview_without_token_file_or_neuron_sdk(monkeypatch, capsys):
    import json

    from modelchef.cli import main
    monkeypatch.setattr('sys.argv', ['modelchef', 'neuron', 'configs/neuron-native.yaml'])
    main()
    assert json.loads(capsys.readouterr().out)['backend'] == 'neuronx'


def test_training_resume_with_mocked_xla_matches_uninterrupted(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import sys
    import types

    import numpy as np

    from modelchef import neuron

    monkeypatch.setattr(neuron, '_sdk', lambda: object())
    xm = types.ModuleType('torch_xla.core.xla_model')
    xm.xla_device = lambda: torch.device('cpu')
    xm.optimizer_step = lambda optimizer: optimizer.step()
    count = []
    xm.mark_step = lambda: count.append(1)
    xm.save = torch.save
    package = types.ModuleType('torch_xla')
    core = types.ModuleType('torch_xla.core')
    package.core, core.xla_model = core, xm
    for name, value in [('torch_xla', package), ('torch_xla.core', core),
                        ('torch_xla.core.xla_model', xm)]:
        monkeypatch.setitem(sys.modules, name, value)
    data = tmp_path / 'tokens.bin'
    np.arange(100, dtype='<u4').tofile(data)
    config = recipe().model_copy(update={'tokens_path': str(data),
                                        'output_dir': str(tmp_path / 'run'),
                                        'steps': 2, 'checkpoint_every': 1})
    neuron.train(config)
    final = tmp_path / 'run' / 'step-000002.pt'
    expected = torch.load(final, weights_only=True)
    final.unlink()
    neuron.train(config, tmp_path / 'run' / 'step-000001.pt')
    resumed = torch.load(final, weights_only=True)
    assert len(count) == 3
    for key in expected['model']:
        torch.testing.assert_close(expected['model'][key], resumed['model'][key], rtol=0, atol=0)
    config.tokens_path = str(tmp_path / 'other.bin')
    np.arange(101, dtype='<u4').tofile(config.tokens_path)
    with pytest.raises(ValueError, match='identical'):
        neuron.train(config, final)


def test_compilation_writes_fixed_shape_artifact_manifest(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    import json
    import types

    from modelchef import neuron

    config = recipe().model_copy(update={'output_dir': str(tmp_path / 'output')})
    model = neuron.create_decoder(config)
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model': model.state_dict(), 'recipe_sha256': neuron.fingerprint(config)}, checkpoint)
    calls = []

    def trace(model, inputs, **options):
        calls.append((tuple(inputs.shape), options))
        return torch.jit.trace(model, inputs, check_trace=False)

    monkeypatch.setattr(neuron, '_sdk', lambda: types.SimpleNamespace(trace=trace))
    result = neuron.compile_model(config, checkpoint)
    assert calls[0][0] == (1, 8)
    assert result['numerical_validation'] == 'not_run'
    folder = tmp_path / 'output' / 'compiled'
    assert json.loads((folder / 'manifest.json').read_text())['artifact_sha256']
    reloaded = torch.jit.load(str(folder / 'model.pt'))
    tokens = torch.randint(260, (1, 8))
    torch.testing.assert_close(model(tokens), reloaded(tokens))
    with pytest.raises(ValueError, match='never overwritten'):
        neuron.compile_model(config, checkpoint)


def test_neuron_cloud_bundle_and_shutdown_do_not_install_cuda(tmp_path):
    pytest.importorskip('torch')
    import io
    import tarfile

    import numpy as np
    import yaml

    from modelchef.aws_training import build_bundle, training_preflight, training_user_data

    (tmp_path / 'src/modelchef').mkdir(parents=True)
    (tmp_path / 'pyproject.toml').write_text('test')
    (tmp_path / 'README.md').write_text('test')
    np.arange(100, dtype='<u4').tofile(tmp_path / 'tokens.bin')
    config = recipe().model_copy(update={'tokens_path': 'tokens.bin', 'steps': 4,
                                        'checkpoint_every': 2})
    (tmp_path / 'job.yaml').write_text(yaml.safe_dump(config.model_dump(mode='json')))
    settings = {'job_backend': 'neuron_probe', 'pipeline_config': 'job.yaml',
                'instance_type': 'trn1.2xlarge', 'max_runtime_minutes': 60,
                'storage_gb': 100, 'artifact_bucket': 'test', 'instance_profile_name': 'test'}
    bundle = build_bundle(tmp_path, settings)
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        assert 'inputs/tokens.bin' in archive.getnames()
    assert training_preflight(tmp_path, settings)['cpu_forward_backward']
    script = training_user_data(settings)
    assert '/opt/aws_neuronx_venv_pytorch_2_9/bin/activate' in script
    assert 'modelchef.aws_neuron' in script
    assert 'systemd-run --unit=modelchef-deadline' in script
    assert 'trap finish EXIT' in script
    assert 'modelchef.aws_guest' not in script
    assert '/opt/pytorch/bin/activate' not in script
