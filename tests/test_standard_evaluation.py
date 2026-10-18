"""Offline contracts for explicit standard evaluation of local checkpoints."""
import json

import pytest

from modelchef.comparison import standard_plan


@pytest.fixture
def run_folder(tmp_path, monkeypatch):
    folder = tmp_path / 'run'
    folder.mkdir()
    (folder / 'config.json').write_text(json.dumps({'training': {}, 'tokenizer': {}}))
    (folder / 'tokenizer.json').write_text('{}')
    (folder / 'checkpoints').mkdir()
    (folder / 'checkpoints/step-000010.pt').write_bytes(b'fixture')
    monkeypatch.setattr('modelchef.comparison.detect_hardware',
                        lambda: {'backend': 'cpu', 'bf16_supported': False})
    return folder


def test_native_preview_selects_checkpoint_without_export_or_download(run_folder, tmp_path):
    output = tmp_path / 'evaluation'
    result = standard_plan(run_folder, ['arc_easy'], 20, output, device='cpu',
                           checkpoint='step-000010.pt', batch_size=2, cpu_threads=3)
    assert not output.exists()
    assert result['executed'] is False
    assert result['checkpoint'] == str(run_folder / 'checkpoints/step-000010.pt')
    command = result['preparation_commands'][0]
    assert command[-2:] == ['--output', str(output / 'model')]
    assert result['environment'] == {'OMP_NUM_THREADS': '3', 'MKL_NUM_THREADS': '3'}
    command = result['command']
    assert command[command.index('--batch_size') + 1] == '2'
    assert command[command.index('--device') + 1] == 'cpu'


def test_native_requires_explicit_checkpoint_for_unfinished_run(run_folder, tmp_path):
    with pytest.raises(ValueError, match='--checkpoint'):
        standard_plan(run_folder, output=tmp_path / 'evaluation')
    with pytest.raises(ValueError, match='inside this run'):
        standard_plan(run_folder, checkpoint=tmp_path / 'unrelated.pt')


def test_evaluation_does_not_overwrite_existing_export(run_folder, tmp_path):
    output = tmp_path / 'evaluation'
    (output / 'model').mkdir(parents=True)
    with pytest.raises(ValueError, match='fresh --output'):
        standard_plan(run_folder, output=output, checkpoint='step-000010.pt')


def test_explicit_execution_exports_then_evaluates_with_thread_limit(run_folder, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr('importlib.util.find_spec', lambda name: object())
    monkeypatch.setattr('subprocess.run', lambda cmd, **kw: calls.append((cmd, kw)))
    result = standard_plan(run_folder, ['arc_easy'], 5, tmp_path / 'eval', True,
                           'cpu', checkpoint='step-000010.pt', cpu_threads=2)
    assert result['executed']
    assert len(calls) == 2
    assert 'export' in calls[0][0]
    assert 'lm_eval' in calls[1][0]
    assert all(kw['check'] and kw['env']['OMP_NUM_THREADS'] == '2' for _, kw in calls)


def test_missing_harness_fails_before_export(run_folder, tmp_path, monkeypatch):
    monkeypatch.setattr('importlib.util.find_spec', lambda name: None)
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('Must not export'))
    with pytest.raises(ImportError, match='evaluation extra'):
        standard_plan(run_folder, output=tmp_path / 'eval', run=True,
                      checkpoint='step-000010.pt')


@pytest.mark.parametrize('options', [{'batch_size': 0}, {'cpu_threads': 0}])
def test_invalid_resource_limits_rejected(run_folder, options):
    with pytest.raises(ValueError, match='positive'):
        standard_plan(run_folder, checkpoint='step-000010.pt', **options)


def test_selected_native_export_preserves_logits_and_existing_export(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('safetensors')
    transformers = pytest.importorskip('transformers')
    tokenizers = pytest.importorskip('tokenizers')
    from modelchef.chat import export_model
    from modelchef.engines import create_model
    from modelchef.run_config import RunConfig

    folder = tmp_path / 'run'
    (folder / 'checkpoints').mkdir(parents=True)
    config = RunConfig(model={'layers': 1, 'hidden_size': 16, 'attention_heads': 2,
                              'kv_heads': 1, 'intermediate_size': 32, 'context_length': 32})
    (folder / 'config.json').write_text(config.model_dump_json())
    vocab = {word: i for i, word in enumerate(['<pad>', '<unk>', '<bos>', '<eos>', 'a', 'b'])}
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token='<unk>'))
    tokenizer.save(str(folder / 'tokenizer.json'))
    model = create_model(config.model, len(vocab)).eval()
    torch.save({'model': model.state_dict()}, folder / 'checkpoints/step-000010.pt')
    (folder / 'export').mkdir()
    (folder / 'export/keep.txt').write_text('existing export')
    destination = tmp_path / 'evaluation/model'
    assert export_model(folder, 'step-000010.pt', destination) == destination
    exported = transformers.AutoModelForCausalLM.from_pretrained(
        destination, local_files_only=True, attn_implementation='eager').eval()
    ids = torch.tensor([[2, 4, 5, 4]])
    with torch.inference_mode():
        assert torch.allclose(model(ids), exported(ids).logits, atol=1e-5, rtol=1e-4)
    assert (folder / 'export/keep.txt').read_text() == 'existing export'
    assert not (folder / 'report.json').exists()
