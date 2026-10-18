import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
from modelchef.adapters import merged_state
from modelchef.chat import export_model
from modelchef.engines import create_model, load_parent
from modelchef.pipeline import run_pipeline
from modelchef.run_config import RunConfig


def recipe(tmp_path, model_type='decoder_rope'):
    source = tmp_path / 'corpus.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Lesson {i}: science uses observation. ' * 8})
                             + '\n' for i in range(30)))
    return RunConfig(name='extension', explain='brief', output_dir=str(tmp_path / 'runs'),
        data={'provider': 'local_jsonl', 'local_path': str(source), 'max_tokens': 6000},
        processing={'min_chars': 1, 'near_dedup': False}, tokenizer={'vocab_size': 300},
        model={'type': model_type, 'layers': 1, 'hidden_size': 32, 'attention_heads': 4,
               'kv_heads': 2, 'intermediate_size': 64, 'context_length': 32,
               'num_experts': 3, 'experts_per_token': 2},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})


def test_native_lora_freezes_base_and_merges_exactly(tmp_path):
    config = recipe(tmp_path)
    base = create_model(config.model, 300)
    original = {name: p.detach().clone() for name, p in base.named_parameters()}
    architecture = config.model.model_copy(deep=True)
    architecture.peft.method = 'lora'
    adapted = create_model(architecture, 300)
    load_parent(adapted, base.state_dict(), config.model)
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=.01)
    ids = torch.tensor([[1, 2, 3]])
    for _ in range(2):
        optimizer.zero_grad()
        adapted(ids).square().mean().backward()
        optimizer.step()
    assert all(torch.equal(original[name], p) for name, p in adapted.named_parameters() if name in original)
    base.load_state_dict(merged_state(adapted.state_dict(), architecture.peft))
    assert torch.allclose(base(ids), adapted(ids), atol=1e-6, rtol=1e-5)


def test_workflow_preflight_and_end_to_end(tmp_path):
    import yaml

    from modelchef.workflow import cook
    torch.set_num_threads(1)
    first = recipe(tmp_path)
    first_path = tmp_path / 'first.yaml'
    first_path.write_text(yaml.safe_dump(first.model_dump(mode='json')))
    data = tmp_path / 'sft.jsonl'
    data.write_text(''.join(json.dumps({'prompt': f'Question {i}', 'response': '1'}) + '\n'
                            for i in range(30)))
    second_path = tmp_path / 'second.yaml'
    second_path.write_text(yaml.safe_dump({
        'data': {'provider': 'local_jsonl', 'local_path': str(data), 'max_tokens': 6000},
        'processing': {'min_chars': 1, 'near_dedup': False},
        'training': {'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                     'micro_batch_size': 1, 'gradient_accumulation': 1},
        'evaluation': {'benchmark': 'none', 'batches': 1, 'generate_tokens': 1},
        'model': {'peft': {'method': 'lora'}}, 'stage': {'kind': 'sft'}}))
    workflow = tmp_path / 'workflow.yaml'
    workflow.write_text(yaml.safe_dump({'stages': [
        {'name': 'base', 'recipe': str(first_path)},
        {'name': 'adapt', 'recipe': str(second_path)}]}))
    assert cook(workflow, tmp_path)['mode'] == 'local'
    assert not (tmp_path / 'artifacts/workflows').exists()
    result = cook(workflow, tmp_path, execute=True)
    assert result['status'] == 'completed'
    assert (Path(result['export_dir']) / 'model.safetensors').exists()
    assert result['stages']['adapt']['parent']['checkpoint_sha256']
    workflow.write_text(yaml.safe_dump({'stages': [{'name': 'bad', 'recipe': str(second_path)}]}))
    with pytest.raises(ValueError, match='earlier run'):
        cook(workflow, tmp_path)


def test_moe_training_reload_and_mixtral_export(tmp_path):
    transformers = pytest.importorskip('transformers')
    torch.set_num_threads(1)
    config = recipe(tmp_path, 'decoder_moe')
    result = run_pipeline(config, tmp_path)
    exported = export_model(result['run_dir'])
    model = create_model(config.model, 300)
    # The tiny corpus can produce fewer than the requested tokenizer tokens.
    from modelchef.engines import load_tokenizer
    tokenizer = load_tokenizer(Path(result['run_dir']))
    model = create_model(config.model, tokenizer.get_vocab_size()).eval()
    model.load_state_dict(torch.load(result['checkpoint'], weights_only=True)['model'])
    restored = transformers.AutoModelForCausalLM.from_pretrained(exported, local_files_only=True).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        assert torch.allclose(model(ids), restored(ids).logits, atol=1e-6, rtol=1e-5)
    assert restored.config.model_type == 'mixtral'


def test_nvfp4_integration_simulation_and_actual_prerequisite(tmp_path, monkeypatch):
    from modelchef.low_precision import apply_nvfp4, require_nvfp4
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 10:
        with pytest.raises(ValueError, match='Blackwell'):
            require_nvfp4()

    class SimulatedTE:
        @staticmethod
        def Linear(inputs, outputs, bias, params_dtype, device):
            return torch.nn.Linear(inputs, outputs, bias=bias, dtype=params_dtype)

        @staticmethod
        def autocast(**kwargs):
            assert kwargs['enabled']
            return nullcontext()

    monkeypatch.setattr('modelchef.low_precision.require_nvfp4', lambda: (SimulatedTE, object()))
    c = recipe(tmp_path).model
    model = create_model(c, 300)
    expected = model(torch.tensor([[1, 2, 3]])).detach()
    apply_nvfp4(model)
    actual = model(torch.tensor([[1, 2, 3]]))
    assert torch.equal(expected, actual)
    actual.sum().backward()
    assert model.blocks[0].attention.q.weight.grad is not None


@pytest.mark.skipif(os.name == 'nt', reason='Windows CPU PyTorch wheel lacks torchrun libuv support')
@pytest.mark.parametrize('model_type', ['decoder_rope', 'decoder_moe'])
def test_two_cpu_rank_ddp_and_resume(tmp_path, model_type):
    import yaml
    config = recipe(tmp_path, model_type)
    config.training.distributed = 'ddp'
    config.evaluation.evaluate_test = False
    config.training.profile_phases = True
    if model_type == 'decoder_rope':
        config.data.storage = 'disk'
    path = tmp_path / 'recipe.yaml'
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')))
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, 'PYTHONPATH': str(root / 'src'), 'OMP_NUM_THREADS': '1'}
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
               '--module', '--', 'modelchef.cli', 'pipeline', str(path), '--run', '--quiet']
    executed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=120, check=False)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    folder = next((tmp_path / 'runs').iterdir())
    report = json.loads((folder / 'report.json').read_text())
    assert report['distributed']['world_size'] == 2
    assert report['checkpoint_reload_verified']
    assert report['test'] is None and report['test_evaluation'] == 'withheld'
    assert len(report['phase_metrics_by_rank']) == 2
    final = torch.load(report['checkpoint'], weights_only=True)['model']
    executed = subprocess.run(command + ['--resume', str(folder / 'checkpoints/step-000001.pt')],
                              cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120, check=False)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    recovered = torch.load(report['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(value, recovered[name]) for name, value in final.items())
    data = tmp_path / 'post.jsonl'
    data.write_text(''.join(json.dumps({'prompt': f'Q{i}', 'response': '1',
                                       'chosen': '1', 'rejected': '2'}) + '\n' for i in range(30)))
    config.data.local_path = str(data)
    config.data.storage = 'memory'
    config.stage.kind = 'sft' if model_type == 'decoder_rope' else 'dpo'
    config.stage.parent_run = str(folder)
    config.name = 'post'
    if model_type == 'decoder_rope':
        config.model.peft.method = 'lora'
    path.write_text(yaml.safe_dump(config.model_dump(mode='json')))
    executed = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True,
                              timeout=120, check=False)
    assert executed.returncode == 0, executed.stdout + executed.stderr
    post = next((tmp_path / 'runs').glob('post-*'))
    assert json.loads((post / 'report.json').read_text())['status'] == 'completed'
