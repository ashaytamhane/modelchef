import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pydantic')
pytest.importorskip('tokenizers')

from ashayllm.objectives import dpo_loss, group_advantages, grpo_loss, verified_reward
from ashayllm.pipeline import run_pipeline
from ashayllm.run_config import RunConfig


@pytest.fixture
def parent(tmp_path):
    torch.set_num_threads(1)
    path = tmp_path / 'raw.jsonl'
    path.write_text(''.join(json.dumps({'text': f'Topic {i}: students study science. ' * 8})
                            + '\n' for i in range(30)))
    config = RunConfig(name='parent', explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(path), 'max_tokens': 6000},
        processing={'min_chars': 1, 'near_dedup': False}, tokenizer={'vocab_size': 300},
        model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4, 'kv_heads': 2,
               'intermediate_size': 64, 'context_length': 64},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})
    return config, run_pipeline(config, tmp_path)


def test_preference_and_reward_objectives_have_correct_direction():
    chosen = torch.tensor([0.0], requires_grad=True)
    loss = dpo_loss(chosen, torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.0]), 0.1)
    loss.backward()
    assert chosen.grad.item() < 0  # Gradient descent increases chosen log probability.
    assert loss.item() == pytest.approx(0.693147, abs=1e-5)
    assert verified_reward(' +07 ', '7', 'integer') == 1
    assert verified_reward('7 extra words', '7', 'integer') == 0
    assert verified_reward('Yes', 'yes', 'exact_match') == 0
    assert verified_reward('<think>Two plus three.</think><answer>5</answer>', '5',
                           'integer', 'think_answer') == 1
    assert verified_reward('<answer>5</answer><answer>6</answer>', '5',
                           'integer', 'think_answer') == 0
    rewards = torch.tensor([0.0, 1.0])
    advantages = group_advantages(rewards)
    assert advantages.tolist() == [-1, 1]
    assert group_advantages(torch.ones(3)).tolist() == [0, 0, 0]
    policy = torch.zeros(2, 1, requires_grad=True)
    grpo_loss(policy, torch.zeros_like(policy), torch.zeros_like(policy), advantages,
              torch.ones_like(policy, dtype=torch.bool), 0.2, 0.02).backward()
    assert policy.grad[0].item() > 0 and policy.grad[1].item() < 0


@pytest.mark.parametrize('kind', ['midtrain', 'sft', 'dpo', 'rlvr', 'distill'])
def test_posttraining_checkpoint_recovery_and_parent_is_unchanged(parent, tmp_path, kind):
    config, base = parent
    source = tmp_path / 'stage.jsonl'
    rows = [{'text': f'Specialized finance lesson {i}: bank revenue grows. ' * 8,
             'prompt': f'Question {i}', 'response': '1', 'chosen': '1', 'rejected': '2',
             'answer': '1'} for i in range(30)]
    source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    recipe = config.model_dump()
    recipe['data']['local_path'] = str(source)
    recipe['stage'] = {'kind': kind, 'parent_run': base['run_dir'],
                       'teacher_run': base['run_dir'] if kind == 'distill' else None,
                       'group_size': 2, 'rollout_tokens': 2}
    config = RunConfig.model_validate(recipe)
    before = Path(base['checkpoint']).read_bytes()
    result = run_pipeline(config, tmp_path)
    assert result['stage'] == kind and result['checkpoint_recovery'] == 'passed'
    weights = torch.load(result['checkpoint'], weights_only=True)['model']
    checkpoint = Path(result['run_dir']) / 'checkpoints/step-000001.pt'
    resumed = run_pipeline(config, tmp_path, checkpoint)
    recovered = torch.load(resumed['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(weights[key], recovered[key]) for key in weights)
    assert Path(base['checkpoint']).read_bytes() == before
    assert result['parent']['tokenizer_sha256'] == result['tokenizer_sha256']


def test_fp4_simulation_preserves_gradient_and_zero_values():
    from ashayllm.quantization import fake_nvfp4
    value = torch.linspace(-6, 6, 32).reshape(2, 16).requires_grad_()
    quantized = fake_nvfp4(value)
    assert torch.isfinite(quantized).all()
    quantized.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
    assert torch.equal(fake_nvfp4(torch.zeros(2, 16)), torch.zeros(2, 16))


def test_local_pretrained_import_and_sft(parent, tmp_path):
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('safetensors')
    from ashayllm.chat import export_model
    from ashayllm.pretrained import register_model, stage_config
    config, base = parent
    hf = tmp_path / 'hf-model'
    tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_file=str(
        Path(base['run_dir']) / 'tokenizer.json'), eos_token='<eos>', bos_token='<bos>',
        unk_token='<unk>', pad_token='<pad>')
    tokenizer.save_pretrained(hf)
    model = transformers.GPT2LMHeadModel(transformers.GPT2Config(vocab_size=len(tokenizer),
        n_embd=32, n_head=4, n_layer=1, n_positions=64))
    model.save_pretrained(hf, safe_serialization=True)
    imported = register_model(hf, tmp_path, context=64)
    source = tmp_path / 'sft.jsonl'
    source.write_text(''.join(json.dumps({'prompt': f'Question {i}', 'response': '1'})
                            + '\n' for i in range(30)))
    recipe = tmp_path / 'sft.yaml'
    stage_config(imported, 'sft', str(source), recipe, steps=2)
    import yaml
    config = RunConfig.model_validate(yaml.safe_load(recipe.read_text()))
    config.training.accelerator = 'cpu'
    config.training.precision = 'fp32'
    result = run_pipeline(config, tmp_path)
    assert result['stage'] == 'sft'
    exported = export_model(result['run_dir'])
    restored = transformers.AutoModelForCausalLM.from_pretrained(exported, local_files_only=True)
    assert restored.config.model_type == 'gpt2'


def test_reasoning_is_optional_and_student_can_share_teacher_vocabulary(parent, tmp_path):
    from ashayllm.pretrained import initialize_student
    _, base = parent
    student = initialize_student(base['run_dir'], tmp_path, architecture={
        'layers': 1, 'hidden_size': 16, 'attention_heads': 2, 'kv_heads': 1,
        'intermediate_size': 32, 'context_length': 64})
    assert (Path(student) / 'tokenizer.json').read_bytes() == (
        Path(base['run_dir']) / 'tokenizer.json').read_bytes()
    source = tmp_path / 'reasoning.jsonl'
    source.write_text(''.join(json.dumps({'prompt': f'Q{i}', 'reasoning': 'Add one.', 'answer': '2'})
                            + '\n' for i in range(30)))
    recipe = RunConfig.model_validate_json((Path(student) / 'config.json').read_text()).model_dump()
    recipe['data'].update(provider='local_jsonl', local_path=str(source), max_tokens=6000)
    recipe['processing'].update(min_chars=1, near_dedup=False)
    recipe['training'].update(accelerator='cpu', precision='fp32', steps=2, warmup_steps=0,
                               micro_batch_size=1, gradient_accumulation=1)
    recipe['evaluation'].update(batches=1, generate_tokens=1)
    recipe['stage'].update(kind='distill', parent_run=student, teacher_run=base['run_dir'],
                          reasoning_format='think_answer')
    result = run_pipeline(RunConfig.model_validate(recipe), tmp_path)
    assert result['stage'] == 'distill'
    text = (Path(result['run_dir']) / 'train.jsonl').read_text()
    assert '<think>' in text and '<answer>' in text
    assert RunConfig().stage.reasoning_format == 'plain'


def test_math_attention_and_compile_preserve_checkpoint_names(parent, tmp_path):
    config, _ = parent
    config.model.attention_kernel = 'math'
    config.training.compile = True
    config.training.compile_backend = 'eager'
    config.training.optimizer_kernel = 'foreach'
    result = run_pipeline(config, tmp_path)
    assert result['execution']['compile'] and result['checkpoint_recovery'] == 'passed'
    weights = torch.load(result['checkpoint'], weights_only=True)['model']
    assert 'embedding.weight' in weights and not any('_orig_mod' in key for key in weights)


def test_custom_cuda_probe_preserves_rng_and_rejects_cpu(monkeypatch):
    from ashayllm import execution
    before = torch.get_rng_state().clone()
    def probe(path):
        torch.manual_seed(42)
        return {'checked': str(path)}
    monkeypatch.setattr(execution, '_check_custom_kernel', probe)
    assert execution.check_custom_kernel('example')['checked'] == 'example'
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match='CUDA tensors'):
        execution.custom_swiglu('example', torch.ones(2), torch.ones(2))
