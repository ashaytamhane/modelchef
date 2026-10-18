import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pydantic')
pytest.importorskip('tokenizers')

from modelchef.objectives import dpo_loss, group_advantages, grpo_loss, verified_reward
from modelchef.pipeline import run_pipeline
from modelchef.run_config import RunConfig


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


def test_structured_token_budget_counts_the_actual_sequence_once():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from modelchef.objectives import format_prompt, pack_structured
    tokenizer = Tokenizer(models.WordLevel({
        '<unk>': 0, 'User': 1, ':': 2, 'Assistant': 3, 'Q': 4, '1': 5,
        '<pad>': 6, '<eos>': 7}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = RunConfig(stage={'kind': 'sft', 'parent_run': 'parent'},
        data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'}, model={'context_length': 8})
    row = {'prompt': 'Q', 'response': '1'}
    size = len(tokenizer.encode(format_prompt(row['prompt'])).ids) + len(
        tokenizer.encode(' ' + row['response']).ids) + 1
    (x, y), used = pack_structured([row], tokenizer, config, size)
    assert used == size and x.shape == y.shape == (1, 8)
    assert (y != -100).sum() == 2  # Response and EOS, with prompt/padding masked.
    with pytest.raises(ValueError, match='token budget'):
        pack_structured([row], tokenizer, config, size - 1)


def test_structured_bos_is_context_and_not_a_response_target():
    from tokenizers import Tokenizer, models, pre_tokenizers

    from modelchef.objectives import response_pair
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, '<bos>': 1, '<pad>': 2,
        '<eos>': 3, 'User': 4, ':': 5, 'Assistant': 6, 'Hello': 7, 'Hi': 8}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    x, y = response_pair('Hello', 'Hi', tokenizer, 16, add_bos=True)
    assert x[0] == 1 and y[0] == -100
    assert [value for value in y if value != -100] == [8, 3]


def test_dpo_accuracy_uses_reference_adjusted_rewards(monkeypatch):
    from contextlib import nullcontext

    from modelchef import objectives

    class Model:
        def __init__(self, chosen):
            self.chosen = chosen

        def eval(self):
            return self

    policy, reference = Model(2.0), Model(3.0)
    monkeypatch.setattr(objectives, 'sequence_logps', lambda model, x, y:
                        torch.tensor([model.chosen if x[0, 0] == 1 else 1.0]))
    config = RunConfig(stage={'kind': 'dpo', 'parent_run': 'parent'},
        data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'},
        training={'micro_batch_size': 1})
    blocks = (torch.tensor([[1]]), torch.tensor([[1]]),
              torch.tensor([[2]]), torch.tensor([[2]]))
    result = objectives.evaluate_stage(policy, reference, blocks, None, config, 'cpu', nullcontext)
    assert result['policy_likelihood_accuracy'] == 1
    assert result['preference_accuracy'] == 0
    assert result['reference_adjusted_margin'] == -1


@pytest.mark.parametrize('kind', ['midtrain', 'sft', 'dpo', 'rlvr', 'distill'])
@pytest.mark.parametrize('adapter', [False, True])
def test_posttraining_checkpoint_recovery_and_parent_is_unchanged(parent, tmp_path, kind, adapter):
    config, base = parent
    if adapter:
        config.model.peft.method = 'lora'
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
    from modelchef.quantization import fake_nvfp4
    value = torch.linspace(-6, 6, 32).reshape(2, 16).requires_grad_()
    quantized = fake_nvfp4(value)
    assert torch.isfinite(quantized).all()
    quantized.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
    assert torch.equal(fake_nvfp4(torch.zeros(2, 16)), torch.zeros(2, 16))


@pytest.mark.parametrize('lora', [False, True, 'qlora'])
def test_local_pretrained_import_and_sft(parent, tmp_path, lora):
    transformers = pytest.importorskip('transformers')
    pytest.importorskip('safetensors')
    from modelchef.chat import export_model
    from modelchef.pretrained import register_model, stage_config
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
    config.training.checkpoint_every = 1
    if lora:
        pytest.importorskip('peft')
        if lora == 'qlora':
            import importlib.util
            if not importlib.util.find_spec('bitsandbytes'):
                pytest.skip('QLoRA requires bitsandbytes')
            if not torch.cuda.is_available():
                pytest.skip('QLoRA integration requires CUDA')
            config.training.accelerator = 'cuda'
            config.training.precision = 'bf16'
        config.model.peft.method = 'qlora' if lora == 'qlora' else 'lora'
        config.model.peft.target_modules = ['c_attn']
        from modelchef.engines import create_model, load_parent
        initial = create_model(config.model, len(tokenizer))
        parent_weights = torch.load(Path(imported) / 'checkpoints/step-000000.pt', weights_only=True)['model']
        parent_config = RunConfig.model_validate_json((Path(imported) / 'config.json').read_text())
        load_parent(initial, parent_weights, parent_config.model)
        frozen = {name: p.detach().clone() for name, p in initial.named_parameters() if not p.requires_grad}
    result = run_pipeline(config, tmp_path)
    assert result['stage'] == 'sft'
    exported = export_model(result['run_dir'])
    restored = transformers.AutoModelForCausalLM.from_pretrained(exported, local_files_only=True)
    assert restored.config.model_type == 'gpt2'
    if lora:
        assert 0 < result['trainable_parameters'] < result['parameters']
        assert (Path(result['run_dir']) / 'adapter/adapter_model.safetensors').exists()
        trained = create_model(config.model, len(tokenizer))
        weights = torch.load(result['checkpoint'], weights_only=True)['model']
        trained.load_state_dict(weights)
        for name, p in trained.named_parameters():
            if name in frozen:
                assert torch.equal(p, frozen[name]), name
        assert any('lora_B' in name and value.abs().sum() > 0 for name, value in weights.items())
        trained.eval()
        restored.eval()
        ids = torch.tensor([[1, 2, 3]], device='cuda' if lora == 'qlora' else 'cpu')
        with torch.no_grad():
            if lora == 'qlora':
                assert torch.isfinite(trained(ids)).all()
                assert torch.isfinite(restored(ids.cpu()).logits).all()
            else:
                assert torch.allclose(trained(ids), restored(ids).logits, atol=1e-5, rtol=1e-5)
        resumed = run_pipeline(config, tmp_path, Path(result['run_dir']) / 'checkpoints/step-000001.pt')
        recovered = torch.load(resumed['checkpoint'], weights_only=True)['model']
        assert all(torch.equal(value, recovered[name]) for name, value in weights.items())


def test_reasoning_is_optional_and_student_can_share_teacher_vocabulary(parent, tmp_path):
    from modelchef.pretrained import initialize_student
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


def test_fp16_overflow_backs_off_and_recovers_without_corrupting_weights(parent, tmp_path, monkeypatch):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip('FP16 CUDA GradScaler integration requires an NVIDIA GPU')
    from modelchef import pipeline
    config, _ = parent
    config.training.accelerator = 'cuda'
    config.training.precision = 'fp16'
    config.training.max_runtime_seconds = 60
    original_create = pipeline.create_model
    overflows = []

    def create_with_one_overflow(*args, **kwargs):
        model = original_create(*args, **kwargs)

        def overflow_once(gradient):
            if not overflows:
                overflows.append(True)
                return torch.full_like(gradient, float('inf'))
            return gradient

        model.embedding.weight.register_hook(overflow_once)
        return model

    monkeypatch.setattr(pipeline, 'create_model', create_with_one_overflow)
    result = run_pipeline(config, tmp_path)
    checkpoint = torch.load(result['checkpoint'], map_location='cpu', weights_only=True)
    assert result['optimizer_steps'] == config.training.steps
    assert result['skipped_optimizer_updates'] >= 1
    assert checkpoint['skipped_optimizer_updates'] == result['skipped_optimizer_updates']
    assert all(torch.isfinite(value).all() for value in checkpoint['model'].values())


def test_custom_cuda_probe_preserves_rng_and_rejects_cpu(monkeypatch):
    from modelchef import execution
    before = torch.get_rng_state().clone()
    def probe(path):
        torch.manual_seed(42)
        return {'checked': str(path)}
    monkeypatch.setattr(execution, '_check_custom_kernel', probe)
    assert execution.check_custom_kernel('example')['checked'] == 'example'
    assert torch.equal(before, torch.get_rng_state())
    with pytest.raises(ValueError, match='CUDA tensors'):
        execution.custom_swiglu('example', torch.ones(2), torch.ones(2))
