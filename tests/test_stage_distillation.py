import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('tokenizers')

from modelchef.objectives import objective
from modelchef.pipeline import run_pipeline
from modelchef.run_config import RunConfig


@pytest.mark.parametrize('kind', ['pretrain', 'midtrain', 'sft'])
def test_optional_teacher_training_and_exact_resume(tmp_path, kind):
    torch.set_num_threads(1)
    source = tmp_path / 'data.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Topic {i}: students learn science. ' * 8,
        'prompt': f'Question {i}', 'response': 'Students learn science.'}) + '\n' for i in range(30)))
    c = RunConfig(name='teacher', explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(source), 'max_tokens': 6000},
        processing={'min_chars': 1, 'near_dedup': False}, tokenizer={'vocab_size': 300},
        model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4, 'kv_heads': 2,
               'intermediate_size': 64, 'context_length': 64},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})
    teacher = run_pipeline(c, tmp_path)
    teacher_bytes = Path(teacher['checkpoint']).read_bytes()
    recipe = c.model_dump()
    recipe['name'] = 'student'
    recipe['stage'] = {'kind': kind, 'teacher_run': teacher['run_dir'],
                       'parent_run': None if kind == 'pretrain' else teacher['run_dir']}
    if kind == 'pretrain':
        recipe['tokenizer']['algorithm'] = 'pretrained'
    student = RunConfig.model_validate(recipe)
    prepared = run_pipeline(student, tmp_path, prepare_only=True)
    if kind == 'pretrain':
        assert 'parent' not in prepared
        initial = torch.load(prepared['checkpoint'], weights_only=True)['model']
        trained = torch.load(teacher['checkpoint'], weights_only=True)['model']
        assert any(not torch.equal(initial[key], trained[key]) for key in initial)
    result = run_pipeline(student, tmp_path, prepared['checkpoint'])
    assert result['stage'] == kind and result['checkpoint_recovery'] == 'passed'
    assert result['tokenizer_sha256'] == teacher['tokenizer_sha256']
    assert result['distillation']['mask'] == ('response-only' if kind == 'sft' else 'all target tokens')
    metrics = [json.loads(line) for line in (Path(result['run_dir']) / 'metrics.jsonl').read_text().splitlines()]
    assert all('teacher_kl' in row for row in metrics)
    a = torch.load(result['checkpoint'], weights_only=True)['model']
    resumed = run_pipeline(student, tmp_path, Path(result['run_dir']) / 'checkpoints/step-000001.pt')
    b = torch.load(resumed['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(a[key], b[key]) for key in a)
    assert Path(teacher['checkpoint']).read_bytes() == teacher_bytes

    if kind == 'pretrain':
        import io
        import tarfile

        from modelchef.aws_training import build_bundle
        (tmp_path / 'pyproject.toml').write_text('[project]\nname="test"\n')
        bundle = build_bundle(tmp_path, {'max_bundle_bytes': 10000000},
                              raw_override=student.model_dump(mode='json'))
        with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
            names = archive.getnames()
            assert 'inputs/teacher_run/tokenizer.json' in names
            assert 'inputs/teacher_run/checkpoints/step-000002.pt' in names


@pytest.mark.parametrize('kind', ['pretrain', 'midtrain', 'sft'])
def test_teacher_loss_masks_targets_and_freezes_teacher(kind):
    from torch import nn
    model, teacher = nn.Embedding(5, 5), nn.Embedding(5, 5)
    c = RunConfig(stage={'kind': kind, 'teacher_run': 'teacher',
                        'parent_run': None if kind == 'pretrain' else 'parent',
                        'distillation_alpha': 1},
                  data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'})
    x, y = torch.tensor([[1, 2, 3]]), torch.tensor([[-100, 3, -100]])
    loss, metrics = objective(model, (x, y), c, reference=teacher)
    loss.backward()
    assert torch.isfinite(loss) and metrics['teacher_kl'] >= 0
    assert teacher.weight.grad is None
    assert model.weight.grad[1].count_nonzero() == 0
    assert model.weight.grad[3].count_nonzero() == 0
    assert model.weight.grad[2].count_nonzero() > 0


@pytest.mark.parametrize('kind', ['dpo', 'rlvr'])
def test_online_distillation_rejects_unsupported_objectives(kind):
    with pytest.raises(ValueError, match='Online token distillation supports'):
        RunConfig(stage={'kind': kind, 'parent_run': 'parent', 'teacher_run': 'teacher'},
                  data={'provider': 'local_jsonl', 'local_path': 'data.jsonl'})
