import json

import pytest

from modelchef.performance_gate import project_run
from modelchef.run_config import PerformanceGate


def test_gate_accounts_for_future_stages_and_safety():
    common = {'measured_tokens': 10000, 'measured_seconds': 10, 'remaining_tokens': 1000000,
              'safety_multiplier': 2, 'overhead_seconds': 100}
    assert project_run(**common, available_seconds=2100)['passed']
    rejected = project_run(**common, available_seconds=2000)
    assert not rejected['passed']
    assert rejected['projected_remaining_seconds'] == 2100


def test_memory_rejects_an_otherwise_fast_run():
    result = project_run(measured_tokens=10000, measured_seconds=10, remaining_tokens=100,
                         available_seconds=1000, safety_multiplier=2, overhead_seconds=100,
                         peak_allocated_bytes=95, total_memory_bytes=100)
    assert not result['passed']


def test_invalid_measurement_and_window():
    with pytest.raises(ValueError):
        project_run(measured_tokens=100, measured_seconds=0, remaining_tokens=100,
                    available_seconds=1000, safety_multiplier=2, overhead_seconds=100)
    with pytest.raises(ValueError):
        PerformanceGate(probe_steps=10, warmup_steps=10)


def test_rejected_gate_preserves_real_checkpoint(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    from modelchef.pipeline import run_pipeline
    from modelchef.run_config import RunConfig

    source = tmp_path / 'documents.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Document {i} discusses learning, '
        'language and science for curious people in a clear and thoughtful way.'}) + '\n'
        for i in range(60)))
    config = RunConfig(name='gate-check', data={'provider': 'local_jsonl',
        'local_path': str(source), 'max_documents': 60, 'max_tokens': 10000},
        processing={'min_chars': 20, 'near_dedup': False,
                    'validation_fraction': 0.2, 'test_fraction': 0.2},
        tokenizer={'vocab_size': 300}, model={'layers': 1, 'hidden_size': 32,
        'attention_heads': 2, 'kv_heads': 1, 'intermediate_size': 64, 'context_length': 16},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 3, 'warmup_steps': 0,
        'micro_batch_size': 1, 'gradient_accumulation': 1, 'performance_gate':
        {'probe_steps': 2, 'warmup_steps': 0, 'overhead_seconds': 0}},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})
    monkeypatch.setenv('MODELCHEF_DEADLINE_EPOCH', '1')
    with pytest.raises(RuntimeError, match='Performance gate rejected'):
        run_pipeline(config, tmp_path)
    run = next((tmp_path / 'artifacts/training').iterdir())
    assert not json.loads((run / 'performance-gate.json').read_text())['passed']
    checkpoint = torch.load(run / 'checkpoints/step-000002.pt', weights_only=True)
    assert checkpoint['step'] == 2 and checkpoint['optimizer']
    assert json.loads((run / 'report.json').read_text())['status'] == 'failed'
