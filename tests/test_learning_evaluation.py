from contextlib import nullcontext

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('pydantic')

from modelchef.corpus import PII
from modelchef.pipeline import evaluate
from modelchef.run_config import Evaluation


def test_all_evaluation_covers_every_example_and_ignores_masked_targets():
    class FixedModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([2., 0.]).expand(*x.shape, 2)

    model = FixedModel()
    x = torch.zeros(3, 2, dtype=torch.long)
    y = torch.tensor([[0, -100], [1, 0], [-100, 1]])
    result = evaluate(model, (x, y), 'all', 2, 'cpu', nullcontext)
    expected = torch.nn.functional.cross_entropy(model(x).reshape(-1, 2), y.flatten())
    assert result['loss'] == pytest.approx(expected.item())
    assert result['tokens'] == 4
    assert result['coverage'] == 'all'
    assert result['evaluated_examples'] == result['available_examples'] == 3
    limited = evaluate(model, (x, y), 1, 2, 'cpu', nullcontext)
    assert limited['tokens'] == 3
    assert limited['coverage'] == 'bounded'
    assert limited['evaluated_examples'] == 2


def test_evaluation_batch_setting_accepts_all_and_rejects_invalid_limits():
    assert Evaluation(batches='all').batches == 'all'
    assert Evaluation(batches=3).batches == 3
    for invalid in [0, -1, 'everything']:
        with pytest.raises(ValueError):
            Evaluation(batches=invalid)


def test_pii_filter_preserves_decimal_measurements_but_redacts_phone_numbers():
    values = 'touch: 0.428571429; vision: 2.428571429; change: -0.123456789.'
    assert PII.sub('[REDACTED]', values) == values
    for phone in ['+1 (202) 555-0123', '202.555.0123', '2025550123', '+91 98765 43210']:
        assert '[REDACTED]' in PII.sub('[REDACTED]', 'Contact ' + phone + ' today.')
    assert PII.sub('[REDACTED]', 'a@example.com') == '[REDACTED]'
