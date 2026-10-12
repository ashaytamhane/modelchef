import copy

import pytest

torch = pytest.importorskip('torch')

from modelchef.model import Decoder
from modelchef.optimizers import create_optimizer
from modelchef.run_config import Architecture, RunConfig, Training


def test_in_backward_matches_adamw_and_checkpoint_resume_with_tied_weights():
    torch.set_num_threads(1)
    torch.manual_seed(3)
    config = Architecture(layers=2, hidden_size=32, attention_heads=4, kv_heads=2,
                          intermediate_size=64, context_length=16, gradient_checkpointing=True)
    baseline = Decoder(config, 64)
    optimized = copy.deepcopy(baseline)
    t = Training(optimizer='adamw_in_backward', gradient_accumulation=1, gradient_clip_norm=0)
    ordinary = torch.optim.AdamW(baseline.parameters(), lr=t.learning_rate,
                                betas=t.betas, eps=t.optimizer_eps, weight_decay=t.weight_decay)
    backward = create_optimizer([{'params': list(optimized.parameters()), 'weight_decay': t.weight_decay}], t, 'cpu')
    x, y = torch.randint(64, (1, 16)), torch.randint(64, (1, 16))

    def update(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        optimizer.step()
        return loss

    for _ in range(2):
        update(baseline, ordinary)
        update(optimized, backward)
    assert all(torch.allclose(a, b, atol=1e-7) for a, b in zip(baseline.parameters(), optimized.parameters()))
    assert all(parameter.grad is None for parameter in optimized.parameters())
    assert torch.isfinite(backward.gradient_norm())
    resumed = Decoder(config, 64)
    resumed.load_state_dict(optimized.state_dict())
    restored = create_optimizer([{'params': list(resumed.parameters()), 'weight_decay': t.weight_decay}], t, 'cpu')
    restored.load_state_dict(copy.deepcopy(backward.state_dict()))
    update(optimized, backward)
    update(resumed, restored)
    assert all(torch.equal(a, b) for a, b in zip(optimized.parameters(), resumed.parameters()))


@pytest.mark.parametrize('change', [
    {'gradient_accumulation': 2}, {'gradient_clip_norm': 1}, {'precision': 'fp16'},
    {'compile': True}, {'distributed': 'ddp'}])
def test_in_backward_rejects_incompatible_training(change):
    with pytest.raises(ValueError, match='In-backward AdamW requires'):
        RunConfig(training={'optimizer': 'adamw_in_backward', 'gradient_accumulation': 1,
                            'gradient_clip_norm': 0, **change})


def test_eight_bit_optimizer_requires_cuda():
    p = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError, match='NVIDIA CUDA'):
        create_optimizer([{'params': [p]}], Training(optimizer='adamw8bit'), 'cpu')
