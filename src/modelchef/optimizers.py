"""Optional low-memory AdamW policies; ordinary AdamW remains the default."""
import torch


class InBackwardAdamW(torch.optim.Optimizer):
    """Update each parameter after its gradient accumulates; no global gradient buffer."""
    def __init__(self, groups, training, factory):
        super().__init__(groups, {'lr': training.learning_rate})
        self.optimizers, self.handles, self.norms = [], [], []
        for index, group in enumerate(self.param_groups):
            for parameter in group['params']:
                optimizer = factory([parameter], lr=training.learning_rate, betas=training.betas,
                                    eps=training.optimizer_eps, weight_decay=group['weight_decay'])
                self.optimizers.append(optimizer)

                def update(p, optimizer=optimizer, index=index):
                    if p.grad is None:
                        return
                    norm = p.grad.detach().float().norm()
                    if not torch.isfinite(norm):
                        raise FloatingPointError('Nonfinite gradient; in-backward update interrupted')
                    self.norms.append(norm)
                    optimizer.param_groups[0]['lr'] = self.param_groups[index]['lr']
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                self.handles.append(parameter.register_post_accumulate_grad_hook(update))

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.norms.clear()

    def step(self, closure=None):
        if closure is not None:
            raise ValueError('Optimizer closures are not supported in backward')

    def gradient_norm(self):
        return torch.stack(self.norms).square().sum().sqrt() if self.norms else torch.tensor(0.)

    def state_dict(self):
        return {'format': 'in-backward-adamw-v1', 'base': super().state_dict(),
                'optimizers': [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state):
        if state.get('format') != 'in-backward-adamw-v1' or len(state['optimizers']) != len(self.optimizers):
            raise ValueError('In-backward optimizer checkpoint differs from this model')
        super().load_state_dict(state['base'])
        for optimizer, saved in zip(self.optimizers, state['optimizers']):
            optimizer.load_state_dict(saved)


def create_optimizer(groups, training, device):
    eight_bit = training.optimizer.startswith('adamw8bit')
    backward = training.optimizer.endswith('_in_backward')
    factory = torch.optim.AdamW
    if eight_bit:
        if device != 'cuda' or torch.version.hip:
            raise ValueError('8-bit AdamW requires an NVIDIA CUDA GPU and the optimizer-8bit extra')
        from modelchef.low_precision import prepare_bnb_libraries
        prepare_bnb_libraries()
        import bitsandbytes as bnb
        factory = bnb.optim.AdamW8bit
    if backward:
        return InBackwardAdamW(groups, training, factory)
    options = ({'fused': True} if training.optimizer_kernel == 'fused' else
               {'foreach': True} if training.optimizer_kernel == 'foreach' else {}) if not eight_bit else {}
    return factory(groups, lr=training.learning_rate, betas=training.betas,
                   weight_decay=training.weight_decay, eps=training.optimizer_eps, **options)
