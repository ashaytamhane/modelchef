"""Native low-rank adapters with unchanged base parameter names."""
import math

import torch
import torch.nn.functional as F


class LoRALinear(torch.nn.Linear):
    def forward(self, inputs):
        base = F.linear(inputs, self.weight, self.bias)
        return base + F.linear(F.linear(self.lora_dropout(inputs), self.lora_A),
                               self.lora_B) * self.lora_scaling


def apply_lora(model, config):
    targets = set(config.target_modules)
    # Offer the same default names as HF Llama, mapped to the native attention layers.
    aliases = {'q_proj': 'attention.q', 'k_proj': 'attention.k', 'v_proj': 'attention.v',
               'o_proj': 'attention.out', 'gate_proj': 'gate', 'up_proj': 'up', 'down_proj': 'down'}
    targets = {aliases.get(name, name) for name in targets}
    selected = [(name, layer) for name, layer in model.named_modules()
                if isinstance(layer, torch.nn.Linear) and any(
                    name == target or name.endswith('.' + target) for target in targets)]
    matched = {target for target in targets if any(name == target or name.endswith('.' + target)
                                                   for name, _ in selected)}
    if targets != matched:
        raise ValueError(f'LoRA targets do not match native layers: {sorted(targets - matched)}')
    model.requires_grad_(False)
    for _, layer in selected:
        layer.__class__ = LoRALinear
        layer.lora_A = torch.nn.Parameter(layer.weight.new_empty(config.rank, layer.in_features))
        layer.lora_B = torch.nn.Parameter(layer.weight.new_zeros(layer.out_features, config.rank))
        torch.nn.init.kaiming_uniform_(layer.lora_A, a=math.sqrt(5))
        layer.lora_dropout = torch.nn.Dropout(config.dropout)
        layer.lora_scaling = config.alpha / (math.sqrt(config.rank) if config.use_rslora else config.rank)


def merged_state(state, config):
    """Return dense weights without modifying the training checkpoint."""
    result = {name: value.clone() for name, value in state.items() if '.lora_' not in name}
    scaling = config.alpha / (math.sqrt(config.rank) if config.use_rslora else config.rank)
    for name, value in state.items():
        if name.endswith('.lora_A'):
            prefix = name.removesuffix('.lora_A')
            result[prefix + '.weight'] += (state[prefix + '.lora_B'] @ value) * scaling
    return result
