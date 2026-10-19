"""Allocation-free native-model accounting, with explicit unmeasured terms."""
import math


def parameter_layout(architecture, vocab_size):
    """Count unique native parameters and FSDP units, including tied embeddings."""
    if architecture.backend != 'native' or architecture.peft.method != 'none':
        raise ValueError('Analytical parameter accounting requires a native model without PEFT')
    h, f = architecture.hidden_size, architecture.intermediate_size
    kv = h // architecture.attention_heads * architecture.kv_heads
    attention = 2 * h * h + 2 * h * kv
    bias = 2 * h + 2 * kv if architecture.bias else 0
    feedforward = 3 * h * f + (2 * f + h if architecture.bias else 0)
    if architecture.type == 'decoder_moe':
        feedforward = architecture.num_experts * 3 * h * f + h * architecture.num_experts
    block = attention + bias + feedforward + 2 * h
    root = vocab_size * h * (1 if architecture.tie_embeddings else 2) + h
    return {'parameters': root + architecture.layers * block,
            'block_parameters': block, 'root_parameters': root,
            'largest_sharding_unit_parameters': max(root, block)}


def memory_plan(config, *, gpu_count=1, memory_bytes=None, hourly_usd=None,
                measured_tokens_per_second=None, remaining_tokens=None):
    """FP32 master/gradient/Adam accounting; activation numbers are scenarios, not bounds."""
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError('gpu_count must be a positive integer')
    for name, value in [('memory_bytes', memory_bytes), ('hourly_usd', hourly_usd),
                        ('measured_tokens_per_second', measured_tokens_per_second)]:
        if value is not None and (isinstance(value, bool) or not math.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be positive and finite')
    if remaining_tokens is not None and (isinstance(remaining_tokens, bool) or
            not isinstance(remaining_tokens, int) or remaining_tokens < 0):
        raise ValueError('remaining_tokens must be a nonnegative integer')
    if config.training.optimizer != 'adamw' or config.model.quantization != 'none':
        raise ValueError('Memory accounting currently requires unquantized standard AdamW')
    layout = parameter_layout(config.model, config.tokenizer.vocab_size)
    parameter_count = layout['parameters']
    model_config, training = config.model, config.training
    batch_tokens = training.micro_batch_size * model_config.context_length
    # Expose individual storage terms rather than claiming one activation formula fits every kernel.
    element_bytes = 4 if training.precision == 'fp32' else 2
    logits = batch_tokens * config.tokenizer.vocab_size * 4  # objective casts logits to FP32
    residual = batch_tokens * model_config.hidden_size * element_bytes
    saved_layers = 1 if model_config.gradient_checkpointing else model_config.layers
    activation_scenario = saved_layers * batch_tokens * (
        8 * model_config.hidden_size + 3 * model_config.intermediate_size) * element_bytes
    if model_config.gradient_checkpointing:
        activation_scenario += model_config.layers * residual
    attention_math = training.micro_batch_size * model_config.attention_heads * model_config.context_length**2 * element_bytes
    plans = {}
    for strategy in ['ddp', 'fsdp2']:
        persistent = math.ceil(parameter_count / gpu_count) * 16 if strategy == 'fsdp2' else parameter_count * 16
        gathered = (2 * layout['largest_sharding_unit_parameters'] * element_bytes
                    if strategy == 'fsdp2' else 0)
        scenario = persistent + activation_scenario + logits + gathered
        plans[strategy] = {'persistent_state_bytes_per_gpu': persistent,
            'weights_bytes_per_gpu': math.ceil(parameter_count / gpu_count) * 4 if strategy == 'fsdp2' else parameter_count * 4,
            'gradients_bytes_per_gpu': math.ceil(parameter_count / gpu_count) * 4 if strategy == 'fsdp2' else parameter_count * 4,
            'adam_moments_bytes_per_gpu': math.ceil(parameter_count / gpu_count) * 8 if strategy == 'fsdp2' else parameter_count * 8,
            'two_gathered_units_scenario_bytes': gathered,
            'illustrative_training_bytes_per_gpu': scenario,
            'persistent_state_exceeds_capacity': persistent > memory_bytes if memory_bytes else None,
            'fit_verified': False}
    cost = None
    if measured_tokens_per_second is not None:
        cost = {'measured_global_tokens_per_second': measured_tokens_per_second,
                'hourly_usd_all_gpus': hourly_usd,
                'usd_per_million_tokens': hourly_usd * 1e6 / (3600 * measured_tokens_per_second)
                    if hourly_usd else None,
                'remaining_compute_hours': remaining_tokens / measured_tokens_per_second / 3600
                    if remaining_tokens is not None else None}
        if cost['remaining_compute_hours'] is not None and hourly_usd:
            cost['remaining_compute_usd'] = cost['remaining_compute_hours'] * hourly_usd
    return {'scope': 'Planning scenarios, not a fit guarantee or measured speedup; no allocation or launch.',
        **layout, 'gpu_count': gpu_count, 'memory_bytes_per_gpu': memory_bytes,
        'global_batch_sequences': training.micro_batch_size * training.gradient_accumulation * gpu_count,
        'global_batch_input_tokens': batch_tokens * training.gradient_accumulation * gpu_count,
        'fp32_master_state_bytes_per_parameter': 16,
        'activation_scenario_bytes': activation_scenario, 'one_residual_tensor_bytes': residual,
        'fp32_logits_tensor_bytes': logits, 'one_math_attention_matrix_bytes': attention_math,
        'strategies': plans, 'cost_from_measurement': cost,
        'limitations': ['Actual vocabulary may differ from requested tokenizer.vocab_size.',
            'Activation scenario is a heuristic, not a lower/upper bound; kernels and autograd matter.',
            'Checkpointing retains layer inputs; the activation scenario does not model every saved tensor.',
            'Auto precision assumes two-byte compute only for illustration; resolve on target hardware.',
            'FSDP shards have per-tensor padding; largest unit can be the embedding/output group.',
            'DDP buckets, allocator fragmentation, CUDA/NCCL, optimizer temporaries and checkpoint staging are additional.',
            'Gather scenario assumes two largest units; prefetch/settings can change overlap.',
            'Cost uses measured GLOBAL throughput for this exact configuration; excludes storage/setup/idle time.']}
