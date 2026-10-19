"""Optional, single-core NeuronX training and fixed-shape inference compilation.

No AWS provisioning or SDK imports happen when planning. Install the SDK in its
own AWS-supported environment, never into the project's CUDA environment.
"""
import hashlib
import json
import math
import platform
from pathlib import Path

from pydantic import Field, model_validator

from modelchef.run_config import Architecture, Section


class NeuronRecipe(Section):
    model: Architecture = Field(default_factory=Architecture)
    vocab_size: int = Field(default=4096, ge=260)
    tokens_path: str
    output_dir: str = 'artifacts/neuron'
    batch_size: int = Field(default=1, gt=0)
    sequence_length: int = Field(default=128, gt=1)
    steps: int = Field(default=10, gt=0)
    learning_rate: float = Field(default=0.0003, gt=0)
    seed: int = Field(default=42, ge=0)
    checkpoint_every: int = Field(default=5, gt=0)

    @model_validator(mode='after')
    def supported(self):
        model_config = self.model
        if (model_config.backend != 'native' or model_config.type != 'decoder_rope' or model_config.peft.method != 'none'
                or model_config.quantization != 'none' or model_config.custom_swiglu_path
                or model_config.attention_kernel not in {'auto', 'math'} or model_config.gradient_checkpointing
                or model_config.attention_dropout or model_config.residual_dropout):
            raise ValueError('Neuron prototype supports the native dense decoder without '
                             'PEFT, quantization, custom CUDA, dropout or activation checkpointing')
        if self.sequence_length > model_config.context_length:
            raise ValueError('sequence_length exceeds model context_length')
        return self


def fingerprint(config):
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()


def plan(config, action):
    return {'backend': 'neuronx', 'action': action, 'training_hardware': 'Trainium',
            'inference_hardware': 'Inferentia2', 'shape': [config.batch_size,
            config.sequence_length], 'precision': 'fp32', 'single_logical_core': True,
            'recipe_sha256': fingerprint(config), 'launches_aws': False,
            'note': 'Hardware-unverified integration. Fixed shapes; no KV cache, vLLM, '
                    'multi-core distribution, S3 recovery or AWS budget controller in this command.'}


def create_decoder(config):
    """Reuse parameter layout; use portable matmul attention instead of CUDA SDPA."""
    import types

    import torch

    from modelchef.model import Decoder

    def forward(attention, x):
        batch_size, sequence_length, _ = x.shape
        model_config, head_dim = attention.attention_config, attention.dim
        q = attention.rope(attention.q(x).view(batch_size, sequence_length, model_config.attention_heads, head_dim).transpose(1, 2))
        k = attention.rope(attention.k(x).view(batch_size, sequence_length, model_config.kv_heads, head_dim).transpose(1, 2))
        v = attention.v(x).view(batch_size, sequence_length, model_config.kv_heads, head_dim).transpose(1, 2)
        repeat = model_config.attention_heads // model_config.kv_heads
        k, v = k.repeat_interleave(repeat, 1), v.repeat_interleave(repeat, 1)
        scores = (q @ k.transpose(-2, -1)) * head_dim ** -0.5
        mask = torch.ones((sequence_length, sequence_length), device=x.device, dtype=torch.bool).triu(1)
        weights = scores.masked_fill(mask, float('-inf')).softmax(-1)
        return attention.out((weights @ v).transpose(1, 2).contiguous().view(batch_size, sequence_length, -1))

    decoder = Decoder(config.model, config.vocab_size)
    for block in decoder.blocks:
        block.attention.forward = types.MethodType(forward, block.attention)
    return decoder


def _sdk():
    if platform.system() != 'Linux':
        raise ValueError('Neuron execution requires Linux and a separate AWS Neuron environment')
    try:
        import torch_neuronx
    except ImportError as error:
        raise ValueError('Install the AWS-supported Neuron SDK/PyTorch stack in a separate '
                         'Neuron DLAMI environment; do not upgrade the CUDA environment') from error
    return torch_neuronx


def _hash_file(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def train(config, resume=None):
    _sdk()
    import numpy as np
    import torch
    import torch_xla.core.xla_model as xm
    from torch.nn import functional

    from modelchef.progress import report

    path = Path(config.tokens_path)
    if not path.is_file() or path.stat().st_size % 4:
        raise ValueError('tokens_path must be a little-endian uint32 packed token file')
    tokens = np.memmap(path, dtype='<u4', mode='r')
    if len(tokens) <= config.sequence_length or int(tokens.max()) >= config.vocab_size:
        raise ValueError('Token file is too short or contains IDs outside vocab_size')
    data_hash = _hash_file(path)
    output = Path(config.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(config.seed)
    generator = torch.Generator().manual_seed(config.seed)
    report('Loading the native decoder on one Neuron logical core; the first step compiles the training graph.')
    device = xm.xla_device()
    model = create_decoder(config).to(device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, foreach=False)
    start = 0
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        if saved['recipe_sha256'] != fingerprint(config) or saved['data_sha256'] != data_hash:
            raise ValueError('Resume requires the identical recipe and token file')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['sampler_rng'])
        start = saved['step']
    for step in range(start + 1, config.steps + 1):
        starts = torch.randint(len(tokens) - config.sequence_length,
                               (config.batch_size,), generator=generator).tolist()
        batch = torch.from_numpy(np.stack([np.array(tokens[i:i + config.sequence_length + 1],
                                                 dtype=np.int64) for i in starts])).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch[:, :-1])
        loss = functional.cross_entropy(logits.reshape(-1, config.vocab_size),
                                        batch[:, 1:].reshape(-1))
        loss.backward()
        xm.optimizer_step(optimizer)
        xm.mark_step()
        value = float(loss.detach().cpu())
        if not math.isfinite(value):
            raise RuntimeError('Neuron training produced non-finite loss; checkpoint not advanced')
        report(f'Neuron optimizer step {step}/{config.steps}: next-token loss={value:.4f}.')
        if step % config.checkpoint_every == 0 or step == config.steps:
            checkpoint = output / f'step-{step:06d}.pt'
            if checkpoint.exists():
                raise ValueError(f'Refusing to overwrite checkpoint: {checkpoint}')
            temporary = checkpoint.with_suffix('.tmp')
            xm.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'step': step, 'sampler_rng': generator.get_state(),
                     'recipe_sha256': fingerprint(config), 'data_sha256': data_hash}, str(temporary))
            temporary.replace(checkpoint)
    return {**plan(config, 'train'), 'optimizer_steps': config.steps,
            'output_dir': str(output), 'status': 'completed'}


def compile_model(config, checkpoint):
    sdk = _sdk()
    import torch

    from modelchef.progress import report

    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if saved['recipe_sha256'] != fingerprint(config):
        raise ValueError('Compilation recipe must match the training checkpoint')
    model = create_decoder(config).eval()
    model.load_state_dict(saved['model'])
    output = Path(config.output_dir) / 'compiled'
    if output.exists():
        raise ValueError('Use a fresh output directory; compiled artifacts are never overwritten')
    output.mkdir(parents=True)
    report('Tracing and compiling fixed-shape decoder logits for Neuron; this is not a generation server.')
    inputs = torch.zeros((config.batch_size, config.sequence_length), dtype=torch.long)
    compiled = sdk.trace(model, inputs, compiler_workdir=str(output / 'compiler-work'))
    torch.jit.save(compiled, str(output / 'model.pt'))
    manifest = {**plan(config, 'compile'), 'checkpoint_sha256': _hash_file(checkpoint),
                'artifact_sha256': _hash_file(output / 'model.pt'),
                'status': 'compiled', 'numerical_validation': 'not_run'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
