"""Opt-in single-node FSDP2 for native dense next-token training on CUDA."""
import json
import math
import os
import random
import time
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist

from modelchef.corpus import file_digest
from modelchef.phase_metrics import PhaseMetrics
from modelchef.sharded_checkpoint import (
    checkpoint_io_group,
    load_sharded,
    prune_sharded,
    publish_sharded_to_s3,
    read_manifest,
    save_sharded,
)
from modelchef.training_metrics import close_metrics, record_metrics, tensorboard_info


def require_fsdp(config):
    world = int(os.environ.get('WORLD_SIZE', '1'))
    local = int(os.environ.get('LOCAL_WORLD_SIZE', str(world)))
    rank = int(os.environ.get('LOCAL_RANK', '0'))
    if world < 2:
        raise ValueError('FSDP2 requires torchrun --standalone --nproc-per-node=2 (or more)')
    if world != local:
        raise ValueError('FSDP2 currently requires a single node with shared checkpoint storage')
    if not torch.cuda.is_available() or torch.version.hip or torch.cuda.device_count() < world:
        raise ValueError('FSDP2 requires at least one visible NVIDIA CUDA GPU per local rank')
    torch.cuda.set_device(rank)
    return rank


def shard_native(model, precision, device='cuda'):
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
    mesh = init_device_mesh(device, (dist.get_world_size(),))
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16 if precision == 'bf16' else None,
                                  reduce_dtype=torch.float32)
    for block in model.blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    fully_shard(model, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
    model.to_empty(device=torch.device(device, torch.cuda.current_device()) if device == 'cuda'
                   else torch.device(device))
    # Nonpersistent RoPE buffers are not part of a state dict and must be materialized explicitly.
    for block in model.blocks:
        attention = block.attention
        attention.frequencies = 1 / model.config.rope_theta ** (
            torch.arange(0, attention.dim, 2, device=device).float() / attention.dim)
    return model


def run_fsdp(config, root, resume=None):
    from torch.distributed.checkpoint.state_dict import StateDictOptions, set_model_state_dict

    from modelchef.disk_data import load_blocks
    from modelchef.engines import create_model, load_tokenizer
    from modelchef.pipeline import evaluate, run_pipeline, trim_padded_pair

    local_rank = require_fsdp(config)
    resume = Path(resume).resolve() if resume is not None else None
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    dist.init_process_group('nccl', timeout=timedelta(seconds=config.training.distributed_timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    folder, result = None, {}
    try:
        preparation = [None]
        with checkpoint_io_group() as group:
            if rank == 0:
                try:
                    if resume:
                        manifest = read_manifest(resume, world)
                        folder = resume.parent.parent
                        from modelchef.run_config import RunConfig
                        stored = RunConfig.model_validate_json((folder / 'config.json').read_text())
                        if stored.model_dump() != config.model_dump():
                            raise ValueError('Resume configuration differs; refusing to modify the saved run')
                        prepared = {'run_dir': str(folder), 'step': manifest['step']}
                    else:
                        prepared = run_pipeline(config, root, prepare_only=True)
                    preparation[0] = {'prepared': prepared}
                except Exception as error:  # noqa: BLE001 - broadcast preparation errors to waiting ranks
                    preparation[0] = {'error': f'{type(error).__name__}: {error}'}
            dist.broadcast_object_list(preparation, src=0, group=group)
        if 'error' in preparation[0]:
            raise RuntimeError(preparation[0]['error'])
        result = preparation[0]['prepared']
        folder = Path(result['run_dir'])
        training, model_config = config.training, config.model
        random.seed(training.seed + rank)
        torch.manual_seed(training.seed + rank)
        torch.use_deterministic_algorithms(training.deterministic)
        torch.backends.cuda.matmul.allow_tf32 = training.allow_tf32
        supported = torch.tensor(int(torch.cuda.is_bf16_supported()), device='cuda')
        dist.all_reduce(supported, op=dist.ReduceOp.MIN)
        precision = training.precision
        if precision == 'bf16' and not supported.item():
            raise ValueError('BF16 must be supported by every rank; select FP32')
        if precision == 'auto':
            precision = 'bf16' if supported.item() else 'fp32'
        phases = PhaseMetrics('cuda', training.profile_phases)
        tokenizer = load_tokenizer(folder)
        blocks = load_blocks(folder)
        identities = {name: file_digest(folder / name) for name in
                      ['config.json', 'tokenizer.json', 'tokens.pt', 'source-manifest.json']}
        for optional in ['tokenizer-roles.json', 'processing-sources.json']:
            if (folder / optional).exists():
                identities[optional] = file_digest(folder / optional)
        with phases.measure('initialization'):
            with torch.device('meta'):
                model = create_model(model_config, tokenizer.get_vocab_size())
            model = shard_native(model, precision)
            groups = [{'params': [p for p in model.parameters() if p.ndim >= 2],
                       'weight_decay': training.weight_decay},
                      {'params': [p for p in model.parameters() if p.ndim < 2], 'weight_decay': 0}]
            options = {'fused': True} if training.optimizer_kernel == 'fused' else (
                {'foreach': True} if training.optimizer_kernel == 'foreach' else {})
            optimizer = torch.optim.AdamW(groups, lr=training.learning_rate, betas=training.betas,
                                         eps=training.optimizer_eps, **options)
            sampler = torch.Generator().manual_seed(training.seed + rank)
            step, consumed = 0, 0
            if resume:
                metadata, state = load_sharded(model, optimizer, resume, identities)
                step, consumed = metadata['step'], metadata['tokens_seen']
                sampler.set_state(state['sampler_rng'])
                torch.set_rng_state(state['torch_rng'])
                torch.cuda.set_rng_state(state['cuda_rng'], local_rank)
                random.setstate(state['python_rng'])
            else:
                saved = (torch.load(result['checkpoint'], map_location='cpu', mmap=True,
                                    weights_only=True) if rank == 0 else {})
                set_model_state_dict(model, saved.get('model', {}), options=StateDictOptions(
                    full_state_dict=True, broadcast_from_rank0=True))
                del saved
        dtype = torch.bfloat16 if precision == 'bf16' else torch.float32

        def amp():
            return torch.autocast('cuda', dtype=dtype) if precision == 'bf16' else nullcontext()

        def validation(split):
            with phases.measure('validation' if split == 'validation' else 'test'):
                # All ranks execute the same number of forwards: FSDP forwards are collective.
                return evaluate(model, blocks[split][:2], config.evaluation.batches,
                                config.evaluation.batch_size or training.micro_batch_size, 'cuda', amp)

        checkpoints = folder / 'checkpoints'
        initial_tokens = consumed
        started = time.monotonic()
        latest = Path(resume) if resume else None
        latest_step = step if resume else None
        published = set()
        result.update(status='running', distributed={'strategy': 'fsdp2', 'world_size': world,
            'backend': 'nccl', 'precision': precision, 'reshard_after_forward': True,
            'resume_world_size': 'same only'}, test_evaluation='withheld' if not
            config.evaluation.evaluate_test else 'enabled')

        def checkpoint():
            nonlocal latest, latest_step
            path = checkpoints / f'step-{step:06d}.dcp'
            if latest is not None and latest_step == step:
                if latest not in published:
                    publish_sharded_to_s3(root, folder, latest)
                    published.add(latest)
                return latest
            if path.exists():
                import uuid
                suffix = [uuid.uuid4().hex[:12] if rank == 0 else None]
                dist.broadcast_object_list(suffix, src=0)
                path = checkpoints / f'step-{step:06d}-{suffix[0]}.dcp'
            with phases.measure('checkpoint'):
                state = {'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state(local_rank), 'python_rng': random.getstate()}
                latest = save_sharded(model, optimizer, path, {'step': step, 'tokens_seen': consumed,
                    'identities': identities}, state)
                latest_step = step
                publish_sharded_to_s3(root, folder, latest)
                published.add(latest)
                keep = int(os.environ.get('MODELCHEF_LOCAL_CHECKPOINT_KEEP', '2'))
                prune_sharded(checkpoints, keep, protected=(latest, resume))
            return latest

        result['tensorboard'] = tensorboard_info(folder)
        result['initial_validation'] = validation('validation')
        if os.environ.get('MODELCHEF_RECOVERY_S3_URI') or (training.checkpoint_at_start and not resume):
            checkpoint()
        while step < training.steps:
            expired = torch.tensor(int(time.monotonic() - started > training.max_runtime_seconds), device='cuda')
            dist.all_reduce(expired, op=dist.ReduceOp.MAX)
            if expired.item():
                checkpoint()
                raise TimeoutError(f'FSDP2 runtime cap reached; resume from {latest}')
            factor = min(1., (step + 1) / max(1, training.warmup_steps))
            if training.schedule == 'cosine' and step >= training.warmup_steps:
                fraction = (step - training.warmup_steps) / max(1, training.steps - training.warmup_steps - 1)
                factor = training.min_learning_rate_ratio + (1 - training.min_learning_rate_ratio) * (
                    1 + math.cos(math.pi * fraction)) / 2
            for group in optimizer.param_groups:
                group['lr'] = training.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            loss_total, tokens = 0., 0
            for _ in range(training.gradient_accumulation):
                indices = torch.randint(len(blocks['train'][0]), (training.micro_batch_size,), generator=sampler)
                x, y = tuple(part[indices].to('cuda') for part in blocks['train'])
                if config.tokenizer.packing == 'document':
                    x, y = trim_padded_pair(x, y)
                with phases.measure('forward_backward'):
                    with amp():
                        logits = model(x)
                        loss = torch.nn.functional.cross_entropy(logits.float().flatten(0, 1), y.flatten())
                    invalid = torch.tensor(int(not torch.isfinite(loss)), device='cuda')
                    dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                    if invalid.item():
                        raise FloatingPointError('Non-finite FSDP2 training loss')
                    (loss / training.gradient_accumulation).backward()
                    loss_total += loss.item() / training.gradient_accumulation
                    tokens += (y != -100).sum().item()
                    del logits, loss
            with phases.measure('optimizer'):
                # DTensor dispatch computes the norm over all parameter shards.
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm or float('inf'))
                norm_value = norm.full_tensor().item() if hasattr(norm, 'full_tensor') else norm.item()
                if not math.isfinite(norm_value):
                    raise FloatingPointError('Non-finite FSDP2 gradient norm')
                optimizer.step()
            metrics = torch.tensor([loss_total, tokens], device='cuda', dtype=torch.float64)
            dist.all_reduce(metrics)
            consumed += int(metrics[1].item())
            step += 1
            row = {'step': step, 'training_loss': metrics[0].item() / world,
                'gradient_norm': norm_value, 'learning_rate': optimizer.param_groups[0]['lr'],
                'tokens_seen': consumed, 'target_tokens_per_second': (consumed-initial_tokens) /
                max(time.monotonic()-started, 1e-9)}
            if step % training.evaluate_every == 0 or step == training.steps:
                row['validation'] = validation('validation')
            if rank == 0:
                record_metrics(folder, row)
                if step % training.log_every == 0 or step == 1 or step == training.steps:
                    from modelchef.progress import report
                    report(f'FSDP2 step {step}/{training.steps}: loss={row["training_loss"]:.4f}')
            if step % training.checkpoint_every == 0 or step == training.steps:
                checkpoint()
        checkpoint()
        result['validation'] = validation('validation')
        result['test'] = validation('test') if config.evaluation.evaluate_test else None
        model.eval()
        probe = blocks['validation'][0][:1].to('cuda')
        with torch.no_grad(), amp():
            expected = model(probe).detach()
        # Load the committed shards into the existing model; no second full model is allocated.
        with phases.measure('checkpoint_reload'):
            load_sharded(model, optimizer, latest, identities)
        with torch.no_grad(), amp():
            tolerance = .03 if precision == 'bf16' else 1e-5
            if not torch.allclose(expected, model(probe), atol=tolerance, rtol=tolerance):
                raise RuntimeError('FSDP2 checkpoint reload logits differ')
        from modelchef.benchmarks import evaluate_choices
        examples = json.loads((folder / 'benchmark-examples.json').read_text())
        with phases.measure('benchmark'):
            result['benchmark_scores'] = (evaluate_choices(model, tokenizer, examples, 'cuda',
                model_config.context_length, amp) if examples else None)
        ids = tokenizer.encode(config.evaluation.prompt).ids or [tokenizer.token_to_id('<bos>')]
        if config.tokenizer.add_bos and ids[0] != tokenizer.token_to_id('<bos>'):
            ids.insert(0, tokenizer.token_to_id('<bos>'))
        generated = torch.tensor([ids], device='cuda')
        with phases.measure('generation'), torch.no_grad():
            for _ in range(config.evaluation.generate_tokens):
                # Every rank forwards the SAME prefix. Only rank zero samples, then broadcasts.
                with amp():
                    scores = model(generated[:, -model_config.context_length:])[:, -1].float()
                token = torch.empty((1, 1), dtype=torch.long, device='cuda')
                if rank == 0:
                    scores /= config.evaluation.temperature
                    if config.evaluation.top_k:
                        boundary = scores.topk(min(config.evaluation.top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < boundary, -float('inf'))
                    token.copy_(torch.multinomial(scores.softmax(-1), 1))
                dist.broadcast(token, src=0)
                generated = torch.cat((generated, token), -1)
                if token.item() == tokenizer.token_to_id('<eos>'):
                    break
        result['generation'] = tokenizer.decode(generated[0].tolist())
        torch.cuda.synchronize()
        peaks = torch.tensor([max(torch.cuda.max_memory_allocated(), phases.cuda_peaks()['peak_allocated_bytes']),
                              max(torch.cuda.max_memory_reserved(), phases.cuda_peaks()['peak_reserved_bytes'])],
                             device='cuda', dtype=torch.int64)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        result.update(status='completed', checkpoint=str(latest), optimizer_steps=step,
            tokens_seen=consumed, checkpoint_reload_verified=True,
            checkpoint_recovery='passed', parameters=sum(p.numel() for p in model.parameters()),
            global_batch_size=training.micro_batch_size*training.gradient_accumulation*world,
            performance={'training_loop_seconds': time.monotonic()-started,
                'tokens_per_second': (consumed-initial_tokens)/max(time.monotonic()-started, 1e-9),
                'peak_cuda_allocated_bytes': peaks[0].item(),
                'peak_cuda_reserved_bytes': peaks[1].item(),
                'scope': 'Global supervised tokens; includes validation/checkpoint/reload work.'},
            phase_metrics=phases.rows)
        # Gather the small phase reports only, never parameter tensors.
        all_phases = [None] * world
        dist.all_gather_object(all_phases, phases.rows)
        if rank == 0:
            result['phase_metrics_by_rank'] = all_phases
            (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        # Refresh the recovery manifest with final reports without rewriting the immutable DCP.
        publish_sharded_to_s3(root, folder, latest)
        return result
    except Exception as error:
        if rank == 0 and folder is not None:
            result.update(status='failed', error=f'{type(error).__name__}: {error}')
            if 'latest' in locals() and latest:
                result['checkpoint'] = str(latest)
            (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        raise
    finally:
        if rank == 0 and folder is not None:
            close_metrics(folder)
        dist.destroy_process_group()
