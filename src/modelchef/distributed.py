"""torchrun DDP with one shared preparation pass and per-rank recovery state."""
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
from torch.nn.parallel import DistributedDataParallel

from modelchef.engines import create_model, load_parent, load_tokenizer
from modelchef.hardware import detect_hardware, resolve_runtime
from modelchef.objectives import evaluate_stage, objective
from modelchef.phase_metrics import PhaseMetrics
from modelchef.progress import report
from modelchef.training_metrics import close_metrics, record_metrics, tensorboard_info


def run_distributed(config, root, resume=None):
    from modelchef.pipeline import evaluate, run_pipeline
    from modelchef.run_config import RunConfig
    from modelchef.sharded_checkpoint import checkpoint_io_group, publish_checkpoint_to_s3
    if int(os.environ.get('WORLD_SIZE', '1')) < 2:
        raise ValueError('DDP requires torchrun --standalone --nproc_per_node=2 (or more)')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    local_rank = int(os.environ['LOCAL_RANK'])
    if config.training.accelerator in {'auto', 'cuda', 'rocm'} and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    runtime = resolve_runtime(config.training.model_dump(), detect_hardware())
    if runtime['device'] == 'mps':
        raise ValueError('PyTorch DDP does not support the MPS backend')
    device = runtime['device']
    dist.init_process_group('nccl' if device == 'cuda' else 'gloo',
                            timeout=timedelta(seconds=config.training.distributed_timeout_seconds))
    rank, world = dist.get_rank(), dist.get_world_size()
    try:
        preparation = [None]
        with checkpoint_io_group() as group:
            if rank == 0:
                try:
                    preparation[0] = {'report': run_pipeline(config, root, resume, prepare_only=True)}
                except Exception as error:  # noqa: BLE001 - broadcast preparation failure to waiting ranks
                    preparation[0] = {'error': f'{type(error).__name__}: {error}'}
            dist.broadcast_object_list(preparation, src=0, group=group)
        if 'error' in preparation[0]:
            raise RuntimeError('Distributed preparation failed: ' + preparation[0]['error'])
        folder = Path(preparation[0]['report']['run_dir'])
        config = RunConfig.model_validate_json((folder / 'config.json').read_text())
        recipe, training = config, config.training
        phases = PhaseMetrics(device, training.profile_phases)
        torch.backends.cuda.matmul.allow_tf32 = training.allow_tf32
        random.seed(training.seed + rank)
        torch.manual_seed(training.seed + rank)
        torch.use_deterministic_algorithms(training.deterministic)
        tokenizer = load_tokenizer(folder)
        from modelchef.disk_data import load_blocks
        blocks = load_blocks(folder)
        saved = torch.load(resume or preparation[0]['report']['checkpoint'],
                           map_location='cpu', weights_only=True)
        if resume and saved.get('world_size') != world:
            raise ValueError('DDP resume requires the same world size')
        with phases.measure('model_initialization'):
            model = create_model(recipe.model, tokenizer.get_vocab_size()).to(device)
            model.load_state_dict(saved['model'])
        reference = None
        if recipe.stage.kind in {'dpo', 'distill'}:
            from modelchef.chat import checkpoint_for
            source = (root / (recipe.stage.teacher_run if recipe.stage.kind == 'distill' else recipe.stage.parent_run)).resolve()
            source_config = RunConfig.model_validate_json((source / 'config.json').read_text())
            architecture = source_config.model if recipe.stage.kind == 'distill' else recipe.model
            reference = create_model(architecture, tokenizer.get_vocab_size()).to(device)
            state = torch.load(checkpoint_for(source), map_location='cpu', weights_only=True)['model']
            load_parent(reference, state, source_config.model)
            reference.requires_grad_(False).eval()
        if recipe.stage.kind == 'dpo':
            for network in [model, reference]:
                for layer in network.modules():
                    if isinstance(layer, torch.nn.Dropout):
                        layer.p = 0
                    if hasattr(layer, 'attention_dropout'):
                        layer.attention_dropout = 0.0
                    if hasattr(layer, 'attention_config'):
                        layer.attention_config = layer.attention_config.model_copy(update={'attention_dropout': 0})
        wrapped = DistributedDataParallel(model,
            device_ids=[local_rank] if device == 'cuda' else None,
            find_unused_parameters=recipe.model.type == 'decoder_moe', broadcast_buffers=False)
        groups = [{'params': [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
                   'weight_decay': training.weight_decay},
                  {'params': [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
                   'weight_decay': 0}]
        options = {'fused': True} if training.optimizer_kernel == 'fused' else (
            {'foreach': True} if training.optimizer_kernel == 'foreach' else {})
        with phases.measure('optimizer_initialization'):
            optimizer = torch.optim.AdamW(groups, lr=training.learning_rate, betas=training.betas,
                                         eps=training.optimizer_eps, **options)
        scaler = torch.amp.GradScaler('cuda', enabled=runtime['requires_grad_scaler'])
        sampler = torch.Generator().manual_seed(training.seed + rank)
        if resume:
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            state = saved['rank_states'][rank]
            sampler.set_state(state['sampler_rng'])
            torch.set_rng_state(state['torch_rng'])
            random.setstate(state['python_rng'])
            if device == 'cuda':
                torch.cuda.set_rng_state(state['cuda_rng'], device=local_rank)
        dtype = {'fp32': torch.float32, 'bf16': torch.bfloat16, 'fp16': torch.float16}[runtime['precision']]

        def amp():
            return nullcontext() if dtype == torch.float32 else torch.autocast(device, dtype=dtype)

        step, consumed = saved['step'], saved['tokens_seen']
        skipped_updates = saved.get('skipped_optimizer_updates', 0)
        if device == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        initial_step = step
        result = preparation[0]['report']
        result['tensorboard'] = tensorboard_info(folder)
        result.update(status='running', distributed={'backend': dist.get_backend(), 'world_size': world},
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

        def checkpoint():
            with checkpoint_io_group() as group:
                state = {'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
                         'python_rng': random.getstate(),
                         'cuda_rng': torch.cuda.get_rng_state(local_rank) if device == 'cuda' else []}
                states = [None] * world
                dist.all_gather_object(states, state, group=group)
                path = folder / 'checkpoints' / f'step-{step:06d}.pt'
                error = [None]
                if rank == 0:
                    payload = {**saved, **states[0], 'model': model.state_dict(),
                               'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                               'rank_states': states, 'world_size': world, 'step': step,
                               'tokens_seen': consumed, 'skipped_optimizer_updates': skipped_updates,
                               'cuda_rng': [states[0]['cuda_rng']] if device == 'cuda' else []}
                    try:
                        with phases.measure('checkpoint'):
                            from modelchef.checkpointing import atomic_save
                            atomic_save(payload, path)
                    except OSError as exc:
                        error[0] = str(exc)
                dist.broadcast_object_list(error, src=0, group=group)
                if error[0]:
                    raise OSError(error[0])
                dist.barrier(group=group)
            publish_checkpoint_to_s3(root, folder, path)
            return path

        if os.environ.get('MODELCHEF_RECOVERY_S3_URI') and resume:
            publish_checkpoint_to_s3(root, folder, Path(resume))
        elif os.environ.get('MODELCHEF_RECOVERY_S3_URI') or (training.checkpoint_at_start and not resume):
            checkpoint()
        while step < training.steps:
            expired = torch.tensor(int(time.monotonic() - started > training.max_runtime_seconds), device=device)
            dist.all_reduce(expired, op=dist.ReduceOp.MAX)
            if expired.item():
                path = checkpoint()
                raise TimeoutError(f'DDP runtime cap reached; resume from {path}')
            factor = min(1.0, (step + 1) / max(1, training.warmup_steps))
            if step >= training.warmup_steps and training.schedule == 'cosine':
                fraction = (step - training.warmup_steps) / max(1, training.steps - training.warmup_steps - 1)
                factor = training.min_learning_rate_ratio + (1 - training.min_learning_rate_ratio) * (1 + math.cos(math.pi * fraction)) / 2
            for group in optimizer.param_groups:
                group['lr'] = training.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            loss_total, tokens = 0.0, 0
            for _ in range(training.gradient_accumulation):
                indices = torch.randint(len(blocks['train'][0]), (training.micro_batch_size,), generator=sampler)
                parts = tuple(part[indices].to(device) for part in blocks['train'])
                with phases.measure('forward'), amp():
                    loss, _ = objective(wrapped, parts, recipe, reference, tokenizer)
                    if getattr(model, 'auxiliary_loss', None) is not None:
                        loss = loss + model.auxiliary_loss
                invalid = torch.tensor(int(not torch.isfinite(loss)), device=device)
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                if invalid.item():
                    raise FloatingPointError('Non-finite DDP loss on at least one rank')
                with phases.measure('backward'):
                    scaler.scale(loss / training.gradient_accumulation).backward()
                loss_total += loss.item() / training.gradient_accumulation
                tokens += sum((part != -100).sum().item() for part in parts[1::2])
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training.gradient_clip_norm or float('inf'),
                                                 error_if_nonfinite=False)
            invalid = torch.tensor(int(not torch.isfinite(norm)), device=device)
            dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
            if invalid.item():
                if not scaler.is_enabled():
                    raise FloatingPointError('Non-finite DDP gradient norm on at least one rank')
                scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
                skipped_updates += 1
                if rank == 0:
                    report(f'FP16 gradient overflow: all ranks skipped the update; '
                           f'loss scale={scaler.get_scale():g}, retrying step {step + 1}.')
                continue
            with phases.measure('optimizer_update'):
                scaler.step(optimizer)
                scaler.update()
            metrics = torch.tensor([loss_total, tokens], dtype=torch.float64, device=device)
            dist.all_reduce(metrics)
            consumed += int(metrics[1].item())
            step += 1
            if rank == 0:
                row = {'step': step, 'training_loss': metrics[0].item() / world,
                       'gradient_norm': norm.item(),
                       'tokens_seen': consumed, 'skipped_optimizer_updates': skipped_updates,
                       'learning_rate': optimizer.param_groups[0]['lr']}
                if step % training.evaluate_every == 0 or step == training.steps:
                    with phases.measure('validation'):
                        row['validation'] = evaluate(model, blocks['validation'][:2], recipe.evaluation.batches,
                                                     recipe.evaluation.batch_size or training.micro_batch_size, device, amp)
                record_metrics(folder, row)
                if step % training.log_every == 0 or step == 1 or step == training.steps:
                    report(f'DDP step {step}/{training.steps}: global mean loss={row["training_loss"]:.4f}')
            if step % training.checkpoint_every == 0 or step == training.steps:
                checkpoint()
        path = checkpoint()
        if device == 'cuda':
            torch.cuda.synchronize()
        elapsed_loop = time.monotonic() - started
        peaks = torch.tensor([max(torch.cuda.max_memory_allocated(), phases.cuda_peaks()['peak_allocated_bytes']),
                              max(torch.cuda.max_memory_reserved(), phases.cuda_peaks()['peak_reserved_bytes'])]
                              if device == 'cuda' else [0, 0], device=device, dtype=torch.int64)
        dist.all_reduce(peaks, op=dist.ReduceOp.MAX)
        if rank == 0:
            result['performance'] = {
                'training_loop_seconds': elapsed_loop,
                'tokens_per_second': (consumed - saved['tokens_seen']) / elapsed_loop,
                'peak_cuda_allocated_bytes': peaks[0].item() if device == 'cuda' else None,
                'peak_cuda_reserved_bytes': peaks[1].item() if device == 'cuda' else None,
                'scope': 'Global tokens; maximum per-rank CUDA peaks; loop includes evaluation/checkpoints.'}
            result.update(status='completed', checkpoint=str(path), optimizer_steps=step,
                          tokens_seen=consumed, checkpoint_reload_verified=True,
                          skipped_optimizer_updates=skipped_updates,
                          global_batch_size=training.micro_batch_size * training.gradient_accumulation * world,
                          tokens_per_second=(step - initial_step) * training.micro_batch_size *
                              training.gradient_accumulation * recipe.model.context_length * world / (time.monotonic() - started))
            result['stage_evaluation'] = {}
            result['test'] = None
            result['test_evaluation'] = 'evaluated' if recipe.evaluation.evaluate_test else 'withheld'
            for split in ['validation', 'test'] if recipe.evaluation.evaluate_test else ['validation']:
                with phases.measure('final_evaluation'):
                    result[split] = evaluate(model, blocks[split][:2], recipe.evaluation.batches,
                                             recipe.evaluation.batch_size or training.micro_batch_size, device, amp)
                result['stage_evaluation'][split] = evaluate_stage(model, reference, blocks[split],
                                                                  tokenizer, recipe, device, amp)
            model.eval()
            probe = blocks['validation'][0][:1].to(device)
            tolerance = 0.03 if dtype == torch.bfloat16 else 0.003 if dtype == torch.float16 else 1e-5
            with torch.no_grad(), amp():
                expected = model(probe).detach()
            with phases.measure('checkpoint_reload'):
                loaded = torch.load(path, map_location='cpu', mmap=True, weights_only=True)
                model.load_state_dict(loaded['model'])
                del loaded
            with torch.no_grad(), amp():
                if not torch.allclose(expected, model(probe), atol=tolerance, rtol=tolerance):
                    raise RuntimeError('DDP checkpoint reload differs')
            from modelchef.benchmarks import evaluate_choices
            examples = json.loads((folder / 'benchmark-examples.json').read_text())
            result['benchmark'] = (evaluate_choices(model, tokenizer, examples, device,
                                                    recipe.model.context_length, amp) if examples else None)
            ids = tokenizer.encode(recipe.evaluation.prompt).ids or [tokenizer.token_to_id('<bos>')]
            generated = torch.tensor([ids], device=device)
            with torch.no_grad():
                for _ in range(recipe.evaluation.generate_tokens):
                    with amp():
                        scores = model(generated[:, -recipe.model.context_length:])[:, -1].float()
                    scores /= recipe.evaluation.temperature
                    if recipe.evaluation.top_k:
                        boundary = scores.topk(min(recipe.evaluation.top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < boundary, -float('inf'))
                    token = torch.multinomial(scores.softmax(-1), 1)
                    generated = torch.cat((generated, token), -1)
                    if token.item() == tokenizer.token_to_id('<eos>'):
                        break
            result['sample'] = tokenizer.decode(generated[0].tolist())
            (folder / 'report.json').write_text(json.dumps(result, indent=2))
        with checkpoint_io_group() as group:
            phase_reports = [None] * world
            dist.all_gather_object(phase_reports, phases.rows, group=group)
            if rank == 0:
                result['phase_metrics_by_rank'] = phase_reports
                (folder / 'report.json').write_text(json.dumps(result, indent=2))
            final_report = [result if rank == 0 else None]
            dist.broadcast_object_list(final_report, src=0, group=group)
        publish_checkpoint_to_s3(root, folder, path)
        return final_report[0]
    except Exception as error:
        if rank == 0 and 'folder' in locals():
            failure = dict(locals().get('result', preparation[0]['report']))
            failure.update(status='failed', error=f'{type(error).__name__}: {error}')
            latest = sorted((folder / 'checkpoints').glob('step-*.pt'))
            if latest:
                failure['checkpoint'] = str(latest[-1])
            (folder / 'report.json').write_text(json.dumps(failure, indent=2))
        raise
    finally:
        if rank == 0 and 'folder' in locals():
            close_metrics(folder)
        dist.destroy_process_group()
