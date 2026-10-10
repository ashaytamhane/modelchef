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
from modelchef.progress import report


def run_distributed(config, root, resume=None):
    from modelchef.pipeline import evaluate, run_pipeline
    from modelchef.run_config import RunConfig
    if int(os.environ.get('WORLD_SIZE', '1')) < 2:
        raise ValueError('DDP requires torchrun --standalone --nproc_per_node=2 (or more)')
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
        if rank == 0:
            try:
                preparation[0] = {'report': run_pipeline(config, root, resume, prepare_only=True)}
            except Exception as error:  # noqa: BLE001 - broadcast every preparation failure to waiting ranks
                preparation[0] = {'error': f'{type(error).__name__}: {error}'}
        dist.broadcast_object_list(preparation, src=0)
        if 'error' in preparation[0]:
            raise RuntimeError('Distributed preparation failed: ' + preparation[0]['error'])
        folder = Path(preparation[0]['report']['run_dir'])
        config = RunConfig.model_validate_json((folder / 'config.json').read_text())
        c, t = config, config.training
        random.seed(t.seed + rank)
        torch.manual_seed(t.seed + rank)
        torch.use_deterministic_algorithms(t.deterministic)
        tokenizer = load_tokenizer(folder)
        blocks = torch.load(folder / 'tokens.pt', weights_only=True)
        saved = torch.load(resume or preparation[0]['report']['checkpoint'],
                           map_location='cpu', weights_only=True)
        if resume and saved.get('world_size') != world:
            raise ValueError('DDP resume requires the same world size')
        model = create_model(c.model, tokenizer.get_vocab_size()).to(device)
        model.load_state_dict(saved['model'])
        reference = None
        if c.stage.kind in {'dpo', 'distill'}:
            from modelchef.chat import checkpoint_for
            source = (root / (c.stage.teacher_run if c.stage.kind == 'distill' else c.stage.parent_run)).resolve()
            source_config = RunConfig.model_validate_json((source / 'config.json').read_text())
            architecture = source_config.model if c.stage.kind == 'distill' else c.model
            reference = create_model(architecture, tokenizer.get_vocab_size()).to(device)
            state = torch.load(checkpoint_for(source), map_location='cpu', weights_only=True)['model']
            load_parent(reference, state, source_config.model)
            reference.requires_grad_(False).eval()
        if c.stage.kind == 'dpo':
            for network in [model, reference]:
                for layer in network.modules():
                    if isinstance(layer, torch.nn.Dropout):
                        layer.p = 0
                    if hasattr(layer, 'attention_dropout'):
                        layer.attention_dropout = 0.0
                    if hasattr(layer, 'c'):
                        layer.c = layer.c.model_copy(update={'attention_dropout': 0})
        wrapped = DistributedDataParallel(model,
            device_ids=[local_rank] if device == 'cuda' else None,
            find_unused_parameters=c.model.type == 'decoder_moe', broadcast_buffers=False)
        groups = [{'params': [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
                   'weight_decay': t.weight_decay},
                  {'params': [p for p in model.parameters() if p.requires_grad and p.ndim < 2],
                   'weight_decay': 0}]
        options = {'fused': True} if t.optimizer_kernel == 'fused' else (
            {'foreach': True} if t.optimizer_kernel == 'foreach' else {})
        optimizer = torch.optim.AdamW(groups, lr=t.learning_rate, betas=t.betas,
                                     eps=t.optimizer_eps, **options)
        scaler = torch.amp.GradScaler('cuda', enabled=runtime['requires_grad_scaler'])
        sampler = torch.Generator().manual_seed(t.seed + rank)
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
        started = time.monotonic()
        initial_step = step
        result = preparation[0]['report']
        result.update(status='running', distributed={'backend': dist.get_backend(), 'world_size': world},
                      trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))

        def checkpoint():
            state = {'sampler_rng': sampler.get_state(), 'torch_rng': torch.get_rng_state(),
                     'python_rng': random.getstate(),
                     'cuda_rng': torch.cuda.get_rng_state(local_rank) if device == 'cuda' else []}
            states = [None] * world
            dist.all_gather_object(states, state)
            path = folder / 'checkpoints' / f'step-{step:06d}.pt'
            if rank == 0:
                payload = {**saved, **states[0], 'model': model.state_dict(),
                           'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                           'rank_states': states, 'world_size': world, 'step': step,
                           'tokens_seen': consumed, 'skipped_optimizer_updates': skipped_updates,
                           'cuda_rng': [states[0]['cuda_rng']] if device == 'cuda' else []}
                temporary = path.with_suffix('.tmp')
                torch.save(payload, temporary)
                temporary.replace(path)
            dist.barrier()
            return path

        while step < t.steps:
            expired = torch.tensor(int(time.monotonic() - started > t.max_runtime_seconds), device=device)
            dist.all_reduce(expired, op=dist.ReduceOp.MAX)
            if expired.item():
                path = checkpoint()
                raise TimeoutError(f'DDP runtime cap reached; resume from {path}')
            factor = min(1.0, (step + 1) / max(1, t.warmup_steps))
            if step >= t.warmup_steps and t.schedule == 'cosine':
                fraction = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps - 1)
                factor = t.min_learning_rate_ratio + (1 - t.min_learning_rate_ratio) * (1 + math.cos(math.pi * fraction)) / 2
            for group in optimizer.param_groups:
                group['lr'] = t.learning_rate * factor
            optimizer.zero_grad(set_to_none=True)
            loss_total, tokens = 0.0, 0
            for _ in range(t.gradient_accumulation):
                indices = torch.randint(len(blocks['train'][0]), (t.micro_batch_size,), generator=sampler)
                parts = tuple(part[indices].to(device) for part in blocks['train'])
                with amp():
                    loss, _ = objective(wrapped, parts, c, reference, tokenizer)
                    if getattr(model, 'auxiliary_loss', None) is not None:
                        loss = loss + model.auxiliary_loss
                invalid = torch.tensor(int(not torch.isfinite(loss)), device=device)
                dist.all_reduce(invalid, op=dist.ReduceOp.MAX)
                if invalid.item():
                    raise FloatingPointError('Non-finite DDP loss on at least one rank')
                scaler.scale(loss / t.gradient_accumulation).backward()
                loss_total += loss.item() / t.gradient_accumulation
                tokens += sum((part != -100).sum().item() for part in parts[1::2])
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.gradient_clip_norm,
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
            scaler.step(optimizer)
            scaler.update()
            metrics = torch.tensor([loss_total, tokens], dtype=torch.float64, device=device)
            dist.all_reduce(metrics)
            consumed += int(metrics[1].item())
            step += 1
            if rank == 0:
                row = {'step': step, 'training_loss': metrics[0].item() / world,
                       'tokens_seen': consumed, 'skipped_optimizer_updates': skipped_updates,
                       'learning_rate': optimizer.param_groups[0]['lr']}
                if step % t.evaluate_every == 0 or step == t.steps:
                    row['validation'] = evaluate(model, blocks['validation'][:2], c.evaluation.batches,
                                                 t.micro_batch_size, device, amp)
                with (folder / 'metrics.jsonl').open('a') as stream:
                    stream.write(json.dumps(row) + '\n')
                if step % t.log_every == 0 or step == 1 or step == t.steps:
                    report(f'DDP step {step}/{t.steps}: global mean loss={row["training_loss"]:.4f}')
            if step % t.checkpoint_every == 0 or step == t.steps:
                checkpoint()
        path = checkpoint()
        if rank == 0:
            result.update(status='completed', checkpoint=str(path), optimizer_steps=step,
                          tokens_seen=consumed, checkpoint_reload_verified=True,
                          skipped_optimizer_updates=skipped_updates,
                          global_batch_size=t.micro_batch_size * t.gradient_accumulation * world,
                          tokens_per_second=(step - initial_step) * t.micro_batch_size *
                              t.gradient_accumulation * c.model.context_length * world / (time.monotonic() - started))
            result['stage_evaluation'] = {}
            for split in ['validation', 'test']:
                result[split] = evaluate(model, blocks[split][:2], c.evaluation.batches,
                                         t.micro_batch_size, device, amp)
                result['stage_evaluation'][split] = evaluate_stage(model, reference, blocks[split],
                                                                  tokenizer, c, device, amp)
            restored = create_model(c.model, tokenizer.get_vocab_size()).to(device).eval()
            restored.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
            model.eval()
            probe = blocks['test'][0][:1].to(device)
            with torch.no_grad():
                if not torch.allclose(model(probe), restored(probe), atol=1e-6, rtol=1e-5):
                    raise RuntimeError('DDP checkpoint reload differs')
            from modelchef.benchmarks import evaluate_choices
            examples = json.loads((folder / 'benchmark-examples.json').read_text())
            result['benchmark'] = (evaluate_choices(model, tokenizer, examples, device,
                                                    c.model.context_length, amp) if examples else None)
            ids = tokenizer.encode(c.evaluation.prompt).ids or [tokenizer.token_to_id('<bos>')]
            generated = torch.tensor([ids], device=device)
            with torch.no_grad():
                for _ in range(c.evaluation.generate_tokens):
                    with amp():
                        scores = model(generated[:, -c.model.context_length:])[:, -1].float()
                    scores /= c.evaluation.temperature
                    if c.evaluation.top_k:
                        boundary = scores.topk(min(c.evaluation.top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < boundary, -float('inf'))
                    token = torch.multinomial(scores.softmax(-1), 1)
                    generated = torch.cat((generated, token), -1)
                    if token.item() == tokenizer.token_to_id('<eos>'):
                        break
            result['sample'] = tokenizer.decode(generated[0].tolist())
            (folder / 'report.json').write_text(json.dumps(result, indent=2))
        final_report = [result if rank == 0 else None]
        dist.broadcast_object_list(final_report, src=0)
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
        dist.destroy_process_group()
