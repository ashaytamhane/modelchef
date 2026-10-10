"""Narrated, bounded pretraining from raw documents to a tested checkpoint."""
import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from modelchef.benchmarks import acquire_benchmark, evaluate_choices, select_benchmark
from modelchef.corpus import acquire, digest, file_digest, process
from modelchef.engines import create_model, load_parent, load_tokenizer
from modelchef.execution import check_custom_kernel, configure_execution, plugin_fingerprint
from modelchef.hardware import detect_hardware, resolve_runtime
from modelchef.objectives import evaluate_stage, objective, pack_structured, prepare_structured


class Narrator:
    def __init__(self, folder, detailed):
        self.folder, self.detailed = folder, detailed

    def say(self, message, detail=False):
        event = {"time": time.time(), "message": message}
        with (self.folder / "events.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(event) + "\n")
        if self.detailed or not detail:
            print(message, flush=True)

    def stage(self, title, explanation):
        self.say(f"\n=== {title} ===")
        self.say(explanation, detail=True)


def train_tokenizer(documents, c):
    if c.algorithm != 'byte_bpe':
        raise ValueError('Pretrained tokenizers must be reused from a parent, not retrained')
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=c.vocab_size, min_frequency=c.min_frequency,
        special_tokens=c.special_tokens, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False)
    tokenizer.train_from_iterator((row["text"] for row in documents), trainer=trainer)
    return tokenizer


def pack(documents, tokenizer, c, context, token_limit):
    tokens = []
    for row in documents:
        ids = tokenizer.encode(row["text"], add_special_tokens=False).ids
        if c.add_bos:
            ids.insert(0, tokenizer.token_to_id("<bos>"))
        if c.append_eos:
            ids.append(tokenizer.token_to_id("<eos>"))
        tokens.extend(ids[:max(0, token_limit - len(tokens))])
        if len(tokens) >= token_limit:
            break
    blocks = (len(tokens) - 1) // context
    if blocks < 1:
        raise ValueError("Split has too few tokens for one context block; increase sample/token cap")
    stream = torch.tensor(tokens, dtype=torch.long)
    # Cross-document attention is allowed; EOS separates documents. No padded losses.
    return (stream[:blocks * context].view(blocks, context),
            stream[1:blocks * context + 1].view(blocks, context)), len(tokens)


def evaluate(model, blocks, batches, batch_size, device, amp):
    model.eval()
    loss_sum, tokens = 0.0, 0
    with torch.no_grad():
        for start in range(0, min(len(blocks[0]), batches * batch_size), batch_size):
            x, y = (part[start:start + batch_size].to(device) for part in blocks)
            with amp():
                logits = model(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten(), reduction="sum")
            loss_sum += loss.item()
            tokens += (y != -100).sum().item()
    model.train()
    nll = loss_sum / tokens
    return {"loss": nll, "perplexity": math.exp(min(nll, 80)), "tokens": tokens,
            "perplexity_capped": nll > 80}


def run_pipeline(config, root, resume=None, prepare_only=False):
    config.validate_execution()
    root = Path(root).resolve()
    if config.training.distributed == 'ddp' and not prepare_only:
        from modelchef.distributed import run_distributed
        return run_distributed(config, root, resume)
    if config.model.peft.method == 'qlora':
        from modelchef.chat import export_model
        from modelchef.run_config import RunConfig
        parent_dir = (root / config.stage.parent_run).resolve()
        parent_recipe = RunConfig.model_validate_json((parent_dir / 'config.json').read_text())
        if parent_recipe.model.peft.method == 'none':
            source = export_model(parent_dir)
            config = config.model_copy(deep=True)
            config.model.pretrained_path = str(source.resolve())
            config.model.pretrained_config_sha256 = digest((source / 'config.json').read_bytes())
    recipe = config.model_dump(mode="json")
    fingerprint = digest(json.dumps(recipe, sort_keys=True).encode())
    folder = (Path(resume).resolve().parent.parent if resume else
              root / config.output_dir / (config.name + "-" + uuid.uuid4().hex[:12]))
    folder.mkdir(parents=True, exist_ok=True)
    from modelchef.run_config import RunConfig
    if resume and RunConfig.model_validate_json((folder / "config.json").read_text()).model_dump(
            mode="json") != recipe:
        raise ValueError("Resume configuration differs; refusing to modify the saved run")
    narrate = Narrator(folder, config.explain == "detailed")
    report = {"status": "running", "run_dir": str(folder), "config_sha256": fingerprint}
    (folder / "config.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    try:
        return _execute(config, root, folder, fingerprint, narrate, report, resume, prepare_only)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        narrate.say(f"Pipeline stopped: {error}. Evidence remains in {folder}.")
        raise


def _execute(c, root, folder, fingerprint, n, report, resume, prepare_only=False):
    if c.model.backend == 'huggingface' and c.stage.kind == 'pretrain':
        raise ValueError('Import the pretrained model, then choose a continuation/post-training stage')
    saved = None
    if resume:
        saved = torch.load(resume, map_location='cpu', weights_only=True)
        required = ['raw.jsonl', 'source-manifest.json', 'benchmark-examples.json',
                    'benchmark-manifest.json', 'tokenizer.json', 'tokens.pt']
        missing = [name for name in required if not (folder / name).is_file()]
        if missing:
            raise ValueError('Resume requires the saved run snapshots; missing: ' + ', '.join(missing))
        if saved['tokenizer_sha256'] != digest((folder / 'tokenizer.json').read_bytes()):
            raise ValueError('Saved tokenizer checksum changed')
        roles_path = folder / 'tokenizer-roles.json'
        roles_hash = digest(roles_path.read_bytes()) if roles_path.exists() else None
        if 'tokenizer_roles_sha256' in saved and saved['tokenizer_roles_sha256'] != roles_hash:
            raise ValueError('Saved tokenizer special-token identities changed')
        if saved.get('tokens_file_sha256') and saved['tokens_file_sha256'] != file_digest(
                folder / 'tokens.pt'):
            raise ValueError('Saved token file checksum changed')
    processing_sources = {str((root / path).resolve()): file_digest(root / path)
                          for path in c.processing.benchmark_files}
    processing_manifest = folder / 'processing-sources.json'
    if resume and processing_manifest.exists():
        if json.loads(processing_manifest.read_text()) != processing_sources:
            raise ValueError('Configured benchmark decontamination files changed since this run started')
    elif not resume:
        processing_manifest.write_text(json.dumps(processing_sources, indent=2), encoding='utf-8')
    n.stage("1. Configuration and hardware", "Validate the complete recipe before data access. "
            "Detect this machine's accelerator; no AWS instance is launched by this command.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(c.training.seed)
    torch.manual_seed(c.training.seed)
    torch.use_deterministic_algorithms(c.training.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = c.training.allow_tf32
    runtime = resolve_runtime(c.training.model_dump(), detect_hardware())
    device = runtime["device"]
    if c.model.quantization == 'nvfp4' and device != 'cuda':
        raise ValueError('NVFP4 requires a CUDA runtime')
    if c.model.peft.method == 'qlora' and device != 'cuda':
        raise ValueError('QLoRA requires a CUDA runtime')
    if c.model.attention_kernel not in {'auto', 'math'} and device != 'cuda':
        raise ValueError('Selected optimized attention kernel requires CUDA/ROCm')
    if c.training.optimizer_kernel == 'fused' and device != 'cuda':
        raise ValueError('Fused optimizer selection currently requires CUDA/ROCm')
    if c.model.custom_swiglu_path:
        kernel_hash = plugin_fingerprint(root / c.model.custom_swiglu_path)
        saved_kernel = folder / 'custom-kernel.sha256'
        if resume and saved_kernel.read_text() != kernel_hash:
            raise ValueError('Custom CUDA source changed since this run started')
        report['custom_kernel'] = check_custom_kernel(root / c.model.custom_swiglu_path)
        saved_kernel.write_text(kernel_hash, encoding='utf-8')
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        runtime["precision"]]
    def amp():
        return (nullcontext() if dtype == torch.float32 else
                torch.autocast(device_type=device, dtype=dtype))
    from modelchef.provenance import environment_provenance
    environment = {**runtime, **environment_provenance(root)}
    report['environment'] = environment
    if resume:
        with (folder / 'resume-environments.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'time': time.time(), **environment}) + '\n')
    else:
        (folder / 'environment.json').write_text(json.dumps(environment, indent=2), encoding='utf-8')
    n.say(f"Device={device}, precision={runtime['precision']}, seed={c.training.seed}.")

    n.stage("2. Acquire a bounded corpus snapshot", f"Source={c.data.dataset}, domain={c.data.domain}. "
            f"Limits: {c.data.max_documents} documents and {c.data.max_download_bytes:,} bytes. "
            "The saved SHA256 identifies the exact sample; raw text is never committed to Git.")
    snapshot = folder / 'raw.jsonl'
    if resume:
        records = [json.loads(line) for line in snapshot.read_text(encoding='utf-8').splitlines()]
        source = json.loads((folder / 'source-manifest.json').read_text())
        if digest(snapshot.read_bytes()) != source['raw_sha256']:
            raise ValueError('Run snapshot checksum changed')
    else:
        records, source = acquire(c, root, lambda message: n.say(message, detail=True))
        snapshot.write_bytes(b''.join((json.dumps(row, ensure_ascii=False) + '\n').encode()
                                     for row in records))
    if not resume:
        (folder / "source-manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    n.stage("2b. Select domain benchmarks", f"Domain '{c.data.domain}' selects "
            f"'{select_benchmark(c)}'. Download only the configured small evaluation sample. "
            "These examples are excluded from tokenizer/training inputs and used only for evaluation.")
    benchmark_snapshot = folder / 'benchmark-examples.json'
    if resume:
        benchmark_source = json.loads((folder / 'benchmark-manifest.json').read_text())
        if benchmark_source.get('sha256') and digest(benchmark_snapshot.read_bytes()) != benchmark_source['sha256']:
            raise ValueError('Saved benchmark sample checksum changed')
        examples = json.loads(benchmark_snapshot.read_text())
        if benchmark_source.get('name') == 'none' and examples:
            raise ValueError('Disabled benchmark snapshot must be empty')
    else:
        examples, benchmark_source = acquire_benchmark(c, root)
        benchmark_snapshot.write_text(json.dumps(examples), encoding='utf-8')
        benchmark_source['sha256'] = digest(benchmark_snapshot.read_bytes())
        (folder / "benchmark-manifest.json").write_text(
            json.dumps(benchmark_source, indent=2), encoding="utf-8")
    processing_config = c.model_copy(deep=True)
    if examples:
        benchmark_path = folder / "benchmark-text.txt"
        if not resume:
            benchmark_path.write_text("\n".join(e["decontamination_text"] for e in examples),
                                      encoding="utf-8")
        elif not benchmark_path.is_file() or benchmark_path.read_text(encoding='utf-8') != '\n'.join(
                e['decontamination_text'] for e in examples):
            raise ValueError('Saved benchmark decontamination text changed')
        processing_config.processing.benchmark_files.append(str(benchmark_path))

    n.stage("3. Clean, filter, deduplicate and split", "Normalize Unicode/HTML/whitespace, reject "
            "length/quality/repetition failures, apply domain keywords, regex PII handling and "
            "lexical deduplication before document-level seeded train/validation/test splitting. "
            "Configured benchmark files enable exact n-gram decontamination; otherwise it is unconfigured.")
    structured = c.stage.kind in {'sft', 'dpo', 'rlvr', 'distill'}
    splits, processing = (prepare_structured(records, processing_config, root) if structured else
                          process(records, processing_config, root))
    if not resume:
        for name, rows in splits.items():
            (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                               encoding="utf-8")
        (folder / "processing.json").write_text(json.dumps(processing, indent=2), encoding="utf-8")
    n.say(f"Filter counts: {processing['counts']}; document splits: {processing['split_documents']}.")
    n.say(f"Benchmark decontamination: {processing['decontamination']}. "
          "PII/dedup checks are limited heuristics, not complete guarantees.")

    n.stage("4. Prepare the tokenizer", "From-scratch pretraining learns byte-level BPE on training "
            "documents only. Later stages retain the parent's vocabulary and special-token identities.")
    parent = None
    if c.stage.kind != 'pretrain':
        from modelchef.chat import checkpoint_for
        from modelchef.run_config import RunConfig
        parent_folder = (root / c.stage.parent_run).resolve()
        parent_config = RunConfig.model_validate_json((parent_folder / 'config.json').read_text())
        execution_fields = {'quantization', 'attention_kernel', 'custom_swiglu_path', 'peft',
                            'gradient_checkpointing'}
        if c.model.peft.method == 'qlora':
            execution_fields |= {'pretrained_path', 'pretrained_config_sha256'}
        if c.model.model_dump(exclude=execution_fields) != parent_config.model.model_dump(
                exclude=execution_fields) or c.tokenizer != parent_config.tokenizer:
            raise ValueError('Parent architecture/tokenizer settings must match; copy the parent recipe')
        tokenizer = load_tokenizer(parent_folder)
        parent_path = checkpoint_for(parent_folder)
        parent = torch.load(parent_path, map_location='cpu', weights_only=True)['model']
        report['parent'] = {'checkpoint_sha256': file_digest(parent_path),
                            'tokenizer_sha256': digest((parent_folder / 'tokenizer.json').read_bytes())}
        parent_manifest = folder / 'parent-manifest.json'
        if resume and json.loads(parent_manifest.read_text()) != report['parent']:
            raise ValueError('Parent checkpoint/tokenizer changed since this stage started')
        parent_manifest.write_text(json.dumps(report['parent']), encoding='utf-8')
        n.say(f'Reuse parent vocabulary and weights for {c.stage.kind}; no vocabulary retraining.')
    elif resume:
        tokenizer = load_tokenizer(folder)
    else:
        tokenizer = train_tokenizer(splits["train"], c.tokenizer)
    if not resume:
        tokenizer.save(str(folder / "tokenizer.json"))
    if not resume and parent is not None and hasattr(tokenizer, 'roles'):
        (folder / 'tokenizer-roles.json').write_text(json.dumps(tokenizer.roles), encoding='utf-8')
    tokenizer_hash = digest((folder / "tokenizer.json").read_bytes())
    roles_path = folder / 'tokenizer-roles.json'
    roles_hash = digest(roles_path.read_bytes()) if roles_path.exists() else None
    n.say(f"BPE vocabulary={tokenizer.get_vocab_size()} (requested {c.tokenizer.vocab_size}).")

    n.stage("5. Tokenize and pack next-token examples", "Encode each split separately, insert configured "
            "BOS/EOS tokens, concatenate within splits, then form fixed-length input/shifted-target pairs. "
            "Tail tokens are dropped; cross-document attention is allowed. Enforce a total token cap.")
    blocks, token_counts = {}, {}
    fractions = {"validation": c.processing.validation_fraction, "test": c.processing.test_fraction,
                 "train": 1 - c.processing.validation_fraction - c.processing.test_fraction}
    for name, rows in splits.items():
        limit = int(c.data.max_tokens * fractions[name])
        blocks[name], token_counts[name] = (pack_structured(rows, tokenizer, c, limit) if structured
            else pack(rows, tokenizer, c.tokenizer, c.model.context_length, limit))
    # Logical data identity is independent of torch ZIP container metadata.
    data_hash = hashlib.sha256(b"".join(part.numpy().tobytes() for name in sorted(blocks)
                                       for part in blocks[name])).hexdigest()
    if resume:
        if (saved['config_sha256'], saved['data_sha256'], saved['tokenizer_sha256']) != (
                fingerprint, data_hash, tokenizer_hash):
            raise ValueError('Resume recipe/data/tokenizer differs from checkpoint')
    else:
        torch.save(blocks, folder / 'tokens.pt')
    token_hash = file_digest(folder / 'tokens.pt')
    n.say(f"Token counts: {token_counts}; context={c.model.context_length}. "
          "This is bounded pretraining data, not a full corpus download.")

    n.stage("6. Construct the training model", "The native decoder uses causal attention, RoPE, GQA, "
            "RMSNorm and SwiGLU. Pretraining starts from random weights; later stages restore a parent. "
            "The HF adapter loads local pretrained models without remote code or implicit downloads.")
    model = create_model(c.model, tokenizer.get_vocab_size()).to(device)
    if parent is not None:
        load_parent(model, parent, parent_config.model)
    if c.model.quantization == 'fp4_fake':
        n.say('Experimental software FP4 fake quantization enabled: FP32/BF16 master weights and '
              'straight-through gradients. No packed-weight storage or native FP4 acceleration.')
    reference = None
    if c.stage.kind in {'dpo', 'rlvr'}:
        reference = create_model(c.model, tokenizer.get_vocab_size()).to(device)
        load_parent(reference, parent, parent_config.model)
        reference.requires_grad_(False).eval()
        # Preference/RL likelihoods must not fluctuate because of dropout.
        for network in [model, reference]:
            for module in network.modules():
                if isinstance(module, torch.nn.Dropout):
                    module.p = 0
                if hasattr(module, 'attention_dropout'):
                    module.attention_dropout = 0.0
            if hasattr(network, 'blocks'):
                for block in network.blocks:
                    block.attention.c = block.attention.c.model_copy(update={'attention_dropout': 0})
    if c.stage.kind == 'distill':
        from modelchef.chat import checkpoint_for
        from modelchef.run_config import RunConfig
        teacher_folder = (root / c.stage.teacher_run).resolve()
        if digest((teacher_folder / 'tokenizer.json').read_bytes()) != tokenizer_hash:
            raise ValueError('Token distillation requires identical student/teacher tokenizers')
        if load_tokenizer(teacher_folder).roles != getattr(tokenizer, 'roles', {}):
            raise ValueError('Teacher/student special-token identities differ')
        teacher_config = RunConfig.model_validate_json((teacher_folder / 'config.json').read_text())
        if teacher_config.model.context_length < c.model.context_length:
            raise ValueError('Teacher context must cover the student context')
        reference = create_model(teacher_config.model, tokenizer.get_vocab_size()).to(device)
        teacher_path = checkpoint_for(teacher_folder)
        reference.load_state_dict(torch.load(teacher_path, map_location='cpu', weights_only=True)['model'])
        reference.requires_grad_(False).eval()
        report['teacher_checkpoint_sha256'] = file_digest(teacher_path)
        teacher_manifest = folder / 'teacher-manifest.json'
        if resume and teacher_manifest.read_text() != report['teacher_checkpoint_sha256']:
            raise ValueError('Teacher checkpoint changed since this stage started')
        teacher_manifest.write_text(report['teacher_checkpoint_sha256'], encoding='utf-8')
    parameters = sum(parameter.numel() for parameter in model.parameters())
    report['trainable_parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n.say(f"Trainable parameters={report['trainable_parameters']:,}; PEFT={c.model.peft.method}.")
    n.say(f"Parameters={parameters:,}; layers={c.model.layers}; width={c.model.hidden_size}; "
          f"attention heads={c.model.attention_heads}; KV heads={c.model.kv_heads}.")
    t = c.training
    report['execution'] = configure_execution(model, t, device)
    groups = [{"params": [p for p in model.parameters() if p.requires_grad and p.ndim >= 2],
               "weight_decay": t.weight_decay},
              {"params": [p for p in model.parameters() if p.requires_grad and p.ndim < 2], "weight_decay": 0}]
    optimizer_options = ({'fused': True} if t.optimizer_kernel == 'fused' else
                         {'foreach': True} if t.optimizer_kernel == 'foreach' else {})
    optimizer = torch.optim.AdamW(groups, lr=t.learning_rate, betas=t.betas,
                                  eps=t.optimizer_eps, **optimizer_options)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["requires_grad_scaler"])
    generator = torch.Generator().manual_seed(t.seed)
    step, consumed, skipped_updates = 0, 0, 0
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if resume:
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, consumed = saved["step"], saved["tokens_seen"]
        skipped_updates = saved.get('skipped_optimizer_updates', 0)
        n.say(f"Restored model, optimizer, scaler and random states at optimizer step {step}.")

    def save_checkpoint(index):
        path = checkpoints / f"step-{index:06d}.pt"
        temp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": index, "tokens_seen": consumed,
                    "skipped_optimizer_updates": skipped_updates,
                    "config_sha256": fingerprint, "data_sha256": data_hash,
                    "tokenizer_sha256": tokenizer_hash, "tokenizer_roles_sha256": roles_hash,
                    "tokens_file_sha256": token_hash,
                    "sampler_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, temp)
        temp.replace(path)
        report['checkpoint'] = str(path)
        n.say(f"Checkpoint saved at step {index}: {path.name}", detail=True)
        return path

    if prepare_only:
        report.update(status='prepared', checkpoint=str(Path(resume).resolve() if resume else save_checkpoint(step)),
                      parameters=parameters, stage=c.stage.kind, token_counts=token_counts)
        (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        return report

    explanations = {
        'pretrain': 'Next-token cross entropy from random initialization.',
        'midtrain': 'Continue next-token training with parent weights and vocabulary.',
        'sft': 'Response-only cross entropy: prompt and padding tokens are masked from loss.',
        'dpo': 'Preference loss against a frozen parent reference, using summed response log likelihoods.',
        'rlvr': 'Sample groups of responses, verify answers, normalize rewards and optimize a '
                'clipped policy objective with reference KL. Equal rewards give zero policy advantage.',
        'distill': 'Mix response cross entropy with temperature-scaled KL from a frozen teacher.'}
    n.stage(f"7. Train stage: {c.stage.kind}", explanations[c.stage.kind] +
            ' Use AdamW, gradient accumulation/clipping and the configured schedule. '
            'A short test validates execution; it does not establish model quality.')
    evaluation_blocks = {name: parts[:2] for name, parts in blocks.items()}
    baseline = evaluate(model, evaluation_blocks["validation"], c.evaluation.batches,
                        t.micro_batch_size, device, amp)
    n.say(f"Initial validation loss={baseline['loss']:.4f}; perplexity={baseline['perplexity']:.2f}.")
    started = time.monotonic()
    initial_step = step
    while step < t.steps:
        if time.monotonic() - started > t.max_runtime_seconds:
            checkpoint_path = save_checkpoint(step)
            raise TimeoutError(f"Training runtime cap reached; resume from {checkpoint_path}")
        factor = min(1.0, (step + 1) / max(1, t.warmup_steps))
        if step >= t.warmup_steps and t.schedule == "cosine":
            progress = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps - 1)
            factor = t.min_learning_rate_ratio + (1 - t.min_learning_rate_ratio) * (
                1 + math.cos(math.pi * progress)) / 2
        lr = t.learning_rate * factor
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        train_loss, step_tokens, auxiliary = 0, 0, {}
        for _ in range(t.gradient_accumulation):
            indices = torch.randint(len(blocks["train"][0]), (t.micro_batch_size,), generator=generator)
            parts = tuple(part[indices].to(device) for part in blocks["train"])
            with amp():
                loss, objective_metrics = objective(model, parts, c, reference, tokenizer)
                if getattr(model, 'auxiliary_loss', None) is not None:
                    loss = loss + model.auxiliary_loss
            for key, value in objective_metrics.items():
                auxiliary[key] = auxiliary.get(key, 0) + value / (
                    t.gradient_accumulation if key in {'reward_mean', 'teacher_kl'} else 1)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / t.gradient_accumulation).backward()
            train_loss += loss.item() / t.gradient_accumulation
            step_tokens += sum((part != -100).sum().item() for part in parts[1::2])
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.gradient_clip_norm,
                                              error_if_nonfinite=False)
        if not torch.isfinite(norm):
            if not scaler.is_enabled():
                raise FloatingPointError('Non-finite training gradient norm')
            # FP16 overflow is expected while GradScaler discovers a safe scale.
            # Skip this update rather than applying corrupted clipped gradients.
            scaler.update(new_scale=scaler.get_scale() * scaler.get_backoff_factor())
            skipped_updates += 1
            n.say(f'FP16 gradient overflow: skipped update and reduced loss scale to '
                  f'{scaler.get_scale():g}; retrying optimizer step {step + 1}.')
            continue
        scaler.step(optimizer)
        scaler.update()
        step += 1
        consumed += step_tokens
        metrics = {"step": step, "training_loss": train_loss, "learning_rate": lr,
                   "gradient_norm": norm.item(), "tokens_seen": consumed,
                   "skipped_optimizer_updates": skipped_updates,
                   "tokens_per_second": (step - initial_step) * t.micro_batch_size *
                   t.gradient_accumulation * c.model.context_length / (time.monotonic() - started)}
        metrics.update(auxiliary)
        if step % t.evaluate_every == 0 or step == t.steps:
            metrics["validation"] = evaluate(model, evaluation_blocks["validation"], c.evaluation.batches,
                                              t.micro_batch_size, device, amp)
        with (folder / "metrics.jsonl").open("a", encoding="utf-8") as output:
            output.write(json.dumps(metrics) + "\n")
        if step % t.log_every == 0 or step == 1 or step == t.steps:
            n.say(f"Step {step}/{t.steps}: loss={train_loss:.4f}, lr={lr:.2g}, "
                  f"tokens/s={metrics['tokens_per_second']:.0f}; "
                  f"validation={metrics.get('validation', 'not scheduled this step')}.")
        if step % t.checkpoint_every == 0 or step == t.steps:
            save_checkpoint(step)
    checkpoint_path = save_checkpoint(step)

    n.stage("8. Final held-out evaluation and checkpoint reload", "Evaluate validation and test splits "
            "without gradient updates. Reload the final weights into a fresh decoder and check matching "
            "logits. Perplexity only applies to this sample/tokenizer; it is not a standard-suite score.")
    validation = evaluate(model, evaluation_blocks["validation"], c.evaluation.batches,
                          t.micro_batch_size, device, amp)
    test = evaluate(model, evaluation_blocks["test"], c.evaluation.batches, t.micro_batch_size, device, amp)
    report['stage_evaluation'] = {
        'validation': evaluate_stage(model, reference, blocks['validation'], tokenizer, c, device, amp),
        'test': evaluate_stage(model, reference, blocks['test'], tokenizer, c, device, amp)}
    model.eval()
    restored = create_model(c.model, tokenizer.get_vocab_size()).to(device)
    restored.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["model"])
    restored.eval()
    probe = blocks["test"][0][:1].to(device)
    with torch.no_grad():
        if not torch.allclose(model(probe), restored(probe), rtol=1e-5, atol=1e-6):
            raise RuntimeError("Reloaded checkpoint logits differ")
    del restored
    n.say(f"Validation loss={validation['loss']:.4f}, test loss={test['loss']:.4f}; "
          "checkpoint reload verified.")
    benchmark_scores = (evaluate_choices(model, tokenizer, examples, device,
                                        c.model.context_length, amp) if examples else None)
    if benchmark_scores:
        n.say(f"Domain benchmark {benchmark_source['name']}: "
              f"accuracy={benchmark_scores['accuracy']:.3f} on {len(examples)} examples; "
              "a protocol check, not a reliable quality estimate.")

    n.stage("9. Generate a sample and save the run report", "Sample from the trained checkpoint using "
            "configured temperature/top-k. Expect mostly incoherent text after a tiny training test. "
            "Save configuration, manifests, tokenizer, metrics, checkpoints and this explanation log.")
    ids = tokenizer.encode(c.evaluation.prompt).ids
    if not ids:
        ids = [tokenizer.token_to_id("<bos>")]
    generated = torch.tensor([ids], device=device)
    with torch.no_grad():
        for _ in range(c.evaluation.generate_tokens):
            with amp():
                scores = model(generated[:, -c.model.context_length:])[:, -1].float()
            scores /= c.evaluation.temperature
            if c.evaluation.top_k:
                boundary = scores.topk(min(c.evaluation.top_k, scores.shape[-1])).values[:, -1:]
                scores = scores.masked_fill(scores < boundary, -float("inf"))
            token = torch.multinomial(scores.softmax(-1), 1)
            generated = torch.cat((generated, token), -1)
            if token.item() == tokenizer.token_to_id("<eos>"):
                break
    sample = tokenizer.decode(generated[0].tolist())
    report.update(status="completed", stage=c.stage.kind, parameters=parameters, optimizer_steps=step,
                  skipped_optimizer_updates=skipped_updates,
                  tokens_seen=consumed, token_counts=token_counts,
                  initial_validation=baseline, validation=validation, test=test,
                  checkpoint_recovery="passed", checkpoint=str(checkpoint_path),
                  elapsed_training_seconds=time.monotonic() - started,
                  source=source, processing=processing, tokenizer_sha256=tokenizer_hash,
                  data_sha256=data_hash, tokens_file_sha256=token_hash, generation=sample)
    report.update(benchmark_source=benchmark_source, benchmark_scores=benchmark_scores)
    (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    n.say(f"Generated sample: {sample}")
    n.say(f"Pipeline completed. Report: {folder / 'report.json'}")
    return report
