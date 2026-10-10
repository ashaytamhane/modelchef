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

from ashayllm.benchmarks import acquire_benchmark, evaluate_choices, select_benchmark
from ashayllm.corpus import acquire, digest, process
from ashayllm.hardware import detect_hardware, resolve_runtime
from ashayllm.model import Decoder


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
        ids = tokenizer.encode(row["text"]).ids
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
            tokens += y.numel()
    model.train()
    nll = loss_sum / tokens
    return {"loss": nll, "perplexity": math.exp(min(nll, 80)), "tokens": tokens,
            "perplexity_capped": nll > 80}


def run_pipeline(config, root, resume=None):
    root = Path(root).resolve()
    recipe = config.model_dump(mode="json")
    fingerprint = digest(json.dumps(recipe, sort_keys=True).encode())
    folder = (Path(resume).resolve().parent.parent if resume else
              root / config.output_dir / (config.name + "-" + uuid.uuid4().hex[:12]))
    folder.mkdir(parents=True, exist_ok=True)
    if resume and json.loads((folder / "config.json").read_text()) != recipe:
        raise ValueError("Resume configuration differs; refusing to modify the saved run")
    narrate = Narrator(folder, config.explain == "detailed")
    report = {"status": "running", "run_dir": str(folder), "config_sha256": fingerprint}
    (folder / "config.json").write_text(json.dumps(recipe, indent=2), encoding="utf-8")
    try:
        return _execute(config, root, folder, fingerprint, narrate, report, resume)
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        (folder / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        narrate.say(f"Pipeline stopped: {error}. Evidence remains in {folder}.")
        raise


def _execute(c, root, folder, fingerprint, n, report, resume):
    n.stage("1. Configuration and hardware", "Validate the complete recipe before data access. "
            "Detect this machine's accelerator; no AWS instance is launched by this command.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(c.training.seed)
    torch.manual_seed(c.training.seed)
    torch.use_deterministic_algorithms(c.training.deterministic)
    torch.backends.cuda.matmul.allow_tf32 = False
    runtime = resolve_runtime(c.training.model_dump(), detect_hardware())
    device = runtime["device"]
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[
        runtime["precision"]]
    def amp():
        return (nullcontext() if dtype == torch.float32 else
                torch.autocast(device_type=device, dtype=dtype))
    (folder / "environment.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    n.say(f"Device={device}, precision={runtime['precision']}, seed={c.training.seed}.")

    n.stage("2. Acquire a bounded corpus snapshot", f"Source={c.data.dataset}, domain={c.data.domain}. "
            f"Limits: {c.data.max_documents} documents and {c.data.max_download_bytes:,} bytes. "
            "The saved SHA256 identifies the exact sample; raw text is never committed to Git.")
    records, source = acquire(c, root, lambda message: n.say(message, detail=True))
    (folder / "source-manifest.json").write_text(json.dumps(source, indent=2), encoding="utf-8")

    n.stage("2b. Select domain benchmarks", f"Domain '{c.data.domain}' selects "
            f"'{select_benchmark(c)}'. Download only the configured small evaluation sample. "
            "These examples are excluded from tokenizer/training inputs and used only for evaluation.")
    examples, benchmark_source = acquire_benchmark(c, root)
    (folder / "benchmark-manifest.json").write_text(
        json.dumps(benchmark_source, indent=2), encoding="utf-8")
    processing_config = c.model_copy(deep=True)
    if examples:
        benchmark_path = folder / "benchmark-text.txt"
        benchmark_path.write_text("\n".join(e["decontamination_text"] for e in examples),
                                  encoding="utf-8")
        processing_config.processing.benchmark_files.append(str(benchmark_path))

    n.stage("3. Clean, filter, deduplicate and split", "Normalize Unicode/HTML/whitespace, reject "
            "length/quality/repetition failures, apply domain keywords, regex PII handling and "
            "lexical deduplication before document-level seeded train/validation/test splitting. "
            "Configured benchmark files enable exact n-gram decontamination; otherwise it is unconfigured.")
    splits, processing = process(records, processing_config, root)
    for name, rows in splits.items():
        (folder / f"{name}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows),
                                           encoding="utf-8")
    (folder / "processing.json").write_text(json.dumps(processing, indent=2), encoding="utf-8")
    n.say(f"Filter counts: {processing['counts']}; document splits: {processing['split_documents']}.")
    n.say(f"Benchmark decontamination: {processing['decontamination']}. "
          "PII/dedup checks are limited heuristics, not complete guarantees.")

    n.stage("4. Train a tokenizer from scratch", "Byte-level BPE learns merges on training documents "
            "only. Validation/test text cannot influence the vocabulary. No pretrained tokenizer is loaded.")
    tokenizer = train_tokenizer(splits["train"], c.tokenizer)
    tokenizer.save(str(folder / "tokenizer.json"))
    tokenizer_hash = digest((folder / "tokenizer.json").read_bytes())
    n.say(f"BPE vocabulary={tokenizer.get_vocab_size()} (requested {c.tokenizer.vocab_size}).")

    n.stage("5. Tokenize and pack next-token examples", "Encode each split separately, insert configured "
            "BOS/EOS tokens, concatenate within splits, then form fixed-length input/shifted-target pairs. "
            "Tail tokens are dropped; cross-document attention is allowed. Enforce a total token cap.")
    blocks, token_counts = {}, {}
    fractions = {"validation": c.processing.validation_fraction, "test": c.processing.test_fraction,
                 "train": 1 - c.processing.validation_fraction - c.processing.test_fraction}
    for name, rows in splits.items():
        blocks[name], token_counts[name] = pack(rows, tokenizer, c.tokenizer,
            c.model.context_length, int(c.data.max_tokens * fractions[name]))
    torch.save(blocks, folder / "tokens.pt")
    token_hash = digest((folder / "tokens.pt").read_bytes())
    # Logical data identity is independent of torch ZIP container metadata.
    data_hash = hashlib.sha256(b"".join(part.numpy().tobytes() for name in sorted(blocks)
                                       for part in blocks[name])).hexdigest()
    n.say(f"Token counts: {token_counts}; context={c.model.context_length}. "
          "This is bounded pretraining data, not a full corpus download.")

    n.stage("6. Initialize a decoder-only transformer", "Random weights with causal scaled dot-product "
            "attention, rotary positions, grouped-query attention, RMSNorm and SwiGLU. "
            "Parameter count comes from actual tensors, not a target label.")
    model = Decoder(c.model, tokenizer.get_vocab_size()).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    n.say(f"Parameters={parameters:,}; layers={c.model.layers}; width={c.model.hidden_size}; "
          f"attention heads={c.model.attention_heads}; KV heads={c.model.kv_heads}.")
    t = c.training
    groups = [{"params": [p for p in model.parameters() if p.ndim >= 2],
               "weight_decay": t.weight_decay},
              {"params": [p for p in model.parameters() if p.ndim < 2], "weight_decay": 0}]
    optimizer = torch.optim.AdamW(groups, lr=t.learning_rate, betas=t.betas, eps=t.optimizer_eps)
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["requires_grad_scaler"])
    generator = torch.Generator().manual_seed(t.seed)
    step, consumed = 0, 0
    checkpoints = folder / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    if resume:
        saved = torch.load(resume, map_location="cpu", weights_only=True)
        if (saved["config_sha256"], saved["data_sha256"], saved["tokenizer_sha256"]) != (
                fingerprint, data_hash, tokenizer_hash):
            raise ValueError("Resume recipe/data/tokenizer differs from checkpoint")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        generator.set_state(saved["sampler_rng"])
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if device == "cuda":
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        step, consumed = saved["step"], saved["tokens_seen"]
        n.say(f"Restored model, optimizer, scaler and random states at optimizer step {step}.")

    def save_checkpoint(index):
        path = checkpoints / f"step-{index:06d}.pt"
        temp = path.with_suffix(".tmp")
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scaler": scaler.state_dict(), "step": index, "tokens_seen": consumed,
                    "config_sha256": fingerprint, "data_sha256": data_hash,
                    "tokenizer_sha256": tokenizer_hash, "sampler_rng": generator.get_state(),
                    "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if device == "cuda" else []}, temp)
        temp.replace(path)
        n.say(f"Checkpoint saved at step {index}: {path.name}", detail=True)
        return path

    n.stage("7. Pretrain and validate", "Measure held-out validation loss before updates, then train "
            "next-token cross entropy using AdamW, gradient accumulation/clipping and the configured "
            "learning-rate schedule. Log loss, throughput and validation metrics. A short run tests "
            "the pipeline; it does not produce a useful language model.")
    baseline = evaluate(model, blocks["validation"], c.evaluation.batches,
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
        train_loss = 0
        for _ in range(t.gradient_accumulation):
            indices = torch.randint(len(blocks["train"][0]), (t.micro_batch_size,), generator=generator)
            x, y = (part[indices].to(device) for part in blocks["train"])
            with amp():
                logits = model(x)
                loss = F.cross_entropy(logits.float().flatten(0, 1), y.flatten())
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            scaler.scale(loss / t.gradient_accumulation).backward()
            train_loss += loss.item() / t.gradient_accumulation
            consumed += y.numel()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t.gradient_clip_norm,
                                              error_if_nonfinite=True)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        metrics = {"step": step, "training_loss": train_loss, "learning_rate": lr,
                   "gradient_norm": norm.item(), "tokens_seen": consumed,
                   "tokens_per_second": (step - initial_step) * t.micro_batch_size *
                   t.gradient_accumulation * c.model.context_length / (time.monotonic() - started)}
        if step % t.evaluate_every == 0 or step == t.steps:
            metrics["validation"] = evaluate(model, blocks["validation"], c.evaluation.batches,
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
    validation = evaluate(model, blocks["validation"], c.evaluation.batches,
                          t.micro_batch_size, device, amp)
    test = evaluate(model, blocks["test"], c.evaluation.batches, t.micro_batch_size, device, amp)
    model.eval()
    restored = Decoder(c.model, tokenizer.get_vocab_size()).to(device)
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
    report.update(status="completed", parameters=parameters, optimizer_steps=step,
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
