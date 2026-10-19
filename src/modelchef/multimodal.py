"""Single-device vision-language SFT and Whisper ASR with local-only model loading."""
import hashlib
import json
import math
import os
import random
import time
import uuid
from contextlib import nullcontext
from pathlib import Path

from modelchef.corpus import file_digest
from modelchef.hardware import detect_hardware, resolve_runtime
from modelchef.multimodal_data import inspect_media, read_audio, read_manifest, split_records
from modelchef.progress import report
from modelchef.training_metrics import close_metrics, record_metrics, tensorboard_info


def model_manifest(path, cap):
    path = Path(path).resolve()
    files = sorted(model_file for model_file in path.iterdir() if model_file.is_file() and
                   model_file.suffix in {'.json', '.safetensors', '.txt', '.model', '.jinja'})
    if not (path / 'config.json').is_file() or not any(model_file.suffix == '.safetensors' for model_file in files):
        raise ValueError('Multimodal model requires a local config and safetensors weights')
    if sum(model_file.stat().st_size for model_file in files) > cap:
        raise ValueError('Local model exceeds max_model_bytes')
    names = {model_file.name for model_file in files}
    for index in path.glob('*.index.json'):
        for name in json.loads(index.read_text())['weight_map'].values():
            if name not in names:
                raise ValueError('Model index references a missing or unsafe weight shard')
    return {model_file.name: file_digest(model_file) for model_file in files}


def plan(config, root):
    root = Path(root)
    records, data = read_manifest(config, root)
    models = model_manifest(root / config.model.path, config.model.max_model_bytes)
    return {'modality': config.modality, 'data': data, 'model_files': models,
            'splits': {key: len(value) for key, value in split_records(records, config).items()},
            'steps': config.training.steps, 'downloads': False, 'aws_launch': False,
            'note': 'Local pretrained vision-language SFT or Whisper speech recognition; --run executes.'}


def load_components(config, root):
    from transformers import AutoModelForImageTextToText, AutoModelForSpeechSeq2Seq, AutoProcessor
    path = str((Path(root) / config.model.path).resolve())
    processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    cls = AutoModelForImageTextToText if config.modality == 'vision' else AutoModelForSpeechSeq2Seq
    model = cls.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                               use_safetensors=True, attn_implementation='sdpa')
    if config.modality == 'speech':
        if model.config.model_type != 'whisper':
            raise ValueError('Speech training currently supports Whisper encoder-decoder models')
        if processor.feature_extractor.sampling_rate != 16000:
            raise ValueError('Speech processor must use 16000 Hz')
        processor.tokenizer.set_prefix_tokens(language=config.model.language,
                                              task=config.model.speech_task)
    model.config.use_cache = False
    generation_options(model, config)
    return model, processor


def encode(row, processor, config, training=True):
    import torch
    field = 'image' if config.modality == 'vision' else 'audio'
    if row.get('media_sha256') and file_digest(Path(row[field])) != row['media_sha256']:
        raise ValueError('Selected media changed during the run; restore the recorded asset')
    if config.modality == 'speech':
        values = dict(processor(audio=read_audio(row['audio']), sampling_rate=16000,
                                return_tensors='pt'))
        if training:
            ids = processor.tokenizer(row['text'], return_tensors='pt')['input_ids']
            if ids.shape[1] > config.max_text_tokens:
                raise ValueError('Transcript exceeds max_text_tokens; shorten the sample')
            # Whisper prepends decoder_start_token_id itself when shifting labels.
            if ids[0, 0].item() == processor.tokenizer.convert_tokens_to_ids('<|startoftranscript|>'):
                ids = ids[:, 1:]
            values['labels'] = ids
        return values
    from PIL import Image
    with Image.open(row['image']) as source:
        image = source.convert('RGB')
    if config.model.vision_format == 'suffix':
        values = dict(processor(images=image, text=row['prompt'], return_tensors='pt',
                                **({'suffix': row['response']} if training else {})))
        if training and 'labels' not in values:
            raise ValueError('Suffix training requires a processor that produces assistant-only labels')
    else:
        messages = [{'role': 'user', 'content': [{'type': 'image'},
                    {'type': 'text', 'text': row['prompt']}]}]
        prefix = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompt = dict(processor(images=image, text=prefix, return_tensors='pt', add_special_tokens=False))
        if not training:
            values = prompt
        else:
            messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': row['response']}]})
            full = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            values = dict(processor(images=image, text=full, return_tensors='pt', add_special_tokens=False))
            length = prompt['input_ids'].shape[1]
            if not torch.equal(values['input_ids'][:, :length], prompt['input_ids']):
                raise ValueError('Chat template does not preserve generation prefix; choose a compatible template')
            labels = values['input_ids'].clone()
            labels[:, :length] = -100
            if 'attention_mask' in values:
                labels[values['attention_mask'] == 0] = -100
            values['labels'] = labels
    if values['input_ids'].shape[1] > config.max_text_tokens:
        raise ValueError('Text plus image tokens exceed max_text_tokens; adjust the recipe or image processor')
    if training and not (values['labels'] != -100).any():
        raise ValueError('Training sample has no supervised target tokens')
    return values


def edit_distance(reference, prediction):
    previous = list(range(len(prediction) + 1))
    for i, token in enumerate(reference, 1):
        current = [i]
        for j, other in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (token != other)))
        previous = current
    return previous[-1]


def generation_options(model, config):
    options = {'max_new_tokens': config.evaluation.max_new_tokens, 'do_sample': False}
    if config.modality == 'speech':
        if getattr(model.generation_config, 'is_multilingual', False):
            options['task'] = config.model.speech_task
            if config.model.language:
                options['language'] = config.model.language
        elif config.model.language or config.model.speech_task != 'transcribe':
            raise ValueError('English-only Whisper models support transcription without language/task overrides')
    return options


def evaluate(model, processor, rows, config, device, amp):
    import torch

    from modelchef.execution import attention_context
    losses, predictions, errors, words = [], [], 0, 0
    model.eval()
    with torch.no_grad():
        for row in rows[:config.evaluation.max_examples]:
            batch = {key: value.to(device) for key, value in encode(row, processor, config).items()}
            with amp(), attention_context(config.model.attention_kernel):
                loss = model(**batch).loss
            losses.append(loss.item())
            prompt = {key: value.to(device) for key, value in encode(row, processor, config, False).items()}
            options = generation_options(model, config)
            with amp(), attention_context(config.model.attention_kernel):
                ids = model.generate(**prompt, **options)
            if config.modality == 'vision':
                ids = ids[:, prompt['input_ids'].shape[1]:]
            prediction = processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
            target = row['response'] if config.modality == 'vision' else row['text']
            predictions.append({'prediction': prediction, 'target': target})
            errors += edit_distance(target.lower().split(), prediction.lower().split())
            words += len(target.split())
    if not all(math.isfinite(loss) for loss in losses):
        raise RuntimeError('Non-finite held-out multimodal loss')
    return {'loss': sum(losses) / len(losses), 'examples': len(losses),
            'word_error_rate': errors / max(1, words) if config.modality == 'speech' else None,
            'exact_match': sum(p['prediction'] == p['target'] for p in predictions) / len(predictions),
            'predictions': predictions, 'scope': 'Held-out local sample; not a standard benchmark suite.'}


def train(config, root, resume=None):
    import torch

    from modelchef.execution import attention_context
    from modelchef.provenance import environment_provenance
    root = Path(root).resolve()
    report('Multimodal 1/5: validate bounded media and local model checksums; no downloads.')
    records, data = read_manifest(config, root)
    source = model_manifest(root / config.model.path, config.model.max_model_bytes)
    splits = split_records(records, config)
    fingerprint = hashlib.sha256(config.model_dump_json().encode()).hexdigest()
    saved = torch.load(resume, map_location='cpu', weights_only=True) if resume else None
    if saved and (saved['config_sha256'], saved['data'], saved['source']) != (fingerprint, data, source):
        raise ValueError('Resume recipe, media or model files changed')
    folder = Path(resume).resolve().parents[1] if resume else (
        root / config.output_dir / (config.name + '-' + uuid.uuid4().hex[:12]))
    folder.mkdir(parents=True, exist_ok=True)
    result = {'status': 'running', 'modality': config.modality, 'run_dir': str(folder)}
    try:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
        runtime = resolve_runtime(config.training.model_dump(), detect_hardware())
        device = runtime['device']
        if config.model.attention_kernel not in {'auto', 'math'} and device != 'cuda':
            raise ValueError('Optimized attention requires compatible CUDA hardware')
        if config.training.optimizer_kernel == 'fused' and device != 'cuda':
            raise ValueError('Fused optimizer requires CUDA')
        dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[runtime['precision']]
        def amp():
            return nullcontext() if dtype == torch.float32 else torch.autocast(device, dtype=dtype)
        torch.manual_seed(config.training.seed)
        random.seed(config.training.seed)
        torch.use_deterministic_algorithms(config.training.deterministic)
        report('Multimodal 2/5: load model/processor; apply freezing and execution policy.')
        model, processor = load_components(config, root)
        model.to(device)
        if config.model.freeze_encoders:
            for name in config.model.encoder_modules:
                try:
                    encoder = model.get_submodule(name)
                except AttributeError as error:
                    raise ValueError(f'Encoder module not found: {name}; inspect model.named_modules()') from error
                encoder.requires_grad_(False)
        if config.model.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        if config.training.compile:
            options = {'backend': config.training.compile_backend}
            if config.training.compile_backend == 'inductor':
                options['mode'] = config.training.compile_mode
            model.forward = torch.compile(model.forward, **options)
        torch.backends.cuda.matmul.allow_tf32 = config.training.allow_tf32
        parameters = [p for p in model.parameters() if p.requires_grad]
        if not parameters:
            raise ValueError('Freezing policy leaves no trainable parameters')
        optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate,
            betas=config.training.betas, weight_decay=config.training.weight_decay,
            eps=config.training.optimizer_eps,
            **({'fused': True} if config.training.optimizer_kernel == 'fused' else
               {'foreach': True} if config.training.optimizer_kernel == 'foreach' else {}))
        scaler = torch.amp.GradScaler('cuda', enabled=device == 'cuda' and dtype == torch.float16)
        sampler = torch.Generator().manual_seed(config.training.seed)
        step = 0
        if saved:
            model.load_state_dict(saved['model'])
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            sampler.set_state(saved['sampler'])
            torch.set_rng_state(saved['torch_rng'])
            if device == 'cuda' and saved['cuda_rng']:
                torch.cuda.set_rng_state_all(saved['cuda_rng'])
            step = saved['step']
        else:
            (folder / 'config.json').write_text(config.model_dump_json(indent=2), encoding='utf-8')
            (folder / 'data-manifest.json').write_text(json.dumps(data, indent=2))
            (folder / 'splits.json').write_text(json.dumps(splits, indent=2))
            (folder / 'environment.json').write_text(json.dumps({**runtime, **environment_provenance(root)}, indent=2))
        def checkpoint():
            target = folder / 'checkpoints' / f'step-{step:06d}.pt'
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix('.tmp')
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                'scaler': scaler.state_dict(), 'sampler': sampler.get_state(),
                'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if device == 'cuda' else [],
                'step': step, 'config_sha256': fingerprint, 'data': data, 'source': source}, temporary)
            temporary.replace(target)
            return target
        report('Multimodal 3/5: train supervised targets; vision prompts/image tokens are masked.')
        started = time.monotonic()
        while step < config.training.steps:
            if time.monotonic() - started > config.training.max_runtime_seconds:
                raise TimeoutError(f'Multimodal runtime cap reached; resume from {checkpoint()}')
            model.train()
            optimizer.zero_grad(set_to_none=True)
            warmup = config.training.warmup_steps
            factor = min(1, (step + 1) / max(1, warmup))
            if step >= warmup and config.training.schedule == 'cosine':
                progress = (step - warmup) / max(1, config.training.steps - warmup - 1)
                floor = config.training.min_learning_rate_ratio
                factor = floor + (1 - floor) * (1 + math.cos(math.pi * progress)) / 2
            for group in optimizer.param_groups:
                group['lr'] = config.training.learning_rate * factor
            loss_total = 0
            for _ in range(config.training.gradient_accumulation):
                index = torch.randint(len(splits['train']), (1,), generator=sampler).item()
                batch = {key: value.to(device) for key, value in encode(splits['train'][index], processor, config).items()}
                with amp(), attention_context(config.model.attention_kernel):
                    loss = model(**batch).loss
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite multimodal training loss')
                loss_total += loss.item() / config.training.gradient_accumulation
                scaler.scale(loss / config.training.gradient_accumulation).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.training.gradient_clip_norm)
            if not torch.isfinite(norm):
                if scaler.is_enabled():
                    scaler.update()
                    continue
                raise RuntimeError('Non-finite multimodal gradient norm')
            scaler.step(optimizer)
            scaler.update()
            step += 1
            row = {'step': step, 'loss': loss_total, 'learning_rate': optimizer.param_groups[0]['lr']}
            if step % config.training.evaluate_every == 0 or step == config.training.steps:
                row['validation'] = evaluate(model, processor, splits['validation'], config, device, amp)
            record_metrics(folder, row)
            if step == 1 or step % config.training.log_every == 0 or step == config.training.steps:
                report(f'Multimodal step {step}/{config.training.steps}: loss={loss_total:.4f}')
            if step % config.training.checkpoint_every == 0:
                checkpoint()
        path = checkpoint()
        report('Multimodal 4/5: evaluate disjoint validation/test assets and reload checkpoint.')
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True)['model'])
        result.update(validation=evaluate(model, processor, splits['validation'], config, device, amp),
                      test=evaluate(model, processor, splits['test'], config, device, amp))
        report('Multimodal 5/5: export model and processor for local inference.')
        export = folder / 'export'
        model.save_pretrained(export, safe_serialization=True)
        processor.save_pretrained(export)
        result.update(status='completed', optimizer_steps=step, checkpoint=str(path), export=str(export),
                      data=data, source=source, elapsed_seconds=time.monotonic() - started)
    except Exception as error:
        result.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        close_metrics(folder)
        result['tensorboard'] = tensorboard_info(folder)
        (folder / 'report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def predict(config, root, media, prompt='', model_path=None):
    import torch
    config = config.model_copy(deep=True)
    if model_path:
        config.model.path = str(model_path)
    model_manifest(Path(root) / config.model.path, config.model.max_model_bytes)
    media = Path(media).resolve()
    inspect_media(media, config)
    runtime = resolve_runtime(config.training.model_dump(), detect_hardware())
    model, processor = load_components(config, root)
    device = runtime['device']
    model.to(device).eval()
    row = {'image': str(media), 'prompt': prompt} if config.modality == 'vision' else {'audio': str(media)}
    inputs = {key: value.to(device) for key, value in encode(row, processor, config, False).items()}
    dtype = {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[runtime['precision']]
    amp = nullcontext() if dtype == torch.float32 else torch.autocast(device, dtype=dtype)
    from modelchef.execution import attention_context
    options = generation_options(model, config)
    with torch.no_grad(), amp, attention_context(config.model.attention_kernel):
        ids = model.generate(**inputs, **options)
    if config.modality == 'vision':
        ids = ids[:, inputs['input_ids'].shape[1]:]
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()
