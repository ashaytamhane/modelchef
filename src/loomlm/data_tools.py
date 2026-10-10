"""Bounded format conversion and sequence distillation without external services."""
import csv
import io
import json
from pathlib import Path

from loomlm.corpus import digest


def convert_data(source, output, format='jsonl', max_bytes=10_000_000, max_documents=1000):
    if max_bytes < 1 or max_documents < 1:
        raise ValueError('Data limits must be positive')
    with Path(source).open('rb') as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError('Input exceeds conversion byte cap')
    text = payload.decode('utf-8')
    if format == 'csv':
        rows = list(csv.DictReader(io.StringIO(text)))[:max_documents]
    elif format == 'text':
        rows = [{'text': paragraph.strip()} for paragraph in text.split('\n\n')
                if paragraph.strip()][:max_documents]
    elif format == 'jsonl':
        rows = [json.loads(line) for line in text.splitlines() if line.strip()][:max_documents]
    else:
        raise ValueError('Supported input formats: text, csv, jsonl')
    if not rows or any(not isinstance(row, dict) for row in rows):
        raise ValueError('Input must produce nonempty object records')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        handle.write(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows))
    return {'documents': len(rows), 'source_sha256': digest(payload),
            'output_sha256': digest(output.read_bytes()), 'output': str(output)}


def teacher_responses(teacher, prompts, output, max_documents=32, max_bytes=1_000_000,
                      max_tokens=32, temperature=0.8):
    import torch

    from loomlm.chat import checkpoint_for
    from loomlm.engines import create_model, load_tokenizer
    from loomlm.hardware import detect_hardware, resolve_runtime
    from loomlm.objectives import format_prompt
    from loomlm.run_config import RunConfig

    if min(max_documents, max_bytes, max_tokens) < 1 or temperature <= 0:
        raise ValueError('Teacher generation limits/temperature must be positive')
    folder = Path(teacher).resolve()
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    tokenizer = load_tokenizer(folder)
    device = resolve_runtime({'precision': 'fp32'}, detect_hardware())['device']
    model = create_model(config.model, tokenizer.get_vocab_size()).to(device)
    checkpoint = checkpoint_for(folder)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True)['model'])
    model.eval()
    torch.manual_seed(config.training.seed)
    with Path(prompts).open('rb') as handle:
        payload = handle.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError('Teacher prompt data exceeds byte cap')
    records = [json.loads(line) for line in payload.splitlines() if line.strip()][:max_documents]
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle, torch.no_grad():
        for index, row in enumerate(records):
            prompt = row['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('Teacher prompts must be nonempty strings')
            ids = tokenizer.encode(format_prompt(prompt)).ids
            if len(ids) >= config.model.context_length:
                raise ValueError('Teacher prompt exceeds context; shorten prompts')
            answer = []
            for _ in range(min(max_tokens, config.model.context_length + 1 - len(ids))):
                logits = model(torch.tensor([ids], device=device))[:, -1].float() / temperature
                token = torch.multinomial(logits.softmax(-1), 1).item()
                if token == tokenizer.token_to_id('<eos>'):
                    break
                ids.append(token)
                answer.append(token)
            response = tokenizer.decode(answer).strip()
            if not response:
                raise ValueError('Teacher emitted an empty response; inspect teacher/prompt')
            handle.write(json.dumps({'prompt': prompt, 'response': response}) + '\n')
            print(f'Teacher response {index + 1}/{len(records)} generated locally.', flush=True)
    manifest = {'method': 'sequence_distillation', 'teacher_checkpoint_sha256': digest(checkpoint.read_bytes()),
                'prompts_sha256': digest(payload), 'output_sha256': digest(output.read_bytes()),
                'max_tokens': max_tokens, 'temperature': temperature, 'seed': config.training.seed}
    output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
