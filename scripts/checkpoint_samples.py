"""Generate identical, greedy text probes at saved native checkpoints; no downloads."""
import argparse
import json
from pathlib import Path

import torch
import yaml

from modelchef.engines import create_model, load_tokenizer
from modelchef.run_config import RunConfig


def samples(folder, prompts_path, max_tokens=48):
    folder = Path(folder).resolve()
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    if config.model.backend != 'native' or config.model.quantization != 'none':
        raise ValueError('This CPU probe supports unquantized native checkpoints')
    if max_tokens < 1:
        raise ValueError('max_tokens must be positive')
    config.model.attention_kernel = 'auto'
    config.model.custom_swiglu_path = None
    prompts = yaml.safe_load(Path(prompts_path).read_text(encoding='utf-8'))['prompts']
    tokenizer = load_tokenizer(folder)
    model = create_model(config.model, tokenizer.get_vocab_size()).eval()
    results = []
    for checkpoint in sorted((folder / 'checkpoints').glob('step-*.pt')):
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(state['model'])
        outputs = []
        for item in prompts:
            system = item.get('system', '')
            text = (('System: ' + system + '\n') if system else '') + 'User: ' + item['prompt'] + '\nAssistant:'
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if len(ids) + max_tokens > config.model.context_length:
                raise ValueError('Probe prompt/output exceeds context; shorten it instead of dropping the system prompt')
            sequence = torch.tensor([ids], dtype=torch.long)
            reply = []
            with torch.no_grad():
                for _ in range(max_tokens):
                    token = model(sequence)[:, -1].argmax(-1, keepdim=True)
                    if token.item() == tokenizer.token_to_id('<eos>'):
                        break
                    reply.append(token.item())
                    sequence = torch.cat((sequence, token), dim=1)
            outputs.append({**item, 'completion': tokenizer.decode(reply)})
        results.append({'step': state['step'], 'outputs': outputs})
    if not results:
        raise ValueError('No saved checkpoints found')
    return {'run_dir': str(folder), 'decoding': 'greedy', 'max_tokens': max_tokens,
            'scope': 'Qualitative probes, not scored benchmarks.', 'checkpoints': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('prompts', type=Path)
    parser.add_argument('--max-tokens', type=int, default=48)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = samples(args.run_dir, args.prompts, args.max_tokens)
    content = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding='utf-8')
    print(content)
