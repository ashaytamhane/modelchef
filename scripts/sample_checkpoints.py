"""Compare fixed-prompt continuations from saved native training checkpoints."""
import argparse
import json
import time
from pathlib import Path

import torch

from modelchef.engines import create_model, load_tokenizer
from modelchef.run_config import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--every', type=int, default=1000)
    args = parser.parse_args()
    if args.every <= 0:
        parser.error('--every must be positive')
    folder = args.run_dir.resolve()
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    if config.model.backend != 'native':
        parser.error('This checkpoint comparison supports native models only')
    torch.set_num_threads(2)  # Avoid competing with the training process for CPU cores.
    tokenizer = load_tokenizer(folder)
    model = create_model(config.model, tokenizer.get_vocab_size()).cpu().eval()
    output = folder / 'checkpoint-samples.jsonl'
    seen = {json.loads(line)['step'] for line in output.read_text().splitlines()} if output.exists() else set()
    prompts = ['Once upon a time, a little fox found', 'Lily lost her favorite toy. She decided to']
    while True:
        for checkpoint in sorted((folder / 'checkpoints').glob('step-*.pt')):
            step = int(checkpoint.stem.split('-')[1])
            if step % args.every or step in seen:
                continue
            print(f'Generating CPU samples from checkpoint step {step}.', flush=True)
            saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
            model.load_state_dict(saved['model'])
            del saved
            samples = []
            for prompt in prompts:
                torch.manual_seed(config.training.seed)
                sequence = torch.tensor([tokenizer.encode(prompt).ids])
                reply = []
                with torch.inference_mode():
                    for _ in range(80):
                        scores = model(sequence[:, -config.model.context_length:])[:, -1].float() / 0.8
                        cutoff = scores.topk(min(40, scores.shape[-1])).values[:, -1:]
                        token = torch.multinomial(scores.masked_fill(scores < cutoff, -float('inf')).softmax(-1), 1)
                        if token.item() == tokenizer.token_to_id('<eos>'):
                            break
                        reply.append(token.item())
                        sequence = torch.cat((sequence, token), dim=-1)
                samples.append({'prompt': prompt, 'continuation': tokenizer.decode(reply)})
            record = {'step': step, 'seed': config.training.seed, 'temperature': 0.8,
                      'top_k': 40, 'max_tokens': 80, 'device': 'cpu', 'samples': samples}
            with output.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record) + '\n')
            print(json.dumps(record, indent=2), flush=True)
            seen.add(step)
        if not args.watch or config.training.steps in seen or (folder / 'report.json').exists():
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
