"""Generate on CPU from an explicitly selected native training checkpoint."""
import argparse
from pathlib import Path

import torch

from modelchef.engines import create_model, load_tokenizer
from modelchef.run_config import RunConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--prompt')
    parser.add_argument('--max-tokens', type=int, default=100)
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error('--max-tokens must be positive')
    folder = args.run_dir.resolve()
    config = RunConfig.model_validate_json((folder / 'config.json').read_text())
    if config.model.backend != 'native':
        parser.error('Only native model checkpoints are supported')
    checkpoints = sorted((folder / 'checkpoints').glob('step-*.pt'))
    checkpoint = args.checkpoint or (checkpoints[-1] if checkpoints else None)
    if checkpoint is None:
        parser.error('No saved checkpoint found')
    torch.set_num_threads(2)
    tokenizer = load_tokenizer(folder)
    model = create_model(config.model, tokenizer.get_vocab_size()).cpu().eval()
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)['model'])
    print(f'Loaded {checkpoint.name} on CPU. This model completes text; /exit quits.', flush=True)
    while True:
        try:
            prompt = args.prompt if args.prompt is not None else input('Story beginning: ')
        except (EOFError, KeyboardInterrupt):
            break
        if prompt.strip() == '/exit':
            break
        torch.manual_seed(config.training.seed)
        ids = tokenizer.encode(prompt).ids or [tokenizer.token_to_id('<bos>')]
        if config.tokenizer.add_bos and ids[0] != tokenizer.token_to_id('<bos>'):
            ids.insert(0, tokenizer.token_to_id('<bos>'))
        sequence = torch.tensor([ids])
        reply = []
        with torch.inference_mode():
            for _ in range(args.max_tokens):
                scores = model(sequence[:, -config.model.context_length:])[:, -1].float() / 0.8
                cutoff = scores.topk(min(40, scores.shape[-1])).values[:, -1:]
                token = torch.multinomial(scores.masked_fill(scores < cutoff, -float('inf')).softmax(-1), 1)
                if token.item() == tokenizer.token_to_id('<eos>'):
                    break
                reply.append(token.item())
                sequence = torch.cat((sequence, token), dim=-1)
        print(prompt + tokenizer.decode(reply), flush=True)
        if args.prompt is not None:
            break


if __name__ == '__main__':
    main()
