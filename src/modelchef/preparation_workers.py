"""Spawn-safe workers for independent text cleaning and tokenization."""
import os
from collections import Counter

from modelchef.corpus import clean_record
from modelchef.resources import ordered_work

_config = _benchmark = _tokenizer = None


def initialize_cleaning(config, benchmark):
    global _config, _benchmark
    _config, _benchmark = config, benchmark


def clean_batch(rows):
    result = []
    for row in rows:
        counts = Counter()
        result.append((clean_record(row, _config, _benchmark, counts), counts))
    return result


def cleaned_records(rows, config, benchmark, counts):
    for cleaned, delta in ordered_work(rows, clean_batch, initialize_cleaning,
                                       (config, benchmark)):
        counts.update(delta)
        yield cleaned


def initialize_tokenizer(serialized):
    global _tokenizer
    # One Rust thread per process avoids workers * native threads oversubscription.
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    from tokenizers import Tokenizer
    _tokenizer = Tokenizer.from_str(serialized)


def encode_batch(rows):
    return [item.ids for item in _tokenizer.encode_batch(
        [row['text'] for row in rows], add_special_tokens=False)]


def token_ids(rows, tokenizer):
    return ordered_work(rows, encode_batch, initialize_tokenizer, (tokenizer.to_str(),))
