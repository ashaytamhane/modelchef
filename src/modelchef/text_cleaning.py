"""Conservative, document-local sentence cleanup and atomic streaming JSONL rewrites."""
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

_END = re.compile(r'''[.!?]+["'”’»\])}]*(?:\s+|$)''')
_ABBREVIATION = re.compile(
    r'(?i)(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|fig|no|vol|pp)\.|'
    r'\b(?:[a-z]\.)+)["\'”’»\])}]*$')


def collapse_duplicate_sentences(text, min_words=6):
    """Keep the first of adjacent exact sentences, comparing whitespace only.

    Case and punctuation are significant. Sentence boundaries are conservative
    .!? heuristics with common English abbreviation guards, not a language parser.
    Short sentences, unterminated fragments and intervening content break runs.
    Memory is bounded by a single document; no corpus-wide duplicate index is used.
    """
    if min_words < 1:
        raise ValueError('min_words must be positive')
    chunks, previous, start, removed, chars = [], None, 0, 0, 0
    for match in _END.finditer(text):
        candidate = text[start:match.end()]
        stripped = candidate.strip()
        if _ABBREVIATION.search(stripped):
            continue
        key = ' '.join(stripped.split())
        eligible = len(re.findall(r'\b\w+\b', key)) >= min_words
        if eligible and key == previous:
            removed += 1
            chars += len(candidate)
        else:
            chunks.append(candidate)
        previous = key if eligible else None
        start = match.end()
    chunks.append(text[start:])
    return ''.join(chunks), removed, chars


def clean_jsonl(source, destination, *, min_words=6, text_field='text',
                max_record_bytes=4000000, overwrite=False):
    """Rewrite only an explicitly selected file; preserve row order and metadata.

    In-place cleanup requires overwrite=True and uses an atomic directory-entry
    replacement, never byte overwrites (including for hard-linked input files).
    On any parse/validation/write failure the previous destination is unchanged.
    No token files, tokenizer, split membership or neighboring files are changed.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if min_words < 1 or max_record_bytes < 1:
        raise ValueError('min_words and max_record_bytes must be positive')
    if destination.exists() and not overwrite:
        raise ValueError('Destination exists; explicit overwrite is required')
    counts = Counter(documents=0, documents_changed=0, sentences_removed=0, chars_removed=0)
    before, after = hashlib.sha256(), hashlib.sha256()
    fd, temporary_name = tempfile.mkstemp(prefix='.sentence-clean-', suffix='.tmp',
                                         dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, 'wb') as writer, source.open('rb') as reader:
            while line := reader.readline(max_record_bytes + 1):
                if len(line) > max_record_bytes:
                    raise ValueError('JSONL record exceeds max_record_bytes')
                before.update(line)
                if not line.strip():
                    writer.write(line)
                    after.update(line)
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get(text_field), str):
                    raise TypeError('Each JSONL row requires a string text field')
                counts['documents'] += 1
                cleaned, removed, chars = collapse_duplicate_sentences(row[text_field], min_words)
                payload = line
                if removed:
                    row[text_field] = cleaned
                    # ModelChef prepared rows use sha256 for the text content.
                    if text_field == 'text' and 'sha256' in row:
                        row['sha256'] = hashlib.sha256(cleaned.encode()).hexdigest()
                    payload = (json.dumps(row, ensure_ascii=False) + '\n').encode()
                    counts['documents_changed'] += 1
                    counts['sentences_removed'] += removed
                    counts['chars_removed'] += chars
                writer.write(payload)
                after.update(payload)
            writer.flush()
            os.fsync(writer.fileno())
        if overwrite:
            os.replace(temporary, destination)
        elif os.name == 'nt':
            # Windows rename refuses an existing destination atomically.
            os.rename(temporary, destination)
        else:
            # Exclusive publication: do not overwrite a destination created during processing.
            os.link(temporary, destination)
            temporary.unlink()
    finally:
        temporary.unlink(missing_ok=True)
    return {'source': str(source), 'destination': str(destination), 'counts': dict(counts),
            'source_sha256': before.hexdigest(), 'output_sha256': after.hexdigest(),
            'min_words': min_words, 'split_membership_changed': False,
            'token_files_updated': False}
