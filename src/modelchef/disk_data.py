"""Bounded JSONL shards, SQLite deduplication and read-only mapped training tokens."""
import hashlib
import json
import sqlite3
import uuid
from collections import Counter
from pathlib import Path

from modelchef.corpus import clean_record, digest, file_digest, group_split, ngrams


class JsonlRows:
    def __init__(self, path, count):
        self.path, self.count = Path(path), count

    def __iter__(self):
        with self.path.open(encoding='utf-8') as source:
            for line in source:
                yield json.loads(line)

    def __len__(self):
        return self.count


def acquire_disk(config, root, explain):
    c = config.data
    paths = [(root / name).resolve() for name in (c.local_paths or [c.local_path])]
    identity = digest(json.dumps(c.model_dump(), sort_keys=True).encode())
    folder = root / 'data/cache' / identity
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / ('raw-' + uuid.uuid4().hex + '.tmp')
    downloaded = count = index = 0
    checksum = hashlib.sha256()
    try:
        with temporary.open('wb') as target:
            for path in paths:
                with path.open('rb') as source:
                    while line := source.readline(min(c.max_record_bytes,
                                                      c.max_download_bytes - downloaded) + 1):
                        if len(line) > c.max_record_bytes:
                            raise ValueError('JSONL record exceeds max_record_bytes')
                        downloaded += len(line)
                        if downloaded > c.max_download_bytes:
                            raise ValueError('Local shards exceed the configured byte budget')
                        if not line.strip():
                            continue
                        row = json.loads(line)
                        current, index = index, index + 1
                        if current < c.offset:
                            continue
                        record = {'text': row[c.text_field], 'source_index': current}
                        if 'split_group' in row:
                            if not isinstance(row['split_group'], str) or not row['split_group']:
                                raise ValueError('split_group must be a nonempty string')
                            record['split_group'] = row['split_group']
                        payload = (json.dumps(record, ensure_ascii=False) + '\n').encode()
                        target.write(payload)
                        checksum.update(payload)
                        count += 1
                        if count >= c.max_documents:
                            break
                if count >= c.max_documents:
                    break
        # Content-addressed snapshots do not collide between concurrent readers/runs.
        raw = folder / (checksum.hexdigest() + '.jsonl')
        temporary.replace(raw)
    finally:
        temporary.unlink(missing_ok=True)
    info = {'schema_version': 1, 'provider': 'local_jsonl', 'storage': 'disk',
            'paths': [str(p) for p in paths], 'documents': count,
            'response_bytes': downloaded, 'raw_sha256': checksum.hexdigest()}
    explain(f'Snapshotted {count} records from {len(paths)} local shards; bounded RAM processing.')
    return JsonlRows(raw, count), info


def _sketch(text, words):
    """64 deterministic minhash permutations in 16 bands; candidates get exact Jaccard."""
    import numpy as np
    shingles = ngrams(text, words)
    if not shingles:
        return shingles, []
    values = np.array([int.from_bytes(hashlib.blake2b(' '.join(s).encode(),
                       digest_size=8).digest(), 'little') for s in shingles], dtype=np.uint64)
    rng = np.random.default_rng(0)
    a = rng.integers(1, 2**63, size=64, dtype=np.uint64) | np.uint64(1)
    b = rng.integers(0, 2**63, size=64, dtype=np.uint64)
    # Unsigned overflow is intentional; bytes and permutations are platform-independent.
    signatures = (values[:, None] * a + b).min(axis=0).astype('<u8')
    return shingles, [signatures[i:i + 4].tobytes() for i in range(0, 64, 4)]


def process_disk(records, config, root):
    c = config.processing
    benchmark = set()
    for path in c.benchmark_files:
        benchmark |= ngrams((root / path).read_text(encoding='utf-8'),
                            c.decontamination_ngram_words)
    folder = root / 'data/cache/processed' / uuid.uuid4().hex
    folder.mkdir(parents=True)
    counts = Counter()
    grouped = False
    connection = sqlite3.connect(folder / 'dedup.sqlite')
    try:
        connection.executescript('''
            CREATE TABLE documents (id INTEGER PRIMARY KEY, fingerprint TEXT, text TEXT,
                                    source_index INTEGER, split TEXT, ordering TEXT, split_group TEXT);
            CREATE INDEX fingerprints ON documents(fingerprint);
            CREATE TABLE bands (band INTEGER, signature BLOB, document INTEGER);
            CREATE INDEX bucket ON bands(band, signature);
        ''')
        for row in records:
            cleaned = clean_record(row, config, benchmark, counts)
            if cleaned is None:
                continue
            text = cleaned['text']
            fingerprint = digest(text.casefold().encode())
            if c.exact_dedup and connection.execute(
                    'SELECT 1 FROM documents WHERE fingerprint=? LIMIT 1', (fingerprint,)).fetchone():
                counts['exact_duplicates'] += 1
                continue
            shingles, bands = _sketch(text, c.dedup_ngram_words) if c.near_dedup else (set(), [])
            duplicate = False
            if bands:
                clauses = ' OR '.join('(b.band=? AND b.signature=?)' for _ in bands)
                params = [value for i, signature in enumerate(bands) for value in (i, signature)]
                candidates = connection.execute(
                    'SELECT DISTINCT d.text FROM documents d JOIN bands b ON d.id=b.document WHERE ' +
                    clauses, params)
                for (previous,) in candidates:
                    other = ngrams(previous, c.dedup_ngram_words)
                    if len(shingles & other) / len(shingles | other) >= c.near_dedup_jaccard:
                        duplicate = True
                        break
                candidates.close()
            if duplicate:
                counts['near_duplicates'] += 1
                continue
            ordering = digest(f'{config.training.seed}:'.encode() + text.encode())
            fraction = int(ordering[:16], 16) / 2**64
            split = ('validation' if fraction < c.validation_fraction else
                     'test' if fraction < c.validation_fraction + c.test_fraction else 'train')
            grouping = cleaned.get('split_group')
            if grouping is not None:
                grouped = True
                split = group_split(grouping, config)
            identifier = connection.execute(
                'INSERT INTO documents(fingerprint,text,source_index,split,ordering,split_group) VALUES(?,?,?,?,?,?)',
                (fingerprint, text, cleaned['source_index'], split, ordering, grouping)).lastrowid
            connection.executemany('INSERT INTO bands VALUES(?,?,?)',
                                   [(i, signature, identifier) for i, signature in enumerate(bands)])
            counts['accepted'] += 1
            if counts['accepted'] % 1000 == 0:
                connection.commit()
        if counts['accepted'] < 3:
            raise ValueError(f'Too few documents after processing: {dict(counts)}')
        sizes = dict(connection.execute('SELECT split,COUNT(*) FROM documents GROUP BY split'))
        if grouped and any(not sizes.get(split) for split in ['train', 'validation', 'test']):
            raise ValueError('Grouped data has an empty split; acquire more independent groups')
        # Keep even tiny inputs usable while preserving deterministic, document-level splits.
        for split in ['train', 'validation', 'test']:
            if not sizes.get(split):
                donor = max(sizes, key=sizes.get)
                identifier = connection.execute(
                    'SELECT id FROM documents WHERE split=? ORDER BY ordering LIMIT 1', (donor,)
                ).fetchone()[0]
                connection.execute('UPDATE documents SET split=? WHERE id=?', (split, identifier))
                sizes[donor] -= 1
                sizes[split] = 1
        splits = {}
        for split in ['train', 'validation', 'test']:
            path = folder / (split + '.jsonl')
            with path.open('w', encoding='utf-8') as target:
                for text, index, grouping in connection.execute(
                        'SELECT text,source_index,split_group FROM documents WHERE split=? ORDER BY ordering', (split,)):
                    target.write(json.dumps({'text': text, 'sha256': digest(text.encode()),
                                             'source_index': index,
                                             **({'split_group': grouping} if grouping else {})}) + '\n')
            splits[split] = JsonlRows(path, sizes[split])
        connection.commit()
    finally:
        connection.close()
    return splits, {'counts': dict(counts), 'split_documents': sizes,
                    'split_policy': 'split_group-disjoint' if grouped else 'document-hash',
                    'decontamination': 'configured' if benchmark else 'not_configured',
                    'storage': 'sqlite/jsonl',
                    'limitations': 'Regex PII is heuristic. LSH near-dedup is approximate and may '
                                    'miss pairs; matched candidates use exact Jaccard.'}


class TokenBlocks:
    """Map tokens on disk; allocate only the requested batch as int64 tensors."""
    def __init__(self, path, tokens, context, shift):
        import numpy as np
        self.tokens = np.memmap(path, dtype='<u4', mode='r', shape=(tokens,))
        self.context, self.shift = context, shift

    def __len__(self):
        return (len(self.tokens) - 1) // self.context

    def __getitem__(self, index):
        import numpy as np
        import torch
        indices = (np.arange(*index.indices(len(self))) if isinstance(index, slice)
                   else index.detach().cpu().numpy() if isinstance(index, torch.Tensor)
                   else np.asarray(index))
        positions = indices[..., None] * self.context + np.arange(self.context) + self.shift
        if np.any(indices < 0) or np.any(indices >= len(self)):
            raise IndexError('Token block index is out of range')
        return torch.from_numpy(np.asarray(self.tokens[positions], dtype=np.int64))


def pack_disk(rows, tokenizer, tokenization, context, limit, folder, split):
    import numpy as np
    path = folder / ('tokens-' + split + '.bin')
    temporary = path.with_suffix('.tmp')
    count = 0
    with temporary.open('wb') as target:
        for row in rows:
            ids = tokenizer.encode(row['text'], add_special_tokens=False).ids
            if tokenization.add_bos:
                ids.insert(0, tokenizer.token_to_id('<bos>'))
            if tokenization.append_eos:
                ids.append(tokenizer.token_to_id('<eos>'))
            ids = ids[:max(0, limit - count)]
            target.write(np.asarray(ids, dtype='<u4').tobytes())
            count += len(ids)
            if count >= limit:
                break
    if (count - 1) // context < 1:
        temporary.unlink(missing_ok=True)
        raise ValueError('Split has too few tokens for a context block; increase corpus/token budget')
    temporary.replace(path)
    metadata = {'file': path.name, 'tokens': count, 'context': context, 'sha256': file_digest(path)}
    return (TokenBlocks(path, count, context, 0), TokenBlocks(path, count, context, 1)), metadata


def load_blocks(folder):
    import torch
    payload = torch.load(folder / 'tokens.pt', map_location='cpu', weights_only=True)
    if payload.get('format') != 'memmap-v1':
        return payload
    blocks = {}
    for name, metadata in payload['splits'].items():
        path = (folder / metadata['file']).resolve()
        if not path.is_relative_to(folder.resolve()) or not path.is_file():
            raise ValueError('Mapped token file is missing or outside the run directory')
        if path.stat().st_size != metadata['tokens'] * 4 or file_digest(path) != metadata['sha256']:
            raise ValueError('Mapped token file checksum/size changed')
        blocks[name] = tuple(TokenBlocks(path, metadata['tokens'], metadata['context'], shift)
                             for shift in (0, 1))
    return blocks
