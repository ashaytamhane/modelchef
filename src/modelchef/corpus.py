"""Bounded corpus acquisition and small-sample filtering with auditable counts."""
import hashlib
import html
import json
import os
import random
import re
import time
import unicodedata
from collections import Counter
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    """Hash artifacts in bounded chunks instead of duplicating model weights in RAM."""
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        while chunk := source.read(1024 * 1024):
            checksum.update(chunk)
    return checksum.hexdigest()


def viewer_headers():
    """Use optional runtime authentication, never recipe/manifest credentials."""
    token = os.environ.get('HF_TOKEN')
    if not token:
        try:
            from huggingface_hub import get_token
        except ImportError:
            token = None
        else:
            token = get_token()
    headers = {'User-Agent': 'ModelChef/0.1'}
    if token:
        if any(char.isspace() for char in token):
            raise ValueError('HF_TOKEN contains whitespace; configure a valid token outside the recipe')
        headers['Authorization'] = 'Bearer ' + token
    return headers


def viewer_page(request, timeout, limit, explain):
    """Bounded retries; failed pages do not advance the offset or byte counter."""
    for attempt in range(7):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read(limit)
        except HTTPError as error:
            code = error.code
            retry_after = error.headers.get('Retry-After') if error.headers else None
            error.close()
            if code not in {429, 500, 502, 503, 504} or attempt == 6:
                raise
            delay = min(5 * 2 ** attempt, 60)
            if retry_after:
                try:
                    requested = float(retry_after)
                except ValueError:
                    try:
                        date = parsedate_to_datetime(retry_after)
                        requested = (date - datetime.now(UTC)).total_seconds()
                    except (TypeError, ValueError, OverflowError):
                        requested = 0
                delay = max(delay, requested)
            if delay > 300:
                raise ValueError('Dataset viewer requires a cooldown longer than five minutes; '
                                 'wait before rerunning or use a local corpus snapshot') from None
            explain(f'Dataset viewer HTTP {code}: waiting {delay:.0f}s before retrying the same '
                    f'page (retry {attempt + 1}/6); training has not started.')
            remaining = delay
            while remaining > 0:
                pause = min(remaining, 60)
                time.sleep(pause)
                remaining -= pause
        except (TimeoutError, ConnectionError, URLError) as error:
            if attempt == 6:
                raise
            delay = min(5 * 2 ** attempt, 60)
            explain(f'Dataset request interrupted ({type(error).__name__}); retrying the same '
                    f'request in {delay}s (retry {attempt + 1}/6).')
            time.sleep(delay)


def acquire(config, root, explain):
    c = config.data
    if c.storage == 'disk':
        from modelchef.disk_data import acquire_disk
        return acquire_disk(config, root, explain)
    identity = digest(json.dumps(c.model_dump(), sort_keys=True).encode())
    cache = root / "data/cache" / identity
    raw, manifest = cache / "raw.jsonl", cache / "source.json"
    if c.provider != "local_jsonl" and raw.exists() and manifest.exists():
        info = json.loads(manifest.read_text())
        content = raw.read_bytes()
        if digest(content) != info["raw_sha256"]:
            raise ValueError("Cached raw snapshot checksum changed; refuse silent reuse")
        explain(f"Reusing verified local snapshot: {info['documents']} documents; no download.")
        return [json.loads(line) for line in content.splitlines()], info
    records, downloaded = [], 0
    if c.provider == "local_jsonl":
        paths = [(root / name).resolve() for name in (c.local_paths or [c.local_path])]
        path = paths[0]
        from contextlib import ExitStack
        from itertools import chain
        with ExitStack() as stack:
            handles = [stack.enter_context(p.open('rb')) for p in paths]
            handle = iter(chain.from_iterable(iter(lambda h=h: h.readline(
                min(c.max_record_bytes, max(0, c.max_download_bytes - downloaded)) + 1), b'')
                for h in handles))
            index = 0
            for line in handle:
                downloaded += len(line)
                if downloaded > c.max_download_bytes:
                    raise ValueError("Local input exceeds configured byte budget")
                if len(line) > c.max_record_bytes:
                    raise ValueError('JSONL record exceeds max_record_bytes')
                if not line.strip():
                    continue
                if index < c.offset:
                    index += 1
                    continue
                row = json.loads(line)
                if config.stage.kind in {"sft", "dpo", "rlvr", "distill"}:
                    fields = config.stage.data_fields()
                    values = {key: row[key] for key in fields}
                    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
                        raise ValueError("Structured fields must be nonempty strings")
                    grouping = row.get('split_group')
                    if grouping is not None and (not isinstance(grouping, str) or not grouping):
                        raise ValueError('split_group must be a nonempty string')
                    records.append({**values, "text": " ".join(values.values()), "source_index": index,
                                    **({'split_group': grouping} if grouping else {})})
                else:
                    grouping = row.get('split_group')
                    if grouping is not None and (not isinstance(grouping, str) or not grouping):
                        raise ValueError('split_group must be a nonempty string')
                    records.append({"text": row[c.text_field], "source_index": index,
                                    **({'split_group': grouping} if grouping else {})})
                if len(records) >= c.max_documents:
                    break
                index += 1
        origin = {"provider": "local_jsonl", "path": str(path), "paths": [str(p) for p in paths]}
    elif c.provider == 'hf_parquet':
        from modelchef.bulk_data import acquire_parquet
        records, downloaded, origin = acquire_parquet(c, cache, explain)
    else:
        headers = viewer_headers()
        explain('Using authenticated dataset viewer requests.' if 'Authorization' in headers
                else 'Using anonymous dataset viewer requests; optional HF_TOKEN or hf auth login enables authentication.')
        explain("Downloading only viewer row pages, not the multi-GB Parquet corpus. "
                "This is an ordered smoke-test slice, not a representative random sample.")
        for offset in range(c.offset, c.offset + c.max_documents, c.page_size):
            if offset != c.offset:
                time.sleep(2)  # Pace viewer requests instead of bursting through the API.
            query = urlencode({"dataset": c.dataset, "config": c.subset, "split": c.split,
                               "offset": offset,
                               "length": min(c.page_size, c.offset + c.max_documents - offset)})
            request = Request("https://datasets-server.huggingface.co/rows?" + query,
                              headers=headers)
            payload = viewer_page(request, c.timeout_seconds,
                                  c.max_download_bytes - downloaded + 1, explain)
            downloaded += len(payload)
            if downloaded > c.max_download_bytes:
                raise ValueError("Download exceeded byte cap; no complete snapshot cached")
            page = json.loads(payload)
            if not page["rows"]:
                break
            for entry in page["rows"]:
                if c.text_field in entry.get("truncated_cells", []):
                    raise ValueError("Viewer text was truncated; use a complete local snapshot")
                records.append({"text": entry["row"][c.text_field],
                                "source_index": entry["row_idx"]})
            explain(f"Acquired {len(records)} rows; {downloaded:,} response bytes / "
                    f"{c.max_download_bytes:,} byte cap.")
        origin = {"provider": "hf_rows", "dataset": c.dataset, "subset": c.subset,
                  "split": c.split, "offset": c.offset,
                  "dataset_card": f"https://huggingface.co/datasets/{c.dataset}",
                  "revision": None,
                  "revision_note": "Viewer API is mutable; reproducibility pins the saved raw SHA256, "
                  "not an upstream Git revision. Preserve snapshot for exact replay."}
    content = b"".join((json.dumps(row, ensure_ascii=False) + "\n").encode() for row in records)
    info = {"schema_version": 1, **origin, "documents": len(records),
            "response_bytes": downloaded, "raw_sha256": digest(content)}
    cache.mkdir(parents=True, exist_ok=True)
    raw.write_bytes(content)
    manifest.write_text(json.dumps(info, indent=2), encoding="utf-8")
    return records, info


def ngrams(text, size):
    words = re.findall(r"\w+", text.casefold())
    return {tuple(words[i:i + size]) for i in range(len(words) - size + 1)}


PII = re.compile(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}|"
                 # Preserve ordinary decimal measurements. Dot-separated phone numbers
                 # have multiple separators; a single decimal point is not sufficient.
                 r"(?<![\w.])(?!(?:\+?\d+\.\d+)(?![\d.]))"
                 r"(?:\+?\d[\d ().-]{7,}\d)(?![\w.])")


def clean_record(row, config, benchmark, counts):
    """Normalize/filter one record; deduplication and splitting are separate policies."""
    c = config.processing
    counts["input"] += 1
    text = row["text"]
    if not isinstance(text, str):
        counts["invalid_text"] += 1
        return None
    text = unicodedata.normalize(c.unicode_normalization, html.unescape(text))
    if c.strip_html:
        text = re.sub(r"<[^>]+>", " ", text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    repetition = 1 - len(set(lines)) / max(len(lines), 1)
    text = " ".join(" ".join(lines).split())
    if c.collapse_consecutive_duplicate_sentences:
        from modelchef.text_cleaning import collapse_duplicate_sentences
        text, removed, chars = collapse_duplicate_sentences(text, c.minimum_sentence_words)
        if removed:
            counts['sentence_cleanup_documents'] += 1
            counts['duplicate_sentences_removed'] += removed
            counts['duplicate_sentence_chars_removed'] += chars
    if not c.min_chars <= len(text) <= c.max_chars:
        counts["length"] += 1
        return None
    if sum(char.isalnum() for char in text) / len(text) < c.min_alnum_fraction:
        counts["quality"] += 1
        return None
    if repetition > c.max_repeated_line_fraction:
        counts["repetition"] += 1
        return None
    if config.data.domain_keywords and not any(
            word.casefold() in text.casefold() for word in config.data.domain_keywords):
        counts["domain"] += 1
        return None
    if c.pii != "off" and PII.search(text):
        if c.pii == "drop":
            counts["pii_dropped"] += 1
            return None
        text, matches = PII.subn("[REDACTED]", text)
        counts["pii_redactions"] += matches
    if benchmark and benchmark & ngrams(text, c.decontamination_ngram_words):
        counts["benchmark_overlap"] += 1
        return None
    return {"text": text, "sha256": digest(text.encode()), "source_index": row["source_index"],
            **({'split_group': row['split_group']} if 'split_group' in row else {})}


def group_split(key, config):
    fraction = int(digest(f'{config.training.seed}:{key}'.encode())[:16], 16) / 2**64
    c = config.processing
    return ('validation' if fraction < c.validation_fraction else
            'test' if fraction < c.validation_fraction + c.test_fraction else 'train')


def process(records, config, root):
    c = config.processing
    if config.data.storage == 'disk':
        from modelchef.disk_data import process_disk
        return process_disk(records, config, root)
    benchmark = set()
    for path in c.benchmark_files:
        benchmark |= ngrams((root / path).read_text(encoding="utf-8"),
                            c.decontamination_ngram_words)
    counts, accepted, exact, near = Counter(), [], set(), []
    for row in records:
        cleaned = clean_record(row, config, benchmark, counts)
        if cleaned is None:
            continue
        text = cleaned['text']
        fingerprint = digest(text.casefold().encode())
        if c.exact_dedup and fingerprint in exact:
            counts["exact_duplicates"] += 1
            continue
        shingles = ngrams(text, c.dedup_ngram_words)
        if c.near_dedup and shingles and any(
                len(shingles & previous) / len(shingles | previous) >= c.near_dedup_jaccard
                for previous in near):
            counts["near_duplicates"] += 1
            continue
        exact.add(fingerprint)
        if c.near_dedup and shingles:
            near.append(shingles)
        accepted.append(cleaned)
    counts["accepted"] = len(accepted)
    if len(accepted) < 3:
        raise ValueError(f"Too few documents after processing: {dict(counts)}; "
                         "adjust filters/domain or choose another bounded sample")
    random.Random(config.training.seed).shuffle(accepted)
    # Rounding two held-out fractions must leave at least one training document,
    # including the smallest supported three-document dataset.
    validation_count = min(len(accepted) - 2, max(1, int(len(accepted) * c.validation_fraction)))
    test_count = min(len(accepted) - validation_count - 1,
                     max(1, int(len(accepted) * c.test_fraction)))
    splits = {"validation": accepted[:validation_count],
              "test": accepted[validation_count:validation_count + test_count],
              "train": accepted[validation_count + test_count:]}
    grouped = any('split_group' in row for row in accepted)
    if grouped:
        splits = {name: [] for name in ['train', 'validation', 'test']}
        for row in accepted:
            key = row.get('split_group', row['sha256'])
            splits[group_split(key, config)].append(row)
        if any(not rows for rows in splits.values()):
            raise ValueError('Grouped data has an empty split; acquire more independent groups')
    return splits, {"counts": dict(counts),
                    "split_policy": 'split_group-disjoint' if grouped else 'document-shuffle',
                    "decontamination": "configured" if c.benchmark_files else "not_configured",
                    "limitations": "Regex PII and lexical near-dedup are heuristics; "
                    "no guarantee of complete PII removal or benchmark decontamination.",
                    "split_documents": {key: len(value) for key, value in splits.items()}}
