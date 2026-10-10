"""Bounded corpus acquisition and small-sample filtering with auditable counts."""
import hashlib
import html
import json
import random
import re
import unicodedata
from collections import Counter
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def digest(value):
    return hashlib.sha256(value).hexdigest()


def acquire(config, root, explain):
    c = config.data
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
        path = (root / c.local_path).resolve()
        with path.open("rb") as handle:
            for index, line in enumerate(handle):
                downloaded += len(line)
                if downloaded > c.max_download_bytes:
                    raise ValueError("Local input exceeds configured byte budget")
                if index < c.offset:
                    continue
                row = json.loads(line)
                if config.stage.kind in {"sft", "dpo", "rlvr", "distill"}:
                    fields = config.stage.data_fields()
                    values = {key: row[key] for key in fields}
                    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
                        raise ValueError("Structured fields must be nonempty strings")
                    records.append({**values, "text": " ".join(values.values()), "source_index": index})
                else:
                    records.append({"text": row[c.text_field], "source_index": index})
                if len(records) >= c.max_documents:
                    break
        origin = {"provider": "local_jsonl", "path": str(path)}
    else:
        explain("Downloading only viewer row pages, not the multi-GB Parquet corpus. "
                "This is an ordered smoke-test slice, not a representative random sample.")
        for offset in range(c.offset, c.offset + c.max_documents, c.page_size):
            query = urlencode({"dataset": c.dataset, "config": c.subset, "split": c.split,
                               "offset": offset,
                               "length": min(c.page_size, c.offset + c.max_documents - offset)})
            request = Request("https://datasets-server.huggingface.co/rows?" + query,
                              headers={"User-Agent": "AshayLLM/0.1"})
            with urlopen(request, timeout=c.timeout_seconds) as response:
                payload = response.read(c.max_download_bytes - downloaded + 1)
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
                 r"(?<!\w)(?:\+?\d[\d ().-]{7,}\d)(?!\w)")


def process(records, config, root):
    c = config.processing
    benchmark = set()
    for path in c.benchmark_files:
        benchmark |= ngrams((root / path).read_text(encoding="utf-8"),
                            c.decontamination_ngram_words)
    counts, accepted, exact, near = Counter(), [], set(), []
    for row in records:
        counts["input"] += 1
        text = row["text"]
        if not isinstance(text, str):
            counts["invalid_text"] += 1
            continue
        text = unicodedata.normalize(c.unicode_normalization, html.unescape(text))
        if c.strip_html:
            text = re.sub(r"<[^>]+>", " ", text)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        repetition = 1 - len(set(lines)) / max(len(lines), 1)
        text = " ".join(" ".join(lines).split())
        if not c.min_chars <= len(text) <= c.max_chars:
            counts["length"] += 1
            continue
        if sum(char.isalnum() for char in text) / len(text) < c.min_alnum_fraction:
            counts["quality"] += 1
            continue
        if repetition > c.max_repeated_line_fraction:
            counts["repetition"] += 1
            continue
        if config.data.domain_keywords and not any(
                word.casefold() in text.casefold() for word in config.data.domain_keywords):
            counts["domain"] += 1
            continue
        if c.pii != "off" and PII.search(text):
            if c.pii == "drop":
                counts["pii_dropped"] += 1
                continue
            text, matches = PII.subn("[REDACTED]", text)
            counts["pii_redactions"] += matches
        if benchmark and benchmark & ngrams(text, c.decontamination_ngram_words):
            counts["benchmark_overlap"] += 1
            continue
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
        accepted.append({"text": text, "sha256": digest(text.encode()),
                         "source_index": row["source_index"]})
    counts["accepted"] = len(accepted)
    if len(accepted) < 10:
        raise ValueError(f"Too few documents after processing: {dict(counts)}; "
                         "adjust filters/domain or choose another bounded sample")
    random.Random(config.training.seed).shuffle(accepted)
    validation_count = max(1, int(len(accepted) * c.validation_fraction))
    test_count = max(1, int(len(accepted) * c.test_fraction))
    splits = {"validation": accepted[:validation_count],
              "test": accepted[validation_count:validation_count + test_count],
              "train": accepted[validation_count + test_count:]}
    return splits, {"counts": dict(counts),
                    "decontamination": "configured" if c.benchmark_files else "not_configured",
                    "limitations": "Regex PII and lexical near-dedup are heuristics; "
                    "no guarantee of complete PII removal or benchmark decontamination.",
                    "split_documents": {key: len(value) for key, value in splits.items()}}
