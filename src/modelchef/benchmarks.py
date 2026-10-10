"""Domain-aware small benchmark adapters; explicit protocols, budgets and provenance."""
import io
import json
import zipfile
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from modelchef.corpus import digest

REGISTRY = {"general": "hellaswag", "finance": "financial_phrasebank"}


def select_benchmark(config):
    return REGISTRY[config.data.domain] if config.evaluation.benchmark == "auto" else (
        config.evaluation.benchmark)


def acquire_benchmark(config, root):
    name = select_benchmark(config)
    if name == "none":
        return [], {"name": "none", "status": "disabled"}
    count, cap = config.evaluation.benchmark_examples, config.evaluation.benchmark_max_download_bytes
    cache = root / "data/cache/benchmarks" / f"{name}-{count}"
    cache.mkdir(parents=True, exist_ok=True)
    snapshot = cache / "examples.json"
    metadata = cache / "manifest.json"
    if snapshot.exists() and metadata.exists():
        source = json.loads(metadata.read_text())
        if digest(snapshot.read_bytes()) != source["sha256"]:
            raise ValueError("Benchmark cache checksum mismatch")
        return json.loads(snapshot.read_text()), source
    if name == "hellaswag":
        query = urlencode({"dataset": "Rowan/hellaswag", "config": "default",
                           "split": "validation", "offset": 0, "length": count})
        url = "https://datasets-server.huggingface.co/rows?" + query
    else:
        url = ("https://huggingface.co/datasets/takala/financial_phrasebank/resolve/main/"
               "data/FinancialPhraseBank-v1.0.zip")
    with urlopen(Request(url, headers={"User-Agent": "ModelChef/0.1"}),
                 timeout=config.data.timeout_seconds) as response:
        payload = response.read(cap + 1)
    if len(payload) > cap:
        raise ValueError("Benchmark download exceeded its byte cap")
    if name == "hellaswag":
        rows = json.loads(payload)["rows"]
        if any(entry.get("truncated_cells") for entry in rows):
            raise ValueError("Benchmark viewer returned truncated fields")
        examples = [{"prompt": entry["row"]["ctx"], "choices": entry["row"]["endings"],
                     "label": int(entry["row"]["label"]),
                     "decontamination_text": entry["row"]["ctx"] + " " + " ".join(
                         entry["row"]["endings"])} for entry in rows]
        card = "https://huggingface.co/datasets/Rowan/hellaswag"
        license_name = "mit"
    else:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            member = next(m for m in archive.infolist() if m.filename.endswith("Sentences_AllAgree.txt"))
            if member.file_size > cap * 5:
                raise ValueError("Benchmark archive expansion exceeds limit")
            lines = archive.read(member).decode("cp1252").splitlines()[:count]
        examples = []
        labels = ["negative", "neutral", "positive"]
        for line in lines:
            sentence, label = line.rsplit("@", 1)
            examples.append({"prompt": "Financial news: " + sentence + "\nSentiment:",
                             "choices": labels, "label": labels.index(label.strip()),
                             "decontamination_text": sentence})
        card = "https://huggingface.co/datasets/takala/financial_phrasebank"
        license_name = "cc-by-nc-sa-3.0"
    if not examples:
        raise ValueError("Benchmark has no usable examples")
    snapshot.write_text(json.dumps(examples), encoding="utf-8")
    source = {"name": name, "examples": len(examples), "dataset_card": card,
              "license": license_name, "download_bytes": len(payload),
              "sha256": digest(snapshot.read_bytes()), "upstream_payload_sha256": digest(payload),
              "revision": None, "sample": "first rows; not a representative/full benchmark",
              "decontamination_scope": "only the evaluated sampled texts"}
    metadata.write_text(json.dumps(source, indent=2), encoding="utf-8")
    return examples, source


def evaluate_choices(model, tokenizer, examples, device, context, amp):
    import torch
    import torch.nn.functional as F

    model.eval()
    results, truncated = [], 0
    with torch.no_grad():
        for example in examples:
            prefix = tokenizer.encode(example["prompt"]).ids
            scores = []
            for choice in example["choices"]:
                candidate = tokenizer.encode(" " + choice).ids
                if not candidate or not prefix:
                    raise ValueError("Empty benchmark tokenization")
                clipped = candidate[:context - 1]
                kept_prefix = prefix[-(context + 1 - len(clipped)):]
                truncated += int(len(clipped) != len(candidate) or len(kept_prefix) != len(prefix))
                ids = torch.tensor([kept_prefix + clipped], device=device)
                with amp():
                    logits = model(ids[:, :-1]).float()
                selected = logits[:, len(kept_prefix) - 1:]
                targets = ids[:, len(kept_prefix):]
                scores.append(-F.cross_entropy(selected.flatten(0, 1), targets.flatten()).item())
            prediction = max(range(len(scores)), key=scores.__getitem__)
            results.append({"prediction": prediction, "label": example["label"], "scores": scores})
    classes = len(examples[0]["choices"])
    f1 = []
    for label in range(classes):
        tp = sum(r["prediction"] == label and r["label"] == label for r in results)
        fp = sum(r["prediction"] == label and r["label"] != label for r in results)
        fn = sum(r["prediction"] != label and r["label"] == label for r in results)
        f1.append(2 * tp / max(1, 2 * tp + fp + fn))
    return {"accuracy": sum(r["prediction"] == r["label"] for r in results) / len(results),
            "macro_f1": sum(f1) / len(f1), "examples": len(results),
            "uniform_random_expected_accuracy": sum(1 / len(e["choices"]) for e in examples) / len(examples),
            "truncated_choices": truncated, "results": results,
            "protocol": "zero-shot, separately encoded continuations, mean token log probability; "
            "left-truncate prompt/right-truncate choice to context. Not lm-eval-harness official scores."}
