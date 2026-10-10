import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from modelchef.benchmarks import select_benchmark
from modelchef.corpus import acquire, process
from modelchef.run_config import RunConfig


def test_strict_recipe_and_domain_benchmarks():
    from pydantic import ValidationError
    assert select_benchmark(RunConfig()) == "hellaswag"
    config = RunConfig(data={"domain": "finance", "domain_keywords": ["revenue"]})
    assert select_benchmark(config) == "financial_phrasebank"
    with pytest.raises(ValidationError):
        RunConfig(model={"attention_heads": 7})
    with pytest.raises(ValidationError):
        RunConfig(training={"learnig_rate": 0.01})
    with pytest.raises(ValidationError):
        RunConfig(data={"domain": "medical", "domain_keywords": ["medicine"]})


def test_processing_dedup_pii_decontamination_and_disjoint_splits(tmp_path):
    config = RunConfig(processing={"min_chars": 20, "near_dedup": False,
                                   "decontamination_ngram_words": 4})
    text = "Education and science explain our world for curious students. "
    rows = [{"text": text + str(i), "source_index": i} for i in range(20)]
    rows += [{"text": rows[0]["text"], "source_index": 21},
             {"text": "Contact somebody@example.com about the complete education plan.",
              "source_index": 22},
             {"text": "Private benchmark example should never enter the training dataset.",
              "source_index": 23}]
    benchmark = tmp_path / "benchmark.txt"
    benchmark.write_text("Private benchmark example should never enter")
    config.processing.benchmark_files = [str(benchmark)]
    splits, info = process(rows, config, tmp_path)
    assert info["counts"]["exact_duplicates"] == 1
    assert info["counts"]["pii_redactions"] == 1
    assert info["counts"]["benchmark_overlap"] == 1
    sets = [{row["sha256"] for row in values} for values in splits.values()]
    assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
    assert "somebody@example.com" not in json.dumps(splits)


def test_acquire_enforces_local_byte_budget(tmp_path):
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"text": "words " * 100}) + "\n")
    config = RunConfig(data={"provider": "local_jsonl", "local_path": str(path),
                            "max_download_bytes": 10})
    with pytest.raises(ValueError, match="byte budget"):
        acquire(config, tmp_path, lambda _: None)


def test_local_byte_budget_bounds_the_actual_read(tmp_path, monkeypatch):
    import io
    original_open = Path.open
    sizes = []

    class BoundedReader(io.BytesIO):
        def readline(self, size=-1):
            sizes.append(size)
            return super().readline(size)

    source = tmp_path / 'huge-line.jsonl'
    reader = BoundedReader(b'{"text":"' + b'x' * 10000 + b'"}\n')

    def open_source(path, *args, **kwargs):
        return reader if path == source else original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, 'open', open_source)
    config = RunConfig(data={'provider': 'local_jsonl', 'local_path': str(source),
                            'max_download_bytes': 32})
    with pytest.raises(ValueError, match='byte budget'):
        acquire(config, tmp_path, lambda _: None)
    assert sizes == [33]


def test_artifact_digest_streams_large_files(tmp_path, monkeypatch):
    import io

    from modelchef.corpus import digest, file_digest
    payload = b'weights' * 400000
    sizes = []

    class Reader(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    monkeypatch.setattr(Path, 'open', lambda *args, **kwargs: Reader(payload))
    assert file_digest(tmp_path / 'weights.pt') == digest(payload)
    assert len(sizes) > 2 and all(size == 1024 * 1024 for size in sizes)


@pytest.mark.parametrize('fractions', [(0.1, 0.1), (0.8, 0.1)])
def test_three_document_sample_has_nonempty_disjoint_splits(tmp_path, fractions):
    config = RunConfig(data={'max_documents': 3}, processing={
        'min_chars': 1, 'near_dedup': False,
        'validation_fraction': fractions[0], 'test_fraction': fractions[1]})
    rows = [{'text': f'Science lesson {i} explains observation.', 'source_index': i}
            for i in range(3)]
    splits, _ = process(rows, config, tmp_path)
    assert {name: len(rows) for name, rows in splits.items()} == {
        'train': 1, 'validation': 1, 'test': 1}
    assert len({row['source_index'] for rows in splits.values() for row in rows}) == 3


@pytest.mark.parametrize('change', ['missing_raw', 'missing_benchmark', 'tokenizer',
                                   'tokens', 'decontamination', 'data'])
def test_resume_refuses_changed_or_missing_evidence_before_overwriting(tmp_path, monkeypatch, change):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tokenizers')
    from modelchef import pipeline
    torch.set_num_threads(1)
    source = tmp_path / 'raw.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Science lesson {i}: observations matter. ' * 5})
                              + '\n' for i in range(12)))
    excluded = tmp_path / 'benchmark.txt'
    excluded.write_text('An unrelated benchmark question about ancient architecture.')
    config = RunConfig(name='evidence-test', explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(source), 'max_tokens': 3000},
        processing={'min_chars': 1, 'near_dedup': False, 'benchmark_files': [str(excluded)]},
        tokenizer={'vocab_size': 300},
        model={'layers': 1, 'hidden_size': 16, 'attention_heads': 2, 'kv_heads': 1,
               'intermediate_size': 32, 'context_length': 8},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})
    result = pipeline.run_pipeline(config, tmp_path)
    folder = Path(result['run_dir'])
    if change == 'missing_raw':
        (folder / 'raw.jsonl').unlink()
    elif change == 'missing_benchmark':
        (folder / 'benchmark-examples.json').unlink()
    elif change == 'decontamination':
        excluded.write_text('Changed benchmark input.')
    elif change == 'data':
        # A raw snapshot can be internally checksum-valid yet differ from the
        # actual token sequences recorded by the checkpoint.
        from modelchef.corpus import digest
        rows = [json.loads(line) for line in (folder / 'raw.jsonl').read_text().splitlines()]
        rows[0]['text'] = 'Changed material about economics and banking. ' * 5
        (folder / 'raw.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        manifest = json.loads((folder / 'source-manifest.json').read_text())
        manifest['raw_sha256'] = digest((folder / 'raw.jsonl').read_bytes())
        (folder / 'source-manifest.json').write_text(json.dumps(manifest))
    else:
        target = folder / ('tokenizer.json' if change == 'tokenizer' else 'tokens.pt')
        target.write_bytes(target.read_bytes() + b' ')
    evidence = {path: path.read_bytes() for path in folder.iterdir()
                if path.is_file() and path.name not in {'report.json', 'events.jsonl'}}

    def forbidden_download(*args, **kwargs):
        pytest.fail('Resume must not reacquire missing snapshots')

    monkeypatch.setattr(pipeline, 'acquire', forbidden_download)
    monkeypatch.setattr(pipeline, 'acquire_benchmark', forbidden_download)
    with pytest.raises(ValueError, match='missing|changed|differs'):
        pipeline.run_pipeline(config, tmp_path, folder / 'checkpoints/step-000001.pt')
    assert all(path.read_bytes() == content for path, content in evidence.items())


def test_causal_logits_cannot_see_future_tokens():
    torch = pytest.importorskip("torch")
    from modelchef.model import Decoder
    from modelchef.run_config import Architecture
    model = Decoder(Architecture(layers=1, hidden_size=32, attention_heads=4,
                                kv_heads=2, intermediate_size=64, context_length=16), 300).eval()
    ids = torch.randint(300, (1, 16))
    changed = ids.clone()
    changed[:, 8:] = torch.randint(300, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(ids)[:, :8], model(changed)[:, :8], atol=1e-6)


def test_complete_cpu_pipeline_and_exact_resume(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("tokenizers")
    from modelchef.pipeline import run_pipeline
    torch.set_num_threads(1)
    path = tmp_path / "input.jsonl"
    path.write_text("".join(json.dumps({"text":
        f"Lesson {index}: " + (f"Students study topic{index} and practice reading with teachers. " * 8)
    }) + "\n" for index in range(30)))
    config = RunConfig(name="cpu-test", explain="brief",
        data={"provider": "local_jsonl", "local_path": str(path), "max_documents": 30,
              "max_tokens": 5000},
        processing={"min_chars": 20, "near_dedup": False}, tokenizer={"vocab_size": 300},
        model={"layers": 1, "hidden_size": 32, "attention_heads": 4, "kv_heads": 2,
               "intermediate_size": 64, "context_length": 16, "residual_dropout": 0.1},
        training={"accelerator": "cpu", "precision": "fp32", "steps": 4, "warmup_steps": 1,
                  "micro_batch_size": 1, "gradient_accumulation": 1, "learning_rate": 0.01,
                  "checkpoint_every": 2},
        evaluation={"benchmark": "none", "batches": 1, "generate_tokens": 2})
    result = run_pipeline(config, tmp_path)
    assert result["status"] == "completed" and result["checkpoint_recovery"] == "passed"
    final = torch.load(result["checkpoint"], weights_only=True)["model"]
    middle = Path(result["run_dir"]) / "checkpoints/step-000002.pt"
    resumed = run_pipeline(config, tmp_path, middle)
    restored = torch.load(resumed["checkpoint"], weights_only=True)["model"]
    assert all(torch.equal(final[key], restored[key]) for key in final)
    assert result["validation"]["loss"] == resumed["validation"]["loss"]
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    from modelchef.chat import export_model
    from modelchef.model import Decoder
    exported = export_model(Path(result["run_dir"]))
    hf_model = transformers.LlamaForCausalLM.from_pretrained(exported).eval()
    native = Decoder(config.model, hf_model.config.vocab_size).eval()
    native.load_state_dict(final)
    probe = torch.randint(hf_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        assert torch.allclose(native(probe), hf_model(probe).logits, atol=1e-5, rtol=1e-4)
