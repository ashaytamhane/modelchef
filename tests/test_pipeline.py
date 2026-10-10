import json
from pathlib import Path

import pytest

pytest.importorskip("pydantic")

from ashayllm.benchmarks import select_benchmark
from ashayllm.corpus import acquire, process
from ashayllm.run_config import RunConfig


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


def test_causal_logits_cannot_see_future_tokens():
    torch = pytest.importorskip("torch")
    from ashayllm.model import Decoder
    from ashayllm.run_config import Architecture
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
    from ashayllm.pipeline import run_pipeline
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
    from ashayllm.chat import export_model
    from ashayllm.model import Decoder
    exported = export_model(Path(result["run_dir"]))
    hf_model = transformers.LlamaForCausalLM.from_pretrained(exported).eval()
    native = Decoder(config.model, hf_model.config.vocab_size).eval()
    native.load_state_dict(final)
    probe = torch.randint(hf_model.config.vocab_size, (1, 12))
    with torch.no_grad():
        assert torch.allclose(native(probe), hf_model(probe).logits, atol=1e-5, rtol=1e-4)
