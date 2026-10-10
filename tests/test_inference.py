from pathlib import Path

import pytest

from ashayllm.inference import serve_plan, vllm_setup_plan


def test_shared_install(monkeypatch, tmp_path):
    monkeypatch.setattr("ashayllm.inference.platform.system", lambda: "Linux")
    plan = vllm_setup_plan(tmp_path, {"backend": "cuda"})
    assert plan["environment"] == str(tmp_path / ".venv")
    assert "vllm==0.31.0" in plan["commands"][0]
    assert "torch==2.13.0" in plan["commands"][0]
    assert str(tmp_path / ".venv/bin/python") in plan["commands"][0]


@pytest.mark.parametrize("backend", ["cpu", "mps", "rocm"])
def test_unsupported_automatic_install(backend, tmp_path):
    with pytest.raises(ValueError, match="requires Linux"):
        vllm_setup_plan(tmp_path, {"backend": backend})


@pytest.fixture
def exported_model(tmp_path):
    model = tmp_path / "export"
    model.mkdir()
    for filename in ["config.json", "tokenizer.json", "model.safetensors"]:
        (model / filename).write_text("{}")
    return {"inference": {"model_path": "export"}}


def test_local_offline_serving(exported_model, tmp_path):
    plan = serve_plan(exported_model, tmp_path)
    assert "--trust-remote-code" not in plan["command"]
    assert plan["environment"]["HF_HUB_OFFLINE"] == "1"
    assert "127.0.0.1" in plan["command"]
    assert str(tmp_path / "export") in plan["command"]
    assert Path(plan["command"][0]).parent.parent.name == ".venv"


def test_remote_model_rejected(tmp_path):
    with pytest.raises(ValueError, match="Local model"):
        serve_plan({"inference": {"model_path": "org/remote-model"}}, tmp_path)


@pytest.mark.parametrize("key,value", [
    ("gpu_memory_utilization", 1), ("gpu_memory_utilization", True),
    ("max_num_seqs", 0), ("max_model_len", -1), ("port", 70000), ("dtype", "bad"),
])
def test_invalid_serving_settings(exported_model, tmp_path, key, value):
    exported_model["inference"][key] = value
    with pytest.raises(ValueError):
        serve_plan(exported_model, tmp_path)
