import pytest

from modelchef.hardware import resolve_runtime, setup_plan


@pytest.mark.parametrize("backend,bf16,device,precision", [
    ("cpu", False, "cpu", "fp32"), ("mps", False, "mps", "fp32"),
    ("cuda", True, "cuda", "bf16"), ("cuda", False, "cuda", "fp32"),
    ("rocm", True, "cuda", "bf16"),
])
def test_runtime_auto(backend, bf16, device, precision):
    result = resolve_runtime({}, {"backend": backend, "bf16_supported": bf16})
    assert (result["device"], result["precision"]) == (device, precision)


def test_cpu_override_on_gpu():
    result = resolve_runtime({"accelerator": "cpu"},
                             {"backend": "cuda", "bf16_supported": True})
    assert result["precision"] == "fp32"


@pytest.mark.parametrize("runtime", [
    {"accelerator": "cuda"}, {"precision": "bf16"}, {"precision": "fp16"},
    {"accelerator": "typo"}, {"precision": "typo"}, None,
])
def test_reject_invalid_or_unavailable(runtime):
    with pytest.raises((ValueError, TypeError)):
        resolve_runtime(runtime, {"backend": "cpu", "bf16_supported": False})


def test_cpu_install_plan(monkeypatch):
    monkeypatch.setattr("modelchef.hardware.shutil.which", lambda _: None)
    monkeypatch.setattr("modelchef.hardware.platform.system", lambda: "Linux")
    plan = setup_plan()
    assert plan["backend"] == "cpu"
    assert "https://download.pytorch.org/whl/cpu" in plan["command"]


def test_rocm_requires_explicit_wheel_index(monkeypatch):
    monkeypatch.setattr("modelchef.hardware.shutil.which", lambda _: None)
    monkeypatch.setattr("modelchef.hardware.platform.system", lambda: "Linux")
    with pytest.raises(ValueError, match="index-url"):
        setup_plan("rocm")


def test_old_nvidia_driver_rejected(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr("modelchef.hardware.shutil.which", lambda _: "/usr/bin/nvidia-smi")
    monkeypatch.setattr("modelchef.hardware.subprocess.run",
                        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="535.0\n"))
    with pytest.raises(ValueError, match="580"):
        setup_plan("cuda")


def test_cpu_training_checkpoint():
    pytest.importorskip("torch")
    from modelchef.smoke import run_smoke
    result = run_smoke("cpu", "fp32")
    assert result["checkpoint_recovery"] == "passed"


def test_fp16_requires_scaler():
    result = resolve_runtime({"precision": "fp16"},
                             {"backend": "cuda", "bf16_supported": True})
    assert result["requires_grad_scaler"]


def test_setup_default_does_not_install(monkeypatch, capsys):
    import sys

    from modelchef.cli import main

    monkeypatch.setattr(sys, "argv", ["modelchef", "setup", "--backend", "cpu"])
    monkeypatch.setattr("modelchef.hardware.shutil.which", lambda _: None)
    def forbidden_install(*args, **kwargs):
        pytest.fail("Dry-run setup attempted installation")
    monkeypatch.setattr("modelchef.cli.subprocess.run", forbidden_install)
    main()
    assert '"command"' in capsys.readouterr().out
