"""Small synthetic training check shared by supported PyTorch backends."""

import tempfile
from contextlib import nullcontext
from pathlib import Path

from ashayllm.hardware import detect_hardware, resolve_runtime


def run_smoke(accelerator="auto", precision="auto"):
    import torch

    runtime = resolve_runtime({"accelerator": accelerator, "precision": precision},
                              detect_hardware())
    device = runtime["device"]
    torch.manual_seed(42)
    model = torch.nn.Linear(64, 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 64, device=device)
    targets = torch.randn(16, 32, device=device)
    before = model.weight.detach().clone()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        runtime["precision"]]
    context = (nullcontext() if dtype == torch.float32 else
               torch.autocast(device_type=device, dtype=dtype))
    scaler = torch.amp.GradScaler("cuda", enabled=runtime["requires_grad_scaler"])
    with context:
        loss = torch.nn.functional.mse_loss(model(inputs).float(), targets)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    if not torch.isfinite(loss) or not all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise RuntimeError("Loss or gradients are invalid")
    scaler.step(optimizer)
    scaler.update()
    if torch.equal(before, model.weight):
        raise RuntimeError("Optimizer did not update weights")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "smoke.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, path)
        state = torch.load(path, map_location=device, weights_only=True)
        clone = torch.nn.Linear(64, 32).to(device)
        clone.load_state_dict(state["model"])
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-3)
        clone_optimizer.load_state_dict(state["optimizer"])
        for original, restored in zip(model.parameters(), clone.parameters()):
            if not torch.equal(original, restored):
                raise RuntimeError("Checkpoint weights differ")
            for key, value in optimizer.state[original].items():
                recovered = clone_optimizer.state[restored][key]
                equal = (torch.equal(value.cpu(), recovered.cpu()) if torch.is_tensor(value)
                         else value == recovered)
                if not equal:
                    raise RuntimeError("Checkpoint optimizer state differs")
    return {"runtime": runtime, "loss": loss.item(), "training_step": "passed",
            "checkpoint_recovery": "passed"}
