"""Verify CUDA training and checkpoint recovery without downloading data."""

import json
import tempfile
from pathlib import Path

import torch


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the project environment")
    torch.manual_seed(42)
    device = torch.device("cuda")
    model = torch.nn.Linear(64, 32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(16, 64, device=device)
    targets = torch.randn(16, 32, device=device)
    precision = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
    before = model.weight.detach().clone()
    with torch.autocast("cuda", dtype=precision, enabled=precision == torch.bfloat16):
        loss = torch.nn.functional.mse_loss(model(inputs).float(), targets)
    loss.backward()
    if not torch.isfinite(loss) or not all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    ):
        raise RuntimeError("Loss or gradients are invalid")
    optimizer.step()
    if torch.equal(before, model.weight):
        raise RuntimeError("Optimizer did not update model weights")
    with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "smoke.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        restored = torch.load(checkpoint, map_location=device, weights_only=True)
        clone = torch.nn.Linear(64, 32).to(device)
        clone.load_state_dict(restored["model"])
        clone_optimizer = torch.optim.AdamW(clone.parameters(), lr=1e-3)
        clone_optimizer.load_state_dict(restored["optimizer"])
        if not all(torch.equal(a, b) for a, b in zip(model.parameters(), clone.parameters())):
            raise RuntimeError("Checkpoint model recovery failed")
    torch.cuda.synchronize()
    print(json.dumps({
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "precision": str(precision),
        "loss": loss.item(), "backward_and_optimizer": "passed",
        "checkpoint_recovery": "passed",
    }, indent=2))


if __name__ == "__main__":
    main()
