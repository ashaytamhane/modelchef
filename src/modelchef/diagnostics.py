"""Environment diagnostics shared by local and EC2 readiness commands."""
from modelchef.hardware import detect_hardware
from modelchef.progress import report


def check_environment() -> dict:
    result = detect_hardware(progress=report)
    try:
        import torch
    except ImportError:
        result["next_action"] = "Run modelchef setup to inspect a hardware-aware installation plan."
        return result
    result["torch_installed"] = True
    result["torch_version"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        report('Running a small GPU matrix multiplication and checking finite results.')
        x = torch.randn(256, 256, device="cuda")
        result["cuda_matmul_smoke_test"] = bool(torch.isfinite(x @ x).all().item())
        report('GPU matrix multiplication check finished.')
    return result
