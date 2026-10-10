"""Initial CLI: environment checks and configuration validation."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ashayllm.hardware import detect_hardware, resolve_runtime, setup_plan


def check_environment() -> dict:
    result = detect_hardware()
    try:
        import torch
    except ImportError:
        result["next_action"] = "Run ashayllm setup to inspect a hardware-aware installation plan."
        return result
    result["torch_installed"] = True
    result["torch_version"] = torch.__version__
    result["cuda_build"] = torch.version.cuda
    result["cuda_available"] = torch.cuda.is_available()
    if result["cuda_available"]:
        x = torch.randn(256, 256, device="cuda")
        result["cuda_matmul_smoke_test"] = bool(torch.isfinite(x @ x).all().item())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(prog="ashayllm")
    command = parser.add_subparsers(dest="command", required=True)
    command.add_parser("doctor", help="Inspect Python, PyTorch and CUDA")
    config_parser = command.add_parser("check-config", help="Validate initial YAML config")
    config_parser.add_argument("path", type=Path)
    runtime_parser = command.add_parser("resolve-runtime", help="Resolve auto device and precision")
    runtime_parser.add_argument("path", type=Path)
    runtime_parser.add_argument("--output", type=Path)
    setup_parser = command.add_parser("setup", help="Plan PyTorch setup; --apply installs packages")
    setup_parser.add_argument("--backend", choices=["auto", "cpu", "cuda", "rocm", "mps"],
                              default="auto")
    setup_parser.add_argument("--index-url")
    setup_parser.add_argument("--torch-version")
    setup_parser.add_argument("--apply", action="store_true")
    smoke_parser = command.add_parser("smoke", help="Test training/checkpoints on resolved hardware")
    smoke_parser.add_argument("--accelerator", default="auto")
    smoke_parser.add_argument("--precision", default="auto")
    inference_setup = command.add_parser("setup-vllm", help="Plan shared vLLM/PyTorch installation")
    inference_setup.add_argument("--root", type=Path, default=Path.cwd())
    inference_setup.add_argument("--version", default="0.31.0")
    inference_setup.add_argument("--apply", action="store_true")
    inference_serve = command.add_parser("serve-vllm", help="Plan or run local vLLM serving")
    inference_serve.add_argument("path", type=Path)
    inference_serve.add_argument("--root", type=Path, default=Path.cwd())
    inference_serve.add_argument("--run", action="store_true")
    aws_test = command.add_parser("aws-test", help="Plan a budget-limited Spot test; --run launches")
    aws_test.add_argument("path", type=Path, nargs="?", default=Path("configs/aws-test.yaml"))
    aws_test.add_argument("--instance-type")
    aws_test.add_argument("--run", action="store_true")
    aws_reconcile = command.add_parser("aws-reconcile", help="Recover reports/cleanup for a saved run")
    aws_reconcile.add_argument("run_id")
    aws_reconcile.add_argument("--path", type=Path, default=Path("configs/aws-test.yaml"))
    pipeline = command.add_parser("pipeline", help="Plan/run bounded end-to-end pretraining")
    pipeline.add_argument("path", type=Path)
    pipeline.add_argument("--run", action="store_true")
    pipeline.add_argument("--resume", type=Path)
    chat_parser = command.add_parser("chat", help="Interact with a trained run locally")
    chat_parser.add_argument("run_dir", type=Path)
    chat_parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    chat_parser.add_argument("--prompt")
    chat_parser.add_argument("--max-tokens", type=int, default=32)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-k", type=int, default=30)
    export_parser = command.add_parser("export", help="Export a trained run for HF/vLLM")
    export_parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    if args.command == "chat":
        from ashayllm.chat import chat
        chat(args.run_dir, args.backend, args.prompt, args.max_tokens, args.temperature, args.top_k)
    elif args.command == "export":
        from ashayllm.chat import export_model
        print(export_model(args.run_dir))
    elif args.command == "pipeline":
        from ashayllm.run_config import RunConfig
        with args.path.open(encoding="utf-8") as stream:
            config = RunConfig.model_validate(yaml.safe_load(stream))
        if args.run:
            from ashayllm.pipeline import run_pipeline
            run_pipeline(config, Path.cwd(), args.resume)
        else:
            print("Validated plan only: no dataset download, training or AWS launch. "
                  "Add --run to execute on this machine.")
            print(json.dumps(config.model_dump(mode="json"), indent=2))
    elif args.command in {"aws-test", "aws-reconcile"}:
        from ashayllm.aws_runner import SpotRunner
        with args.path.open(encoding="utf-8") as stream:
            settings = yaml.safe_load(stream)["aws_test"]
        if getattr(args, "instance_type", None):
            settings["instance_type"] = args.instance_type
        runner = SpotRunner(settings, Path.cwd())
        if args.command == "aws-reconcile":
            result = runner.reconcile(args.run_id)
        elif args.run:
            result = runner.run()
        else:
            result = runner.plan()
            result.pop("user_data")
        print(json.dumps(result, indent=2, default=str))
        if getattr(args, "run", False) and not result.get("test_passed"):
            parser.exit(1, "Spot instance terminated; test success not confirmed in console output\n")
    elif args.command in {"setup-vllm", "serve-vllm"}:
        from ashayllm.inference import launch_server, serve_plan, vllm_setup_plan
        try:
            if args.command == "setup-vllm":
                plan = vllm_setup_plan(args.root, detect_hardware(), args.version)
                print(json.dumps(plan, indent=2))
                if args.apply:
                    if not shutil.which("uv"):
                        parser.error("Install uv and add it to PATH first")
                    if not plan["c_compiler_available"]:
                        parser.error("Install a C compiler first (Ubuntu: sudo apt install "
                                     "build-essential)")
                    if not (Path(plan["environment"]) / "bin/python").is_file():
                        parser.error("Create the project's .venv environment first")
                    for install_command in plan["commands"]:
                        subprocess.run(install_command, check=True)
            else:
                with args.path.open(encoding="utf-8") as stream:
                    plan = serve_plan(yaml.safe_load(stream), args.root)
                print(json.dumps(plan, indent=2))
                if args.run:
                    launch_server(plan)
        except (ValueError, TypeError, OSError, subprocess.SubprocessError) as error:
            parser.error(str(error))
    elif args.command == "doctor":
        print(json.dumps(check_environment(), indent=2))
    elif args.command == "setup":
        try:
            plan = setup_plan(args.backend, args.index_url, args.torch_version)
            print(json.dumps(plan, indent=2))
            if args.apply:
                if not plan["python_supported"]:
                    parser.error("Create a Python 3.11/3.12 environment first")
                if sys.prefix == sys.base_prefix:
                    parser.error("Activate a project virtual environment before --apply")
                if not shutil.which("uv"):
                    parser.error("Install uv and add it to PATH first")
                subprocess.run(plan["command"], check=True)
        except (ValueError, subprocess.SubprocessError) as error:
            parser.error(str(error))
    elif args.command == "smoke":
        from ashayllm.smoke import run_smoke
        try:
            print(json.dumps(run_smoke(args.accelerator, args.precision), indent=2))
        except (ValueError, ImportError) as error:
            parser.error(str(error))
    elif args.command in {"check-config", "resolve-runtime"}:
        with args.path.open(encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if not isinstance(config, dict) or "project" not in config or "runtime" not in config:
            parser.error("Config must contain 'project' and 'runtime' sections")
        if args.command == "resolve-runtime":
            try:
                config["runtime"] = resolve_runtime(config["runtime"], detect_hardware())
            except (ValueError, TypeError) as error:
                parser.error(str(error))
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
