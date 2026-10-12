"""CLI for data, model training stages, inference, evaluation and explicit AWS runs."""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from modelchef.diagnostics import check_environment
from modelchef.hardware import detect_hardware, resolve_runtime, setup_plan
from modelchef.progress import operation


def _settings(path, section):
    with Path(path).open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict) or not isinstance(config.get(section), dict):
        raise TypeError(f"{path} requires a '{section}' mapping")
    return config[section]


def confirm_cloud(plan, settings, yes=False):
    """Show bounded spending assumptions before any billable training launch."""
    market = settings.get('market_type', 'spot')
    rate = (f"current Spot price ${plan['spot_price_usd_hour']}/hour" if market == 'spot'
            else f"configured On-Demand budget rate ${settings['on_demand_hourly_usd']}/hour")
    print(f"AWS launch review: {plan['instance_type']} ({market}); {rate}; "
          f"deadline {settings['max_runtime_minutes']} minutes; "
          f"budget reservation ${plan['reserved_usd']}; overall configured limit "
          f"${settings['total_budget_usd']}. Reservation is a conservative allowance, "
          "not an exact bill. Retained storage can incur additional charges.", file=sys.stderr, flush=True)
    if yes:
        return
    print('Launch this billable AWS job? Type yes to continue: ', end='', file=sys.stderr, flush=True)
    try:
        answer = input()
    except EOFError:
        raise ValueError('AWS launch requires confirmation; preview first, then use --yes for automation') from None
    if answer.strip().lower() != 'yes':
        raise ValueError('AWS launch cancelled; no instance launched')


def cloud_job(path, aws_config, execute=False, command='pipeline', yes=False):
    """Use the same recipe with a budgeted, temporary EC2 lifecycle."""
    from modelchef.aws_runner import SpotRunner
    from modelchef.aws_training import build_bundle

    raw = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(raw, dict):
        raise TypeError('Recipe must contain a mapping')
    if command == 'pipeline':
        from modelchef.run_config import RunConfig
        RunConfig.model_validate(raw).validate_execution()
    elif 'stages' not in raw:
        raise ValueError('cook requires a workflow recipe with stages')
    settings = {**_settings(aws_config, 'aws_test'), 'pipeline_config': str(path.resolve())}
    # Validate and bound all declared inputs before AWS client construction or requests.
    build_bundle(Path.cwd(), settings)
    runner = SpotRunner(settings, Path.cwd())
    plan = runner.plan()
    if execute:
        confirm_cloud(plan, settings, yes)
    result = runner.run() if execute else plan
    result.pop('user_data', None)
    return {**result, 'target': 'aws', 'command': command, 'recipe': str(path),
            'lifecycle': 'One temporary EC2 instance; termination after completion or deadline.'}


def _main() -> None:
    # Randomly initialized models can generate Unicode characters unsupported by
    # the active Windows console code page. Preserve the run instead of crashing
    # while printing an informational sample.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(errors='backslashreplace')
    parser = argparse.ArgumentParser(prog="modelchef")
    command = parser.add_subparsers(dest="command", required=True)
    neuron_parser = command.add_parser('neuron', help='Plan/run isolated NeuronX training or compilation')
    neuron_parser.add_argument('path', type=Path)
    neuron_parser.add_argument('--action', choices=['train', 'compile'], default='train')
    neuron_parser.add_argument('--run', action='store_true')
    neuron_parser.add_argument('--checkpoint', type=Path)
    neuron_parser.add_argument('--resume', type=Path)
    workflow_parser = command.add_parser('cook', help='Plan/run ordered stage recipes locally or on AWS')
    workflow_parser.add_argument('path', type=Path)
    workflow_parser.add_argument('--run', action='store_true')
    multimodal = command.add_parser('multimodal', help='Plan/train local vision-language or speech recipes')
    multimodal.add_argument('path', type=Path)
    multimodal.add_argument('--run', action='store_true')
    multimodal.add_argument('--resume', type=Path)
    publish_parser = command.add_parser('publish-model', help='Preview/publish an allowlisted export to Hugging Face')
    publish_parser.add_argument('path', type=Path, help='Publication recipe')
    publish_parser.add_argument('--run', action='store_true')
    media_predict = command.add_parser('predict-media', help='Generate text from a local image or PCM WAV')
    media_predict.add_argument('path', type=Path, help='Multimodal recipe')
    media_predict.add_argument('--media', required=True, type=Path)
    media_predict.add_argument('--prompt', default='Describe this image.')
    media_predict.add_argument('--model-path', type=Path, help='Override with a trained export directory')
    command.add_parser("doctor", help="Inspect Python, PyTorch and CUDA")
    config_parser = command.add_parser("check-config", help="Validate a runtime config, training recipe or workflow")
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
    aws_train = command.add_parser('aws-train', help='Plan/run budgeted Spot pipeline training')
    aws_train.add_argument('path', type=Path)
    aws_train.add_argument('--instance-type')
    aws_train.add_argument('--run', action='store_true')
    hosting = command.add_parser('aws-serve', help='Plan/deploy persistent on-demand API hosting')
    hosting.add_argument('path', type=Path)
    hosting.add_argument('--action', choices=['deploy', 'status', 'logs', 'stop', 'start', 'delete'], default='deploy')
    hosting.add_argument('--deployment')
    hosting.add_argument('--run', action='store_true')
    aws_reconcile = command.add_parser("aws-reconcile", help="Recover reports/cleanup for a saved run")
    aws_reconcile.add_argument("run_id")
    aws_reconcile.add_argument("--path", type=Path, default=Path("configs/aws-test.yaml"))
    pipeline = command.add_parser("pipeline", help="Plan/run a training stage locally or on AWS")
    pipeline.add_argument("path", type=Path)
    pipeline.add_argument("--run", action="store_true")
    pipeline.add_argument("--resume", type=Path)
    for targeted in [pipeline, workflow_parser]:
        targeted.add_argument('--target', choices=['local', 'aws'], default='local',
                              help='Execution location; local is the default')
        targeted.add_argument('--aws-config', type=Path,
                              help='AWS resource/budget configuration required for --target aws')
        targeted.add_argument('--yes', action='store_true',
                              help='Explicitly approve AWS launch without an interactive prompt; budgets still apply')
    for cloud_parser in [aws_test, aws_train]:
        cloud_parser.add_argument('--yes', action='store_true',
                                  help='Approve the reviewed AWS launch without an interactive prompt')
    dashboard_parser = command.add_parser('dashboard', help='View live saved run metrics in TensorBoard')
    dashboard_parser.add_argument('run_dir', type=Path)
    dashboard_parser.add_argument('--port', type=int, default=6006)
    dashboard_parser.add_argument('--once', action='store_true', help='Export current metrics without starting a server')
    dashboard_parser.add_argument('--no-browser', action='store_true')
    chat_parser = command.add_parser("chat", help="Interact with a trained run locally")
    chat_parser.add_argument("run_dir", type=Path)
    chat_parser.add_argument("--backend", choices=["torch", "vllm"], default="torch")
    chat_parser.add_argument("--prompt")
    chat_parser.add_argument('--checkpoint', type=Path,
                             help='Saved checkpoint filename for torch inference, including unfinished runs')
    chat_parser.add_argument("--max-tokens", type=int, default=32)
    chat_parser.add_argument("--temperature", type=float, default=0.8)
    chat_parser.add_argument("--top-k", type=int, default=30)
    chat_parser.add_argument('--system-prompt', help='System instructions/persona for chat_transcript checkpoints')
    remote_chat = command.add_parser('chat-api', help='Chat with an authenticated HTTPS vLLM deployment')
    remote_chat.add_argument('--endpoint', required=True, help='HTTPS API base ending in /v1')
    remote_chat.add_argument('--model', default='auto', help='Model ID; auto selects a sole served model')
    remote_chat.add_argument('--api-key-env', default='MODELCHEF_API_KEY')
    remote_chat.add_argument('--ca-cert', type=Path, help='Trusted CA for a private certificate')
    remote_chat.add_argument('--prompt')
    remote_chat.add_argument('--max-tokens', type=int, default=64)
    remote_chat.add_argument('--temperature', type=float, default=0.7)
    aws_export = command.add_parser('aws-export', help='Recover a serving export from a completed AWS run')
    aws_export.add_argument('run_id')
    curation = command.add_parser('curate', help='Plan/acquire bounded dataset mixtures with attribution')
    curation.add_argument('path', type=Path)
    curation.add_argument('--run', action='store_true')
    export_parser = command.add_parser("export", help="Export a trained run for HF/vLLM")
    export_parser.add_argument("run_dir", type=Path)
    download = command.add_parser('download-model', help='Preview/download capped safetensors model')
    download.add_argument('repo')
    download.add_argument('--revision', default='main')
    download.add_argument('--max-bytes', type=int, default=500_000_000)
    download.add_argument('--run', action='store_true')
    imported = command.add_parser('import-model', help='Register a local pretrained HF model')
    imported.add_argument('path', type=Path)
    imported.add_argument('--context', type=int, default=128)
    student = command.add_parser('init-student', help='Initialize a native student with teacher vocabulary')
    student.add_argument('teacher', type=Path)
    student.add_argument('--architecture', type=Path)
    stage = command.add_parser('stage-config', help='Create a post-training recipe from a parent run')
    stage.add_argument('parent', type=Path)
    stage.add_argument('--kind', required=True, choices=['midtrain', 'sft', 'dpo', 'rlvr', 'distill'])
    stage.add_argument('--data', required=True)
    stage.add_argument('--output', required=True, type=Path)
    stage.add_argument('--steps', type=int, default=40)
    stage.add_argument('--teacher', type=Path)
    stage.add_argument('--reasoning', choices=['plain', 'think_answer'], default='plain')
    custom = command.add_parser('check-cuda-kernel', help='Check custom SwiGLU forward/backward on CUDA')
    custom.add_argument('path', type=Path)
    conversion = command.add_parser('convert-data', help='Convert bounded text/CSV/JSONL to JSONL')
    conversion.add_argument('source', type=Path)
    conversion.add_argument('--output', required=True, type=Path)
    conversion.add_argument('--format', choices=['text', 'csv', 'jsonl'], default='jsonl')
    conversion.add_argument('--max-bytes', type=int, default=10_000_000)
    conversion.add_argument('--max-documents', type=int, default=1000)
    teacher = command.add_parser('teacher-data', help='Generate bounded local teacher responses')
    teacher.add_argument('teacher', type=Path)
    teacher.add_argument('--prompts', required=True, type=Path)
    teacher.add_argument('--output', required=True, type=Path)
    teacher.add_argument('--max-documents', type=int, default=32)
    teacher.add_argument('--max-tokens', type=int, default=32)
    comparisons = command.add_parser('compare', help='Score local baselines on saved benchmark samples')
    comparisons.add_argument('run_dir', type=Path)
    comparisons.add_argument('--baseline', action='append', default=[], type=Path)
    comparisons.add_argument('--output', type=Path, default=Path('artifacts/comparison.json'))
    standard = command.add_parser('evaluate-standard', help='Plan/run optional lm-eval harness')
    standard.add_argument('model_path', type=Path)
    standard.add_argument('--tasks', nargs='+')
    standard.add_argument('--limit', type=int, default=100)
    standard.add_argument('--output', default='artifacts/standard-evaluation')
    standard.add_argument('--run', action='store_true')
    standard.add_argument('--device', choices=['auto', 'cpu', 'cuda', 'rocm', 'mps'], default='auto')
    parser.add_argument("--debug", action="store_true", help="Show tracebacks for troubleshooting")
    parser.add_argument("--quiet", action="store_true", help="Hide CLI progress on stderr")
    for subparser in command.choices.values():
        subparser.add_argument("--debug", action="store_true", default=argparse.SUPPRESS,
                               help="Show tracebacks for troubleshooting")
        subparser.add_argument("--quiet", action="store_true", default=argparse.SUPPRESS,
                               help="Hide CLI progress on stderr")
    args = parser.parse_args()
    with operation(args.command, args.quiet):
        if args.command == 'neuron':
            from modelchef.neuron import NeuronRecipe, compile_model, plan, train
            config = NeuronRecipe.model_validate(yaml.safe_load(args.path.read_text(encoding='utf-8')))
            if args.action == 'compile' and args.resume:
                raise ValueError('--resume applies only to Neuron training')
            if args.action == 'train' and args.checkpoint:
                raise ValueError('--checkpoint applies only to Neuron compilation')
            if args.run and args.action == 'compile' and not args.checkpoint:
                raise ValueError('Neuron compilation requires --checkpoint')
            result = (train(config, args.resume) if args.action == 'train' else
                      compile_model(config, args.checkpoint)) if args.run else plan(config, args.action)
            print(json.dumps(result, indent=2))
        elif args.command in {'pipeline', 'cook'} and args.target == 'aws':
            if not args.aws_config:
                raise ValueError('--target aws requires --aws-config')
            if getattr(args, 'resume', None):
                raise ValueError('Cloud --resume is not supported; recover the saved results and resume locally')
            result = cloud_job(args.path, args.aws_config, args.run, args.command, args.yes)
            print(json.dumps(result, indent=2, default=str))
            if args.run and not result.get('test_passed'):
                parser.exit(1, 'Cloud job success was not confirmed; inspect the run report and termination status.\n')
        elif args.command in {'pipeline', 'cook'} and args.aws_config:
            raise ValueError('--aws-config requires --target aws')
        elif args.command == 'publish-model':
            from modelchef.publishing import PublishConfig, publish
            config = PublishConfig.model_validate(yaml.safe_load(args.path.read_text(encoding='utf-8')))
            print(json.dumps(publish(config, Path.cwd(), args.run), indent=2))
        elif args.command in {'multimodal', 'predict-media'}:
            from modelchef.multimodal import plan, predict, train
            from modelchef.multimodal_config import MultimodalConfig
            config = MultimodalConfig.model_validate(yaml.safe_load(args.path.read_text(encoding='utf-8')))
            if args.command == 'predict-media':
                print(predict(config, Path.cwd(), args.media, args.prompt, args.model_path))
            elif args.run:
                print(json.dumps(train(config, Path.cwd(), args.resume), indent=2))
            else:
                if args.resume:
                    raise ValueError('--resume requires --run')
                print(json.dumps(plan(config, Path.cwd()), indent=2))
        elif args.command == 'dashboard':
            from modelchef.dashboard import dashboard
            dashboard(args.run_dir, args.port, args.once, not args.no_browser)
        elif args.command == 'cook':
            from modelchef.workflow import cook
            print(json.dumps(cook(args.path, Path.cwd(), args.run), indent=2, default=str))
        elif args.command == 'aws-serve':
            from modelchef.aws_serving import ServingRunner
            settings = _settings(args.path, 'aws_serving')
            runner = ServingRunner(settings, Path.cwd())
            if args.action == 'deploy':
                result = runner.deploy() if args.run else runner.plan()
            elif not args.deployment:
                parser.error('--deployment is required for hosting lifecycle commands')
            elif args.action == 'status' or args.run:
                result = runner.action(args.deployment, args.action)
            else:
                result = {'planned': args.action, 'deployment': args.deployment,
                          'note': 'Add --run to execute this hosting lifecycle command'}
            print(json.dumps(result, indent=2, default=str))
        elif args.command == 'convert-data':
            from modelchef.data_tools import convert_data
            print(json.dumps(convert_data(args.source, args.output, args.format, args.max_bytes,
                                          args.max_documents), indent=2))
        elif args.command == 'teacher-data':
            from modelchef.data_tools import teacher_responses
            print(json.dumps(teacher_responses(args.teacher, args.prompts, args.output,
                                              args.max_documents, max_tokens=args.max_tokens), indent=2))
        elif args.command == 'compare':
            from modelchef.comparison import compare
            print(json.dumps(compare(args.run_dir, args.baseline, args.output), indent=2))
        elif args.command == 'evaluate-standard':
            from modelchef.comparison import standard_plan
            print(json.dumps(standard_plan(args.model_path, args.tasks, args.limit,
                                           args.output, args.run, args.device), indent=2))
        elif args.command == 'download-model':
            from modelchef.pretrained import download_model
            print(json.dumps(download_model(args.repo, Path.cwd(), args.revision,
                                            args.max_bytes, args.run), indent=2))
        elif args.command == 'import-model':
            from modelchef.pretrained import register_model
            print(register_model(args.path, Path.cwd(), args.context))
        elif args.command == 'init-student':
            from modelchef.pretrained import initialize_student
            architecture = yaml.safe_load(args.architecture.read_text()) if args.architecture else None
            print(initialize_student(args.teacher, Path.cwd(), architecture))
        elif args.command == 'stage-config':
            from modelchef.pretrained import stage_config
            print(stage_config(args.parent, args.kind, args.data, args.output, args.steps, args.teacher,
                               args.reasoning))
        elif args.command == 'check-cuda-kernel':
            from modelchef.execution import check_custom_kernel
            print(json.dumps(check_custom_kernel(args.path), indent=2))
        elif args.command == "chat":
            from modelchef.chat import chat
            chat(args.run_dir, args.backend, args.prompt, args.max_tokens, args.temperature, args.top_k,
                 system_prompt=args.system_prompt, checkpoint=args.checkpoint)
        elif args.command == 'chat-api':
            from modelchef.remote_chat import chat
            chat(args.endpoint, args.model, api_key_env=args.api_key_env,
                 prompt=args.prompt, max_tokens=args.max_tokens,
                 temperature=args.temperature, ca_cert=args.ca_cert)
        elif args.command == 'aws-export':
            from modelchef.aws_export import recover_export
            print(json.dumps(recover_export(Path.cwd(), args.run_id), indent=2))
        elif args.command == 'curate':
            from modelchef.curation import curate
            from modelchef.progress import report
            print(json.dumps(curate(args.path, Path.cwd(), args.run, progress=report), indent=2))
        elif args.command == "export":
            from modelchef.chat import export_model
            print(export_model(args.run_dir))
        elif args.command == "pipeline":
            from modelchef.run_config import RunConfig
            with args.path.open(encoding="utf-8") as stream:
                config = RunConfig.model_validate(yaml.safe_load(stream)).validate_execution()
            if args.run:
                from modelchef.pipeline import run_pipeline
                result = run_pipeline(config, Path.cwd(), args.resume)
                if result.get('status') != 'completed':
                    parser.exit(1, f"Training ended with status {result.get('status')}. "
                                f"Inspect {result.get('run_dir')}/report.json before resuming.\n")
            else:
                print("Validated plan only: no dataset download, training or AWS launch. "
                      "Add --run to execute on this machine.")
                print(json.dumps(config.model_dump(mode="json"), indent=2))
        elif args.command in {"aws-test", "aws-train", "aws-reconcile"}:
            from modelchef.aws_runner import SpotRunner
            settings = _settings(args.path, "aws_test")
            if args.command == 'aws-train' and not settings.get('pipeline_config'):
                raise ValueError('aws-train requires pipeline_config in the AWS configuration')
            if getattr(args, "instance_type", None):
                settings["instance_type"] = args.instance_type
            runner = SpotRunner(settings, Path.cwd())
            if args.command == "aws-reconcile":
                result = runner.reconcile(args.run_id)
            elif args.run:
                confirm_cloud(runner.plan(), settings, args.yes)
                result = runner.run()
            else:
                result = runner.plan()
                result.pop("user_data")
            print(json.dumps(result, indent=2, default=str))
            if getattr(args, "run", False) and not result.get("test_passed"):
                parser.exit(1, "EC2 instance terminated; test success not confirmed in console output\n")
        elif args.command in {"setup-vllm", "serve-vllm"}:
            from modelchef.inference import launch_server, serve_plan, vllm_setup_plan
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
        elif args.command == "doctor":
            print(json.dumps(check_environment(), indent=2))
        elif args.command == "setup":
            plan = setup_plan(args.backend, args.index_url, args.torch_version)
            print(json.dumps(plan, indent=2))
            if args.apply:
                if not plan["python_supported"]:
                    parser.error("Create a Python 3.11, 3.12, or 3.13 environment first")
                if sys.prefix == sys.base_prefix:
                    parser.error("Activate a project virtual environment before --apply")
                if not shutil.which("uv"):
                    parser.error("Install uv and add it to PATH first")
                subprocess.run(plan["command"], check=True)
        elif args.command == "smoke":
            from modelchef.smoke import run_smoke
            print(json.dumps(run_smoke(args.accelerator, args.precision), indent=2))
        elif args.command in {"check-config", "resolve-runtime"}:
            with args.path.open(encoding="utf-8") as f:
                config = yaml.safe_load(f)
            if not isinstance(config, dict):
                raise ValueError("Configuration must be a YAML mapping")
            if args.command == "check-config" and "stages" in config:
                from modelchef.workflow import cook
                config = cook(args.path, Path.cwd())
            elif args.command == 'check-config' and 'export_dir' in config:
                from modelchef.publishing import PublishConfig
                config = PublishConfig.model_validate(config).model_dump(mode='json')
            elif args.command == 'check-config' and 'modality' in config:
                from modelchef.multimodal_config import MultimodalConfig
                config = MultimodalConfig.model_validate(config).model_dump(mode='json')
            elif args.command == 'check-config' and 'sources' in config:
                from modelchef.curation import CurationConfig
                config = CurationConfig.model_validate(config).model_dump(mode='json')
            elif args.command == "check-config" and "runtime" not in config:
                from modelchef.run_config import RunConfig
                config = RunConfig.model_validate(config).validate_execution().model_dump(mode="json")
            elif not isinstance(config.get("project"), dict) or not isinstance(config.get("runtime"), dict):
                raise ValueError("Runtime config requires 'project' and 'runtime' mappings")
            if args.command == "resolve-runtime":
                config["runtime"] = resolve_runtime(config["runtime"], detect_hardware())
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            print(json.dumps(config, indent=2))


def main() -> None:
    """Keep routine input/environment errors concise; --debug retains diagnostic tracebacks."""
    try:
        _main()
    except KeyboardInterrupt:
        print("ModelChef interrupted. Check the saved run report before resuming.", file=sys.stderr)
        raise SystemExit(130) from None
    except (ValueError, TypeError, OSError, ImportError, RuntimeError, yaml.YAMLError, subprocess.SubprocessError) as error:
        if "--debug" in sys.argv[1:]:
            raise
        message = str(error)
        if isinstance(error, ModuleNotFoundError):
            package = (error.name or "").split(".")[0]
            extras = {"pydantic": "pipeline", "tokenizers": "pipeline", "safetensors": "pipeline",
                      "transformers": "pretrained", "huggingface_hub": "pretrained", "peft": "peft",
                      "bitsandbytes": "qlora", "transformer_engine": "nvfp4", "boto3": "aws",
                      "botocore": "aws", "vllm": "inference", "lm_eval": "evaluation"}
            extras.update(PIL='multimodal', numpy='train', tensorboard='dashboard')
            if package == "torch":
                message += ". Run modelchef setup, then modelchef setup --apply."
            elif package in extras:
                message += f". Install the extra: uv pip install -e '.[{extras[package]}]'."
        print(f"ModelChef error: {message}\nUse --debug to show the traceback.", file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
