"""Package a complete text workflow for the existing bounded Spot runner."""
import io
import json
import subprocess
import sys
import tarfile
import uuid
from pathlib import Path

import yaml

from modelchef.run_config import RunConfig
from modelchef.workflow import Workflow, cook


def _relocate(value, prefix):
    if isinstance(value, dict):
        paths = {'local_path', 'local_paths', 'pretrained_path', 'custom_swiglu_path',
                 'parent_run', 'teacher_run', 'benchmark_files'}
        return {key: _relocate(item, prefix) if key in paths or isinstance(item, dict) else item
                for key, item in value.items()}
    if isinstance(value, list):
        return [_relocate(item, prefix) for item in value]
    if isinstance(value, str) and value.startswith('inputs/'):
        return prefix + value[len('inputs/'):]
    return value


def build_workflow_bundle(root, settings):
    from modelchef.aws_training import build_bundle

    root = Path(root).resolve()
    path = root / settings['pipeline_config']
    workflow = Workflow.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    if workflow.curation_recipe:
        from modelchef.curation import CurationConfig
        curation = CurationConfig.model_validate(yaml.safe_load((root / workflow.curation_recipe).read_text()))
        collections = {source.collection for source in curation.sources} | {
            source.pretraining_collection for source in curation.sources if source.pretraining_collection}
        settings = {**settings, '_prepared_inputs': {
            str((root / curation.output_dir / (name + '.jsonl')).resolve()) for name in collections}}
    plan = cook(path, root)
    files = {}
    cap = int(settings.get('max_bundle_bytes', 64_000_000))

    def merge(bundle, prefix):
        with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
            for entry in archive.getmembers():
                content = archive.extractfile(entry).read()
                name = entry.name
                if name == 'configs/job.yaml':
                    content = yaml.safe_dump(_relocate(yaml.safe_load(content), prefix), sort_keys=False).encode()
                    name = prefix + 'recipe.yaml'
                elif name.startswith('inputs/'):
                    name = prefix + name[len('inputs/'):]
                    if name.endswith('/config.json'):
                        content = json.dumps(_relocate(json.loads(content), prefix)).encode()
                if name in files and files[name] != content:
                    raise ValueError('Conflicting workflow bundle files')
                files[name] = content
                if sum(map(len, files.values())) > cap:
                    raise ValueError('Uncompressed workflow bundle exceeds max_bundle_bytes')

    recipe = workflow.model_dump(mode='json')
    recipe['output_dir'] = 'artifacts/workflows'
    if workflow.curation_recipe:
        files['configs/curation.yaml'] = (root / workflow.curation_recipe).read_bytes()
        recipe['curation_recipe'] = 'configs/curation.yaml'
    for stage in recipe['stages']:
        raw = plan['resolved_recipes'][stage['name']]
        prefix = 'inputs/stages/' + stage['name'] + '/'
        merge(build_bundle(root, settings, raw_override=raw, linked_stage=True), prefix)
        stage['recipe'] = prefix + 'recipe.yaml'
    if workflow.imported_model:
        from modelchef.pretrained import inspect_model
        config, _, _ = inspect_model(root / workflow.imported_model, workflow.imported_context)
        # Only model/tokenizer assets are needed to register the imported parent on the guest.
        merge(build_bundle(root, settings, raw_override=config.model_dump(mode='json')), 'inputs/imported/')
        recipe['imported_model'] = 'inputs/imported/pretrained-0'
    files['configs/job.yaml'] = yaml.safe_dump(recipe, sort_keys=False).encode()
    if sum(map(len, files.values())) > cap:
        raise ValueError('Uncompressed workflow bundle exceeds max_bundle_bytes')
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode='w:gz') as archive:
        for name, content in sorted(files.items()):
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return output.getvalue()


def workflow_preflight(root, settings):
    """Exercise stage links and final export with two-step CPU recipes, before launch."""
    root = Path(root).resolve()
    path = root / settings['pipeline_config']
    workflow = Workflow.model_validate(yaml.safe_load(path.read_text(encoding='utf-8')))
    plan = cook(path, root)
    directory = root / 'artifacts/aws-workflow-preflight' / uuid.uuid4().hex
    directory.mkdir(parents=True)
    recipe = workflow.model_dump(mode='json')
    recipe['output_dir'] = str(directory / 'workflows')
    recipe['curation_recipe'] = None
    for stage in recipe['stages']:
        config = RunConfig.model_validate(plan['resolved_recipes'][stage['name']])
        config.output_dir = str(directory / 'training')
        if workflow.curation_recipe:
            sample_dir = settings.get('preflight_data_dir')
            if not sample_dir:
                raise ValueError('Remote curation requires preflight_data_dir containing representative local samples')
            config.data.local_paths = [str(root / sample_dir / Path(name).name)
                                      for name in config.data.local_paths or [config.data.local_path]]
            config.data.local_path = None
        config.data.max_documents = min(config.data.max_documents, 256)
        config.data.max_tokens = min(config.data.max_tokens, 262144)
        config.processing.validation_fraction = config.processing.test_fraction = 0.1
        config.training.accelerator, config.training.precision = 'cpu', 'fp32'
        config.training.compile, config.training.allow_tf32 = False, False
        config.training.optimizer_kernel = 'default'
        config.model.attention_kernel, config.model.custom_swiglu_path = 'math', None
        if config.model.quantization == 'nvfp4':
            config.model.quantization = 'none'
        if config.model.peft.method == 'qlora':
            config.model.peft.method = 'lora'
        config.training.steps, config.training.warmup_steps = 2, 0
        config.training.performance_gate = None
        config.training.micro_batch_size = config.training.gradient_accumulation = 1
        config.training.max_runtime_seconds = 60
        config.evaluation.benchmark, config.evaluation.batches = 'none', 1
        config.evaluation.generate_tokens = 1
        if config.stage.kind == 'pretrain':
            config.model.layers, config.model.hidden_size = 1, 64
            config.model.attention_heads, config.model.kv_heads = 4, 2
            config.model.intermediate_size = 128
        stage_path = directory / (stage['name'] + '.yaml')
        stage_path.write_text(yaml.safe_dump(config.model_dump(mode='json')), encoding='utf-8')
        stage['recipe'] = str(stage_path)
    probe = directory / 'workflow.yaml'
    probe.write_text(yaml.safe_dump(recipe), encoding='utf-8')
    subprocess.run([sys.executable, '-m', 'modelchef.cli', 'cook', str(probe), '--run'],
                   cwd=root, check=True, timeout=300 * len(workflow.stages))
    reports = list((directory / 'workflows').glob('*/report.json'))
    if len(reports) != 1 or json.loads(reports[0].read_text())['status'] != 'completed':
        raise RuntimeError('Workflow preflight did not complete; no instance launched')
    return {'passed': True, 'recipe': str(probe), 'steps_per_stage': 2,
            'workflow_dir': str(reports[0].parent),
            'scope': 'CPU stage-link/data/objective/export probe; GPU policy validated on guest'}
