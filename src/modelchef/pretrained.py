"""Explicit, capped pretrained acquisition and registration for the common pipeline."""
import json
import uuid
from pathlib import Path

from modelchef.corpus import digest, file_digest
from modelchef.progress import report as progress


def download_model(repo, root, revision='main', max_bytes=500_000_000, run=False):
    from huggingface_hub import HfApi, snapshot_download
    if max_bytes <= 0:
        raise ValueError('Download cap must be positive')
    progress('Requesting model revision and file sizes from Hugging Face; no weights downloaded yet.')
    info = HfApi().model_info(repo, revision=revision, files_metadata=True)
    files = [f for f in info.siblings if f.rfilename.endswith(('.json', '.safetensors', '.model', '.jinja'))
             or f.rfilename in {'merges.txt', 'vocab.txt'}]
    if any(f.size is None for f in files):
        raise ValueError('Cannot enforce download budget without file sizes')
    size = sum(f.size for f in files)
    if not any(f.rfilename.endswith('.safetensors') for f in files):
        raise ValueError('Only safetensors pretrained weights are supported')
    if size > max_bytes:
        raise ValueError(f'Model files require {size:,} bytes; cap is {max_bytes:,}')
    plan = {'repo': repo, 'revision': info.sha, 'bytes': size,
            'files': [f.rfilename for f in files], 'downloaded': False}
    if run:
        progress(f'Downloading {len(files)} approved model files ({size:,} bytes).')
        output = Path(root) / 'artifacts/pretrained' / (repo.replace('/', '--') + '-' + info.sha[:12])
        snapshot_download(repo, revision=info.sha, allow_patterns=plan['files'], local_dir=output)
        progress('Verifying downloaded file sizes and recording SHA256 checksums.')
        actual = sum((output / name).stat().st_size for name in plan['files'])
        if actual > max_bytes:
            raise ValueError('Downloaded bytes exceed cap')
        plan.update(downloaded=True, path=str(output.resolve()),
                    sha256={name: file_digest(output / name) for name in plan['files']})
        (output / 'acquisition-manifest.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    return plan


def inspect_model(path, context=128):
    """Validate local HF metadata without loading weights or creating run artifacts."""
    from transformers import AutoConfig, AutoTokenizer

    from modelchef.run_config import RunConfig

    path = Path(path).resolve()
    progress('Reading local model configuration and tokenizer; remote model code is disabled.')
    hf = AutoConfig.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError('A fast tokenizer is required; no tokenizer conversion is guessed')
    if tokenizer.eos_token_id is None:
        raise ValueError('Pretrained tokenizer requires an EOS token')
    width = getattr(hf, 'hidden_size', getattr(hf, 'n_embd', 0))
    heads = getattr(hf, 'num_attention_heads', getattr(hf, 'n_head', 0))
    layers = getattr(hf, 'num_hidden_layers', getattr(hf, 'n_layer', 0))
    max_context = getattr(hf, 'max_position_embeddings', getattr(hf, 'n_positions', context))
    if context < 8 or context > max_context:
        raise ValueError(f'Context must be between 8 and {max_context}')
    config = RunConfig(name='pretrained', tokenizer={'algorithm': 'pretrained',
        'vocab_size': len(tokenizer)}, model={'backend': 'huggingface',
        'pretrained_path': str(path), 'pretrained_config_sha256': digest((path / 'config.json').read_bytes()),
        'hidden_size': width, 'attention_heads': heads,
        'kv_heads': getattr(hf, 'num_key_value_heads', heads), 'layers': layers,
        'intermediate_size': getattr(hf, 'intermediate_size', width * 4) or width * 4,
        'context_length': context}, evaluation={'benchmark': 'none'})
    roles = {'<eos>': tokenizer.eos_token_id,
             '<bos>': tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
             '<pad>': tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
             '<unk>': tokenizer.unk_token_id if tokenizer.unk_token_id is not None else tokenizer.eos_token_id}
    if not any(path.glob('*.safetensors')) and not any(path.glob('*.bin')):
        raise ValueError('Pretrained model directory requires local weight files')
    for index_path in path.glob('*.index.json'):
        index = json.loads(index_path.read_text(encoding='utf-8'))
        for name in set(index.get('weight_map', {}).values()):
            shard = (path / name).resolve()
            if not shard.is_relative_to(path) or not shard.is_file():
                raise ValueError('Pretrained model index has a missing or invalid weight shard')
    return config, tokenizer, roles


def register_model(path, root, context=128):
    import torch

    from modelchef.engines import create_model

    path = Path(path).resolve()
    config, tokenizer, roles = inspect_model(path, context)
    folder = Path(root) / 'artifacts/training' / ('pretrained-' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    tokenizer.backend_tokenizer.save(str(folder / 'tokenizer.json'))
    (folder / 'tokenizer-roles.json').write_text(json.dumps(roles), encoding='utf-8')
    (folder / 'config.json').write_text(config.model_dump_json(indent=2), encoding='utf-8')
    progress('Loading pretrained weights into memory for the imported parent checkpoint.')
    model = create_model(config.model, tokenizer.backend_tokenizer.get_vocab_size())
    checkpoint = folder / 'checkpoints/step-000000.pt'
    checkpoint.parent.mkdir()
    progress('Saving the imported parent checkpoint and source metadata; no training is performed.')
    torch.save({'model': model.state_dict()}, checkpoint)
    report = {'status': 'completed', 'stage': 'import', 'optimizer_steps': 0,
              'checkpoint': str(checkpoint.resolve()), 'source': str(path),
              'parameters': sum(model_file.numel() for model_file in model.parameters()),
              'source_config_sha256': digest((path / 'config.json').read_bytes())}
    (folder / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return str(folder)


def initialize_student(teacher, root, architecture=None):
    import torch

    from modelchef.engines import create_model, load_tokenizer
    from modelchef.run_config import Architecture, RunConfig

    teacher = Path(teacher).resolve()
    parent = RunConfig.model_validate_json((teacher / 'config.json').read_text())
    model_config = Architecture.model_validate(architecture or {'layers': 4, 'hidden_size': 128,
        'attention_heads': 4, 'kv_heads': 2, 'intermediate_size': 384,
        'context_length': parent.model.context_length})
    if model_config.backend != 'native':
        raise ValueError('Student initialization creates a random native decoder')
    config = RunConfig(name='student', tokenizer=parent.tokenizer, model=model_config,
                       evaluation={'benchmark': 'none'})
    folder = Path(root) / 'artifacts/training' / ('student-' + uuid.uuid4().hex[:12])
    folder.mkdir(parents=True)
    tokenizer = load_tokenizer(teacher)
    tokenizer.save(str(folder / 'tokenizer.json'))
    (folder / 'tokenizer-roles.json').write_text(json.dumps(tokenizer.roles), encoding='utf-8')
    (folder / 'config.json').write_text(config.model_dump_json(indent=2), encoding='utf-8')
    torch.manual_seed(config.training.seed)
    model = create_model(config.model, tokenizer.get_vocab_size())
    path = folder / 'checkpoints/step-000000.pt'
    path.parent.mkdir()
    torch.save({'model': model.state_dict()}, path)
    (folder / 'report.json').write_text(json.dumps({'status': 'completed', 'stage': 'initialize',
        'optimizer_steps': 0, 'parameters': sum(parameter.numel() for parameter in model.parameters()),
        'checkpoint': str(path), 'tokenizer_sha256': digest((folder / 'tokenizer.json').read_bytes()),
        'note': 'Random student weights; no optimization yet.'}), encoding='utf-8')
    return str(folder)


def stage_config(parent, kind, data, output, steps=40, teacher=None, reasoning='plain'):
    import yaml

    from modelchef.run_config import RunConfig
    parent = Path(parent).resolve()
    config = RunConfig.model_validate_json((parent / 'config.json').read_text())
    recipe = config.model_dump(mode='json')
    recipe.update(name=kind)
    recipe['stage'].update(kind=kind, parent_run=str(parent))
    recipe['stage']['teacher_run'] = str(Path(teacher).resolve()) if teacher else None
    recipe['stage']['reasoning_format'] = reasoning
    recipe['data'].update(provider='local_jsonl', local_path=data, local_paths=[], domain='general',
                          domain_keywords=[], offset=0)
    if kind != 'midtrain':
        recipe['data']['storage'] = 'memory'
    recipe['processing'].update(min_chars=1, near_dedup=False, max_repeated_line_fraction=0.99)
    recipe['training'].update(steps=steps, warmup_steps=min(4, steps - 1), learning_rate=0.00005)
    recipe['evaluation'].update(benchmark='none')
    validated = RunConfig.model_validate(recipe)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as handle:
        handle.write(yaml.safe_dump(validated.model_dump(mode='json'), sort_keys=False))
    return str(output)
