"""Offline publication previews and explicit, allowlisted Hugging Face model commits."""
import hashlib
import json
import re
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field

from modelchef.corpus import file_digest
from modelchef.run_config import Section


class PublishConfig(Section):
    schema_version: Literal[1] = 1
    export_dir: str
    repo_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$')
    private: bool = True
    license: str = Field(min_length=1, pattern=r'^[a-z0-9][a-z0-9_.-]*$')
    summary: str = Field(min_length=1)
    training_data: str = Field(min_length=1)
    data_attribution_path: str | None = None
    evaluation: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    base_model: str | None = None
    pipeline_tag: Literal['text-generation', 'image-text-to-text', 'automatic-speech-recognition'] = 'text-generation'
    max_upload_bytes: int = Field(default=8000000000, gt=0)
    max_text_file_bytes: int = Field(default=64000000, gt=0)
    commit_message: str = 'Publish ModelChef model export'


TEXT_FILES = {'config.json', 'adapter_config.json', 'generation_config.json', 'tokenizer.json',
              'tokenizer_config.json', 'special_tokens_map.json', 'added_tokens.json',
              'vocab.json', 'merges.txt', 'vocab.txt', 'preprocessor_config.json',
              'processor_config.json', 'chat_template.json', 'chat_template.jinja',
              'modelchef-quantization.json'}
BINARY_FILES = {'tokenizer.model', 'spiece.model'}
SECRET = re.compile(rb'hf_[A-Za-z0-9]{20,}|(?:AKIA|ASIA)[A-Z0-9]{16}|'
                    rb'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')


def model_card(config, attribution=''):
    metadata = {'license': config.license, 'library_name': 'transformers',
                'pipeline_tag': config.pipeline_tag, 'tags': ['modelchef']}
    if config.base_model:
        metadata['base_model'] = config.base_model
    return ('---\n' + yaml.safe_dump(metadata, sort_keys=False) + '---\n\n# ' +
        config.repo_id + '\n\n' + config.summary + '\n\n## Training data\n\n' +
        config.training_data + '\n\n## Evaluation\n\n' + config.evaluation +
        '\n\n## Limitations\n\n' + config.limitations + '\n' + attribution)


def attribution_section(config, root):
    """Render reviewed source metadata into the Hub card without publishing raw data."""
    if not config.data_attribution_path:
        return ''
    from modelchef.curation import Citation

    path = (Path(root) / config.data_attribution_path).resolve()
    if path.stat().st_size > config.max_text_file_bytes:
        raise ValueError('Attribution manifest exceeds max_text_file_bytes')
    content = path.read_bytes()
    if SECRET.search(content):
        raise ValueError('Possible credential in attribution manifest')
    manifest = json.loads(content)
    sources = manifest.get('sources', [])
    if not sources or not manifest.get('transformations'):
        raise ValueError('Attribution requires sources and transformations')
    lines = ['\n## Dataset attribution and licenses\n',
             'Attribution manifest SHA256: `' + hashlib.sha256(content).hexdigest() + '`.',
             '\n' + manifest['transformations']]
    for source in sources:
        required = ['name', 'dataset', 'dataset_url', 'license', 'license_url', 'attribution',
                    'snapshot_sha256', 'citations', 'accepted_examples', 'revision_note']
        if any(key not in source or source[key] in ('', None, []) for key in required):
            raise ValueError('Incomplete dataset attribution')
        if not re.fullmatch(r'[a-f0-9]{64}', source['snapshot_sha256']):
            raise ValueError('Dataset attribution needs a valid snapshot SHA256')
        if source['license'] == 'cc-by-nc-4.0' and config.license != 'cc-by-nc-4.0':
            raise ValueError('Noncommercial data recipe requires a cc-by-nc-4.0 research publication label')
        lines.append(f"\n### {source['dataset']} ({source['name']})\n")
        lines.append(f"Source: [{source['dataset']}]({source['dataset_url']}); "
                     f"license: [{source['license']}]({source['license_url']}).")
        lines.extend([source['attribution'],
                      (f"Prepared examples: {source['accepted_examples']}; "
                       f"snapshot SHA256: `{source['snapshot_sha256']}`."), source['revision_note']])
        for citation in source['citations']:
            citation = Citation.model_validate(citation)
            lines.append(', '.join(citation.authors) + '. [' + citation.title + '](' + citation.url + ').')
    lines.append('\nSource licenses retain their own terms; the model license label does not '
                 'relicense datasets. Prepared counts are not proof that every row was consumed. '
                 'Consult the training split/token manifests and publish the actual evaluation scope.\n')
    return '\n\n'.join(lines)


def publish(config, root, run=False):
    from modelchef.progress import report as progress

    progress('Checking allowlisted export files, sizes, checksums and model-card contents.')
    folder = (Path(root) / config.export_dir).resolve()
    if not folder.is_dir():
        raise ValueError('export_dir must be an existing model export directory')
    selected, total = [], 0
    for path in sorted(folder.iterdir()):
        name = path.name
        allowed = (name in TEXT_FILES | BINARY_FILES or
                   re.fullmatch(r'(?:model|adapter_model)(?:-\d+-of-\d+)?\.safetensors', name) or
                   name == 'model.safetensors.index.json')
        if not allowed:
            continue
        if path.is_symlink() or not path.is_file():
            raise ValueError(f'Publication refuses symlinks or non-files: {name}')
        size = path.stat().st_size
        total += size
        if total > config.max_upload_bytes:
            raise ValueError('Export exceeds max_upload_bytes')
        if name not in BINARY_FILES and path.suffix != '.safetensors':
            if size > config.max_text_file_bytes:
                raise ValueError(f'Text export file exceeds max_text_file_bytes: {name}')
            if SECRET.search(path.read_bytes()):
                raise ValueError(f'Possible credential in publication file: {name}; remove it before publishing')
        selected.append({'name': name, 'bytes': size, 'sha256': file_digest(path)})
    names = {item['name'] for item in selected}
    if not {'config.json', 'tokenizer_config.json'} <= names:
        raise ValueError('Publication requires model config and tokenizer_config.json')
    if not any(name.endswith('.safetensors') for name in names):
        raise ValueError('Publication requires exported safetensors weights')
    if config.pipeline_tag == 'automatic-speech-recognition' and not names & {'preprocessor_config.json', 'processor_config.json'}:
        raise ValueError('Speech publication requires processor metadata')
    if config.pipeline_tag == 'image-text-to-text' and not names & {'preprocessor_config.json', 'processor_config.json'}:
        raise ValueError('Vision publication requires processor metadata')
    if 'model.safetensors.index.json' in names:
        index = json.loads((folder / 'model.safetensors.index.json').read_text())
        if not set(index['weight_map'].values()) <= names:
            raise ValueError('Publication index references missing or excluded shards')
    card = model_card(config, attribution_section(config, root)).encode()
    if SECRET.search(card):
        raise ValueError('Possible credential in model card; remove it before publishing')
    total += len(card)
    if total > config.max_upload_bytes:
        raise ValueError('Export and model card exceed max_upload_bytes')
    result = {'repo_id': config.repo_id, 'private': config.private, 'license': config.license,
              'files': selected + [{'name': 'README.md', 'bytes': len(card),
                                   'sha256': hashlib.sha256(card).hexdigest()}],
              'total_bytes': total, 'model_card': card.decode(), 'uploaded': False,
              'note': 'Only listed model/processor files and generated card are published; no run logs or raw data.'}
    if not run:
        return result
    from huggingface_hub import CommitOperationAdd, HfApi, get_token
    token = get_token()
    if not token:
        raise ValueError('Hugging Face authentication missing; run hf auth login outside the repository')
    api = HfApi(token=token)
    progress('Checking Hugging Face repository access and requested visibility.')
    api.create_repo(repo_id=config.repo_id, repo_type='model', private=config.private, exist_ok=True)
    info = api.repo_info(repo_id=config.repo_id, repo_type='model')
    if info.private != config.private:
        raise ValueError('Existing Hub repository visibility differs from recipe; change it explicitly on the Hub')
    operations = []
    for item in selected:
        path = folder / item['name']
        if file_digest(path) != item['sha256']:
            raise ValueError('Export changed during publication preparation; retry with immutable export files')
        operations.append(CommitOperationAdd(path_in_repo=item['name'], path_or_fileobj=str(path)))
    operations.append(CommitOperationAdd(path_in_repo='README.md', path_or_fileobj=card))
    progress(f'Uploading {len(operations)} approved files and creating the model publication commit.')
    commit = api.create_commit(repo_id=config.repo_id, repo_type='model', operations=operations,
                               commit_message=config.commit_message)
    result.update(uploaded=True, commit_url=commit.commit_url, commit_sha=commit.oid)
    return result
