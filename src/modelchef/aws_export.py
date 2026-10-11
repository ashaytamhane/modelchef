"""Recover a verified model export from a completed AWS workflow archive."""

import hashlib
import io
import json
import tarfile
from pathlib import Path, PurePosixPath

ALLOWED = {'.json', '.safetensors', '.model', '.bin', '.txt', '.jinja'}


def recover_export(root, run_id):
    if not run_id.startswith('modelchef-') or not run_id[10:].isalnum():
        raise ValueError('Invalid ModelChef AWS run ID')
    root = Path(root).resolve()
    source = root / 'artifacts/aws-runs' / run_id
    report = json.loads((source / 'report.json').read_text(encoding='utf-8'))
    if report.get('run_id') != run_id or not report.get('test_passed'):
        raise ValueError('AWS run has not been confirmed successful')
    archive_path = source / 'results.tar.gz'
    content = archive_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != report.get('results_sha256'):
        raise ValueError('AWS result archive checksum does not match the run report')
    groups = {}
    extracted_bytes = 0
    with tarfile.open(fileobj=io.BytesIO(content), mode='r:gz') as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if (not member.isfile() or len(path.parts) != 5 or path.parts[0] != 'artifacts'
                    or path.parts[1] != 'training' or path.parts[3] != 'export'
                    or path.name in {'.', '..'} or path.suffix not in ALLOWED):
                continue
            if path.parts[2] in {'.', '..'} or '/' in path.parts[2]:
                continue
            extracted_bytes += member.size
            if extracted_bytes > 2_000_000_000:
                raise ValueError('AWS export exceeds the 2 GB extraction cap')
            entries = groups.setdefault(path.parts[2], {})
            if path.name in entries:
                raise ValueError('Duplicate model export member in AWS results')
            entries[path.name] = member
        if len(groups) != 1:
            raise ValueError('Expected exactly one workflow export in AWS results')
        entries = next(iter(groups.values()))
        if not {'config.json', 'tokenizer.json'} <= entries.keys() or not any(
                name.endswith('.safetensors') for name in entries):
            raise ValueError('AWS export lacks config, tokenizer, or safetensors weights')
        output = root / 'artifacts/exports' / run_id
        if output.exists():
            raise ValueError('Export already exists; choose a new completed run')
        output.mkdir(parents=True)
        for name, member in entries.items():
            with archive.extractfile(member) as stream:
                (output / name).write_bytes(stream.read())
    import yaml

    model_config = json.loads((output / 'config.json').read_text(encoding='utf-8'))
    context = int(model_config['max_position_embeddings'])
    serving_config = root / 'artifacts/aws-serving' / (run_id + '-inference.yaml')
    serving_config.parent.mkdir(parents=True, exist_ok=True)
    serving_config.write_text(yaml.safe_dump({'inference': {
        'model_path': str(output.relative_to(root)).replace('\\', '/'),
        'host': '0.0.0.0', 'port': 8443, 'dtype': 'auto', 'quantization': 'none',
        'gpu_memory_utilization': 0.5, 'max_model_len': context,
        'max_num_seqs': 1, 'enforce_eager': True,
        'security': {'api_key_env': 'MODELCHEF_API_KEY',
                     'tls_certfile': 'artifacts/tls/fullchain.pem',
                     'tls_keyfile': 'artifacts/tls/server.key',
                     'max_body_bytes': 1048576, 'requests_per_minute': 30,
                     'max_concurrent_requests': 1, 'request_timeout_seconds': 120}}},
        sort_keys=False), encoding='utf-8')
    return {'run_id': run_id, 'export_dir': str(output),
            'inference_config': str(serving_config), 'files': sorted(entries)}
