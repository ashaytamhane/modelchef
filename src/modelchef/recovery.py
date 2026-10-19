"""Commit resumable training snapshots to S3 before acknowledging a checkpoint."""
import argparse
import hashlib
import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse


def digest_file(path):
    result = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def location(uri):
    parsed = urlparse(uri)
    prefix = parsed.path.strip('/')
    if parsed.scheme != 's3' or not parsed.netloc or not prefix or parsed.query or parsed.fragment:
        raise ValueError('Recovery location must be s3://BUCKET/PREFIX')
    return parsed.netloc, prefix


def sharded_files(checkpoint, *, staging=False):
    """Validate a committed DCP directory without requiring PyTorch on the controller."""
    checkpoint = Path(checkpoint)
    if checkpoint.is_symlink() or (not staging and checkpoint.suffix != '.dcp') or not checkpoint.is_dir():
        raise ValueError('Recovery requires a committed .dcp directory, not a link or partial write')
    paths = list(checkpoint.rglob('*'))
    if any(path.is_symlink() for path in paths):
        raise ValueError('Recovery refuses symlinked checkpoint artifacts')
    commit = checkpoint / 'manifest.json'
    if not commit.is_file() or commit.stat().st_size > 8_000_000:
        raise ValueError('Sharded checkpoint commit manifest is missing or oversized')
    manifest = json.loads(commit.read_text(encoding='utf-8'))
    if manifest.get('format') != 'modelchef-fsdp2-v1':
        raise ValueError('Unsupported sharded checkpoint format')
    files = manifest.get('files', {})
    world = manifest.get('world_size')
    if (not isinstance(files, dict) or isinstance(world, bool) or not isinstance(world, int)
            or world < 1 or world > len(files)):
        raise ValueError('Invalid sharded checkpoint world size')
    required = {'state/.metadata', *(f'rank-{rank}.pt' for rank in range(world))}
    if not required <= files.keys() or not any(name.endswith('.distcp') for name in files):
        raise ValueError('Sharded checkpoint is missing state metadata, shards or rank RNG files')
    expected = set()
    for name, checksum in files.items():
        path = (checkpoint / name).resolve()
        if not path.is_relative_to(checkpoint.resolve()) or not path.is_file():
            raise ValueError('Unsafe or missing sharded checkpoint file')
        if digest_file(path) != checksum:
            raise ValueError('Sharded checkpoint checksum differs')
        expected.add(path)
    actual = {path.resolve() for path in paths if path.is_file() and path != commit}
    if actual != expected:
        raise ValueError('Sharded checkpoint contains undeclared files')
    return sorted(expected) + [commit]


class RecoveryStore:
    def __init__(self, client, uri):
        self.client = client
        self.bucket, self.prefix = location(uri)
        self.uploaded = set()
        self.hashes = {}

    def snapshot(self, root, folder, checkpoint):
        if Path(checkpoint).is_symlink():
            raise ValueError('Recovery refuses symlinked checkpoints')
        root, folder, checkpoint = (Path(artifact_path).resolve() for artifact_path in (root, folder, checkpoint))
        if not folder.is_relative_to(root) or checkpoint.parent != folder / 'checkpoints':
            raise ValueError('Recovery checkpoint must belong to this workspace run')
        files = {}
        visited = set()

        def include_run(run, selected):
            if run in visited:
                return
            visited.add(run)
            if not run.is_relative_to(root):
                raise ValueError('Recovery parent must be inside the workspace')
            config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
            for field in ['parent_run', 'teacher_run']:
                parent = config.get('stage', {}).get(field)
                if parent:
                    parent = (root / parent).resolve()
                    candidates = sorted((parent / 'checkpoints').glob('step-*.pt'))
                    if not candidates:
                        raise ValueError('Recovery parent checkpoint is missing')
                    include_run(parent, candidates[-1])
            # Raw snapshots, split JSONL, token arrays, tokenizer and provenance are
            # required by strict resume. SQLite is a regenerable processing cache.
            paths = [p for p in run.iterdir() if p.is_file() and
                     p.suffix not in {'.tmp', '.sqlite', '.db'} and
                     p.name not in {'events.jsonl', 'resume-environments.jsonl'}]
            paths.extend(sharded_files(selected) if selected.is_dir() else [selected])
            tensorboard = run / 'tensorboard'
            if tensorboard.is_symlink():
                raise ValueError('Recovery refuses symlinked TensorBoard directories')
            if tensorboard.is_dir():
                paths.extend(p for p in tensorboard.rglob('events.out.tfevents.*') if p.is_file())
            export = run / 'export'
            if export.is_dir():
                paths.extend(p for p in export.iterdir() if p.is_file() and
                             p.suffix in {'.json', '.safetensors', '.txt', '.jinja', '.model'})
            for path in paths:
                if path.is_symlink():
                    raise ValueError('Recovery refuses symlinked artifacts')
                stat = path.stat()
                signature = (str(path), stat.st_size, stat.st_mtime_ns)
                sha = self.hashes.get(signature)
                if sha is None:
                    sha = digest_file(path)
                    self.hashes[signature] = sha
                key = f'{self.prefix}/objects/{sha}'
                if sha not in self.uploaded:
                    self.client.upload_file(str(path), self.bucket, key,
                        ExtraArgs={'ServerSideEncryption': 'AES256', 'Metadata': {'sha256': sha}})
                    self.uploaded.add(sha)
                files[path.relative_to(root).as_posix()] = {'key': key, 'sha256': sha,
                                                           'bytes': stat.st_size}

        include_run(folder, checkpoint)
        manifest = {'schema_version': 1, 'workspace': str(root),
                    'checkpoint': checkpoint.relative_to(root).as_posix(), 'files': files}
        if checkpoint.is_dir():
            manifest['checkpoint_format'] = 'fsdp2'
        body = json.dumps(manifest, sort_keys=True).encode()
        manifest_key = f'{self.prefix}/manifests/{hashlib.sha256(body).hexdigest()}.json'
        # The pointer is written last: a failed upload leaves the previous complete
        # snapshot available. No credentials or arbitrary workspace files are copied.
        for key in [manifest_key, f'{self.prefix}/latest.json']:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=body,
                                   ServerSideEncryption='AES256', ContentType='application/json')
        return manifest

    def fetch(self, destination, max_bytes, progress=None):
        destination = Path(destination).resolve()
        with self.client.get_object(Bucket=self.bucket, Key=f'{self.prefix}/latest.json')['Body'] as body:
            payload = body.read(8_000_001)
        if len(payload) > 8_000_000:
            raise ValueError('Recovery manifest exceeds size limit')
        manifest = json.loads(payload)
        if manifest.get('schema_version') != 1:
            raise ValueError('Unsupported recovery manifest')
        files = manifest['files']
        if max_bytes <= 0 or any(isinstance(item['bytes'], bool) or not isinstance(item['bytes'], int)
                                  or item['bytes'] < 0 for item in files.values()):
            raise ValueError('Invalid recovery byte budget or file size')
        if sum(item['bytes'] for item in files.values()) > max_bytes:
            raise ValueError('Recovery download exceeds --max-bytes')
        if progress:
            progress(f"Restoring {len(files)} verified files ({sum(item['bytes'] for item in files.values()):,} bytes).")
        for name, item in files.items():
            path = (destination / name).resolve()
            if not path.is_relative_to(destination) or path == destination:
                raise ValueError('Unsafe recovery path')
            if item['key'] != f"{self.prefix}/objects/{item['sha256']}":
                raise ValueError('Recovery object is outside the declared prefix')
        checkpoint = None
        if manifest.get('checkpoint_format') == 'fsdp2':
            if (destination / manifest['checkpoint']).is_symlink():
                raise ValueError('Recovery refuses a symlinked checkpoint destination')
            checkpoint = (destination / manifest['checkpoint']).resolve()
            if (not checkpoint.is_relative_to(destination) or checkpoint == destination
                    or checkpoint.suffix != '.dcp' or checkpoint.is_symlink()):
                raise ValueError('Unsafe sharded recovery checkpoint path')
            selected = {name: item for name, item in files.items()
                        if (destination / name).resolve().is_relative_to(checkpoint)}
            if str(checkpoint.relative_to(destination) / 'manifest.json').replace('\\', '/') not in selected:
                raise ValueError('Recovery manifest omits the sharded commit record')
            if checkpoint.exists():
                sharded_files(checkpoint)
                for name, item in selected.items():
                    path = destination / name
                    if not path.is_file() or path.stat().st_size != item['bytes'] or digest_file(path) != item['sha256']:
                        raise ValueError('Recovery refuses to overwrite a different checkpoint')
                stage = None
            else:
                stage = checkpoint.with_name(checkpoint.name + '.download-' + uuid.uuid4().hex)
                stage.mkdir(parents=True)
        else:
            selected, stage = {}, None
        for name, item in files.items():
            if name in selected:
                if stage is None:
                    continue
                path = stage / (destination / name).resolve().relative_to(checkpoint)
            else:
                path = destination / name
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                if path.stat().st_size == item['bytes'] and digest_file(path) == item['sha256']:
                    continue
                raise ValueError(f'Recovery refuses to overwrite a different file: {path}')
            temporary = path.with_suffix(path.suffix + '.download')
            if temporary.is_symlink():
                raise ValueError('Recovery refuses a symlinked temporary download')
            if progress and item['bytes'] >= 64_000_000:
                progress(f"Downloading {name}: {item['bytes']:,} bytes; SHA256 verification follows.")
            self.client.download_file(self.bucket, item['key'], str(temporary))
            if temporary.stat().st_size != item['bytes'] or digest_file(temporary) != item['sha256']:
                temporary.unlink()
                raise ValueError('Recovery object checksum or size differs')
            temporary.replace(path)
            if progress and item['bytes'] >= 64_000_000:
                progress(f'Verified {name}.')
        if stage is not None:
            sharded_files(stage, staging=True)
            if checkpoint.exists():
                raise FileExistsError('Recovery checkpoint appeared during download; refusing overwrite')
            stage.rename(checkpoint)
        return manifest


_store = None


def checkpoint_to_s3(root, folder, checkpoint):
    global _store
    uri = os.environ.get('MODELCHEF_RECOVERY_S3_URI')
    if not uri:
        return None
    if _store is None or (_store.bucket, _store.prefix) != location(uri):
        _store = make_store(uri)
    return _store.snapshot(root, folder, checkpoint)


def make_store(uri, profile=None, region=None):
    import boto3
    from botocore.config import Config
    session = boto3.Session(profile_name=profile, region_name=region)
    client = session.client('s3', config=Config(connect_timeout=10, read_timeout=60,
                                              retries={'mode': 'standard', 'max_attempts': 4}))
    return RecoveryStore(client, uri)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('uri', help='s3://BUCKET/RUN_PREFIX/recovery')
    parser.add_argument('destination', type=Path)
    parser.add_argument('--profile')
    parser.add_argument('--region')
    parser.add_argument('--max-bytes', type=int, default=250_000_000_000)
    args = parser.parse_args()
    if args.max_bytes <= 0:
        parser.error('--max-bytes must be positive')
    manifest = make_store(args.uri, args.profile, args.region).fetch(
        args.destination, args.max_bytes, lambda message: print(message, flush=True))
    print(json.dumps({'restored_checkpoint': str(args.destination / manifest['checkpoint']),
                      'original_workspace': manifest['workspace'],
                      'resume_note': 'For exact resume restore to the original workspace path on a new guest.'}, indent=2))


if __name__ == '__main__':
    main()
