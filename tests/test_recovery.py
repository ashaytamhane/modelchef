import io
import json
from pathlib import Path

import pytest

from modelchef.recovery import RecoveryStore


class S3:
    def __init__(self):
        self.objects = {}
        self.fail = False

    def upload_file(self, path, bucket, key, ExtraArgs):
        if self.fail:
            raise OSError('interrupted transfer')
        assert ExtraArgs['ServerSideEncryption'] == 'AES256'
        self.objects[key] = Path(path).read_bytes()

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


def test_snapshot_restore_and_interruption(tmp_path):
    root = tmp_path / 'guest'
    run = root / 'artifacts/training/run'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'config.json').write_text(json.dumps({'stage': {}}))
    (run / 'raw.jsonl').write_text('exact training data')
    (run / 'tokens.pt').write_bytes(b'token arrays')
    first = run / 'checkpoints/step-000001.pt'
    first.write_bytes(b'weights optimizer rng')
    client = S3()
    store = RecoveryStore(client, 's3://bucket/job/recovery')
    manifest = store.snapshot(root, run, first)
    second = run / 'checkpoints/step-000002.pt'
    second.write_bytes(b'new checkpoint')
    client.fail = True
    with pytest.raises(OSError):
        store.snapshot(root, run, second)
    assert json.loads(client.objects['job/recovery/latest.json']) == manifest
    restored = tmp_path / 'restore'
    store.fetch(restored, 10000)
    assert (restored / manifest['checkpoint']).read_bytes() == first.read_bytes()
    assert (restored / 'artifacts/training/run/raw.jsonl').read_bytes() == (run / 'raw.jsonl').read_bytes()
    with pytest.raises(ValueError, match='max-bytes'):
        store.fetch(tmp_path / 'too-small', 1)
    client.objects['job/recovery/latest.json'] = json.dumps({
        'schema_version': 1, 'files': {'../outside': {'bytes': 1, 'key': 'bad', 'sha256': 'bad'}}}).encode()
    with pytest.raises(ValueError, match='Unsafe'):
        store.fetch(tmp_path / 'unsafe', 100)


def test_lineage_and_checksum(tmp_path):
    parent = tmp_path / 'parent'
    child = tmp_path / 'child'
    for folder in [parent, child]:
        (folder / 'checkpoints').mkdir(parents=True)
        (folder / 'checkpoints/step-000001.pt').write_bytes(folder.name.encode())
        (folder / 'config.json').write_text(json.dumps({'stage':
            {'parent_run': 'parent'} if folder == child else {}}))
    client = S3()
    store = RecoveryStore(client, 's3://bucket/job')
    manifest = store.snapshot(tmp_path, child, child / 'checkpoints/step-000001.pt')
    assert 'parent/checkpoints/step-000001.pt' in manifest['files']
    key = manifest['files']['child/checkpoints/step-000001.pt']['key']
    client.objects[key] = b'corrupt'
    with pytest.raises(ValueError, match='checksum'):
        store.fetch(tmp_path / 'restore', 10000)
