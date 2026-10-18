import io
import json
from pathlib import Path

import pytest

from modelchef.recovery import RecoveryStore, digest_file


class S3:
    def __init__(self):
        self.objects = {}
        self.fail = False
        self.uploads = []

    def upload_file(self, path, bucket, key, ExtraArgs):
        if self.fail:
            raise OSError('interrupted transfer')
        assert ExtraArgs['ServerSideEncryption'] == 'AES256'
        self.objects[key] = Path(path).read_bytes()
        self.uploads.append(key)

    def put_object(self, Bucket, Key, Body, **kwargs):
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):
        return {'Body': io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, path):
        Path(path).write_bytes(self.objects[key])


def sharded_run(root, step=1):
    run = root / 'artifacts/training/run'
    checkpoint = run / f'checkpoints/step-{step:06d}.dcp'
    (checkpoint / 'state').mkdir(parents=True)
    (run / 'config.json').write_text('{"stage": {}, "training": {"distributed": "fsdp2"}}')
    (run / 'tokenizer.json').write_bytes(b'frozen tokenizer')
    (run / 'tokens-train.bin').write_bytes(b'prepared train tokens')
    for name in ['state/.metadata', 'state/__0_0.distcp', 'state/__1_0.distcp', 'rank-0.pt', 'rank-1.pt']:
        (checkpoint / name).write_bytes(f'{step}:{name}:weights-or-optimizer-or-rng'.encode())
    manifest = {'format': 'modelchef-fsdp2-v1', 'world_size': 2,
                'files': {p.relative_to(checkpoint).as_posix(): digest_file(p)
                          for p in checkpoint.rglob('*') if p.is_file()}}
    (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    return run, checkpoint


def test_sharded_snapshot_atomic_restore_and_reuse(tmp_path):
    root = tmp_path / 'guest'
    run, checkpoint = sharded_run(root)
    events = run / 'tensorboard/training/events.out.tfevents.synthetic'
    events.parent.mkdir(parents=True)
    events.write_bytes(b'flushed tensorboard events')
    client = S3()
    store = RecoveryStore(client, 's3://bucket/fsdp/recovery')
    first = store.snapshot(root, run, checkpoint)
    assert first['checkpoint_format'] == 'fsdp2'
    assert events.relative_to(root).as_posix() in first['files']
    store.snapshot(root, run, checkpoint)
    assert len(client.uploads) == len(set(client.uploads))
    destination = tmp_path / 'restored'
    downloaded = store.fetch(destination, 100000)
    for name in first['files']:
        assert (destination / name).read_bytes() == (root / name).read_bytes()
    store.fetch(destination, 100000)  # Idempotent reuse of a committed restored directory.
    assert (destination / downloaded['checkpoint'] / 'manifest.json').is_file()
    (destination / downloaded['checkpoint'] / 'rank-0.pt').write_bytes(b'different')
    with pytest.raises(ValueError, match='checksum'):
        store.fetch(destination, 100000)
    assert (destination / downloaded['checkpoint'] / 'rank-0.pt').read_bytes() == b'different'


def test_interrupted_shard_upload_keeps_prior_s3_commit(tmp_path):
    root = tmp_path / 'guest'
    run, checkpoint = sharded_run(root)
    client = S3()
    store = RecoveryStore(client, 's3://bucket/fsdp/recovery')
    original = store.snapshot(root, run, checkpoint)
    _, second = sharded_run(root, 2)
    upload = client.upload_file
    count = 0

    def interrupted(*args, **kwargs):
        nonlocal count
        count += 1
        if count == 3:
            raise OSError('interrupted shard upload')
        return upload(*args, **kwargs)

    client.upload_file = interrupted
    with pytest.raises(OSError, match='interrupted'):
        store.snapshot(root, run, second)
    assert json.loads(client.objects['fsdp/recovery/latest.json']) == original
    destination = tmp_path / 'restored'
    store.fetch(destination, 100000)
    assert (destination / original['checkpoint'] / 'manifest.json').exists()
    assert not (destination / second.relative_to(root)).exists()


@pytest.mark.parametrize('failed_key', ['manifests/', 'latest.json'])
def test_failed_manifest_publication_preserves_previous_pointer(tmp_path, failed_key):
    run, checkpoint = sharded_run(tmp_path)
    client = S3()
    store = RecoveryStore(client, 's3://bucket/fsdp/recovery')
    original = store.snapshot(tmp_path, run, checkpoint)
    _, second = sharded_run(tmp_path, 2)
    put = client.put_object

    def fail_commit(**kwargs):
        if failed_key in kwargs['Key']:
            raise OSError('commit interrupted')
        put(**kwargs)

    client.put_object = fail_commit
    with pytest.raises(OSError, match='commit interrupted'):
        store.snapshot(tmp_path, run, second)
    assert json.loads(client.objects['fsdp/recovery/latest.json']) == original


@pytest.mark.parametrize('failure', ['corrupt', 'missing'])
def test_failed_shard_download_never_publishes_checkpoint(tmp_path, failure):
    root = tmp_path / 'guest'
    run, checkpoint = sharded_run(root)
    client = S3()
    store = RecoveryStore(client, 's3://bucket/fsdp/recovery')
    manifest = store.snapshot(root, run, checkpoint)
    key = manifest['files'][checkpoint.relative_to(root).as_posix() + '/state/__1_0.distcp']['key']
    if failure == 'corrupt':
        client.objects[key] = b'corrupted'
    else:
        del client.objects[key]
    destination = tmp_path / 'restore'
    with pytest.raises((ValueError, KeyError)):
        store.fetch(destination, 100000)
    assert not (destination / manifest['checkpoint']).exists()
    assert checkpoint.exists()


@pytest.mark.parametrize('damage', ['missing', 'extra', 'escaped'])
def test_refuse_incomplete_or_unsafe_sharded_snapshot(tmp_path, damage):
    run, checkpoint = sharded_run(tmp_path)
    if damage == 'missing':
        (checkpoint / 'state/__1_0.distcp').unlink()
    elif damage == 'extra':
        (checkpoint / 'undeclared').write_bytes(b'not in commit')
    else:
        manifest = json.loads((checkpoint / 'manifest.json').read_text())
        manifest['files']['../../outside'] = 'bad'
        (checkpoint / 'manifest.json').write_text(json.dumps(manifest))
    client = S3()
    with pytest.raises(ValueError):
        RecoveryStore(client, 's3://bucket/fsdp/recovery').snapshot(tmp_path, run, checkpoint)
    assert 'fsdp/recovery/latest.json' not in client.objects


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


def test_training_data_restored_and_not_reuploaded_at_each_checkpoint(tmp_path):
    root = tmp_path / 'guest'
    run = root / 'artifacts/training/run'
    (run / 'checkpoints').mkdir(parents=True)
    (run / 'config.json').write_text('{"stage": {}}')
    names = ['raw.jsonl', 'train.jsonl', 'validation.jsonl', 'test.jsonl',
             'tokens.pt', 'tokens-train.bin', 'tokens-validation.bin', 'tokens-test.bin',
             'tokenizer.json', 'source-manifest.json', 'processing.json', 'data-attribution.json']
    for name in names:
        (run / name).write_bytes(('saved-' + name).encode())
    (run / 'dedup.sqlite').write_bytes(b'regenerable-cache')
    first = run / 'checkpoints/step-000001.pt'
    first.write_bytes(b'first-weights')
    client = S3()
    store = RecoveryStore(client, 's3://bucket/run/recovery')
    initial = store.snapshot(root, run, first)
    second = run / 'checkpoints/step-000002.pt'
    second.write_bytes(b'second-weights')
    final = store.snapshot(root, run, second)
    for name in names:
        relative = 'artifacts/training/run/' + name
        assert final['files'][relative] == initial['files'][relative]
        assert client.uploads.count(final['files'][relative]['key']) == 1
    assert 'artifacts/training/run/dedup.sqlite' not in final['files']
    restored = tmp_path / 'restored'
    store.fetch(restored, 100000)
    for name in names:
        assert (restored / 'artifacts/training/run' / name).read_bytes() == (run / name).read_bytes()
