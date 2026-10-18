from collections import namedtuple

import pytest
import torch

from modelchef.checkpointing import atomic_save, storage_bytes


def test_atomic_checkpoint_roundtrip_and_shared_storage(tmp_path):
    tensor = torch.arange(20)
    payload = {'model': tensor, 'view': tensor[:4], 'nested': [tensor]}
    assert storage_bytes(payload) == tensor.untyped_storage().nbytes()
    path = tmp_path / 'step.pt'
    atomic_save(payload, path)
    assert torch.equal(torch.load(path, weights_only=True)['model'], tensor)
    assert not path.with_suffix('.tmp').exists()


def test_failed_write_preserves_previous_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'step.pt'
    path.write_bytes(b'previous valid checkpoint')

    def fail(payload, stream):
        stream.write(b'partial')
        raise OSError('simulated filesystem write failure')

    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OSError, match='previous checkpoint was preserved'):
        atomic_save({'tensor': torch.ones(4)}, path)
    assert path.read_bytes() == b'previous valid checkpoint'
    assert not path.with_suffix('.tmp').exists()


def test_low_disk_space_rejects_before_touching_checkpoint(tmp_path, monkeypatch):
    path = tmp_path / 'step.pt'
    path.write_bytes(b'previous')
    usage = namedtuple('Usage', 'total used free')
    monkeypatch.setattr('modelchef.checkpointing.shutil.disk_usage', lambda _: usage(100, 99, 1))
    with pytest.raises(OSError, match='GiB available'):
        atomic_save({'tensor': torch.ones(4)}, path)
    assert path.read_bytes() == b'previous'
