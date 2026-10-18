"""Atomic checkpoint writes with disk headroom and explicit I/O failures."""
import os
import shutil
from pathlib import Path

import torch


def storage_bytes(value):
    seen = set()

    def visit(item):
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            identity = (str(item.device), storage.data_ptr(), storage.nbytes())
            if identity in seen:
                return 0
            seen.add(identity)
            return storage.nbytes()
        if isinstance(item, dict):
            return sum(visit(v) for v in item.values())
        if isinstance(item, (list, tuple)):
            return sum(visit(v) for v in item)
        return 0

    return visit(value)


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    required = int(storage_bytes(payload) * 1.1) + 64 * 1024**2
    available = shutil.disk_usage(path.parent).free
    if available < required:
        raise OSError(f'Checkpoint requires about {required / 1024**3:.2f} GiB free '
                      f'for its temporary file; only {available / 1024**3:.2f} GiB available. '
                      'The previous checkpoint was preserved.')
    temporary = path.with_suffix('.tmp')
    try:
        with temporary.open('wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    except Exception as error:
        temporary.unlink(missing_ok=True)
        raise OSError(f'Checkpoint write failed at {path}: {error}. '
                      'The previous checkpoint was preserved; check disk space and filesystem I/O.') from error
