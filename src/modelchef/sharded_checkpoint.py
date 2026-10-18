"""Collective DCP checkpoints: publish only complete directories, never full GPU gathers."""
import json
import os
import shutil
import time
import uuid
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

from modelchef.checkpointing import atomic_save, storage_bytes
from modelchef.corpus import file_digest


class TrainingState:
    def __init__(self, model, optimizer):
        self.model, self.optimizer = model, optimizer

    def state_dict(self):
        initializing = not self.optimizer.state
        model, optimizer = get_state_dict(self.model, self.optimizer)
        if initializing and isinstance(self.optimizer, torch.optim.AdamW):
            # DCP materializes lazy AdamW state with a zero-LR step. That must not
            # advance the bias-correction counter of a step-zero checkpoint.
            for states in [self.optimizer.state, optimizer.get('state', {})]:
                for state in states.values():
                    step = state.get('step')
                    if isinstance(step, torch.Tensor):
                        step.zero_()
                    elif step is not None:
                        state['step'] = 0
        return {'model': model, 'optimizer': optimizer}

    def load_state_dict(self, state):
        set_state_dict(self.model, self.optimizer, model_state_dict=state['model'],
                       optim_state_dict=state['optimizer'])


def _collective_error(error, group=None):
    errors = [None] * dist.get_world_size(group)
    dist.all_gather_object(errors, str(error) if error else None, group=group)
    if any(errors):
        raise RuntimeError('Distributed checkpoint failed: ' + '; '.join(e for e in errors if e))


@contextmanager
def checkpoint_io_group():
    """Use CPU coordination for host I/O that can outlast the NCCL training timeout."""
    deadline = float(os.environ.get('MODELCHEF_DEADLINE_EPOCH', str(time.time() + 3600)))
    group = dist.new_group(backend='gloo', timeout=timedelta(seconds=max(300, deadline-time.time())))
    try:
        yield group
    finally:
        dist.destroy_process_group(group)


def save_sharded(model, optimizer, path, metadata, rank_state):
    """All ranks participate. Shared filesystem and same-world recovery are required."""
    with checkpoint_io_group() as group:
        return _save_sharded(model, optimizer, path, metadata, rank_state, group)


def _save_sharded(model, optimizer, path, metadata, rank_state, group):
    path = Path(path)
    rank, world = dist.get_rank(), dist.get_world_size()
    name = [path.name + '.incomplete-' + uuid.uuid4().hex if rank == 0 else None]
    dist.broadcast_object_list(name, src=0, group=group)
    temporary = path.with_name(name[0])
    error = None
    try:
        if path.exists():
            raise FileExistsError(f'Refusing to overwrite committed checkpoint {path}')
        temporary.mkdir(parents=True, exist_ok=True)
    except Exception as exc:  # noqa: BLE001 - synchronize all rank-local I/O errors
        error = exc
    _collective_error(error, group)
    state = TrainingState(model, optimizer)
    payload = state.state_dict()
    # Estimate local tensor staging without collecting any full DTensor.
    def local(value):
        if hasattr(value, 'to_local'):
            return value.to_local()
        if isinstance(value, dict):
            return {k: local(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [local(v) for v in value]
        return value
    sizes = [None] * world
    dist.all_gather_object(sizes, storage_bytes(local(payload)), group=group)
    required = int(sum(sizes) * 1.15) + world * 64 * 1024**2
    error = None
    if shutil.disk_usage(temporary).free < required:
        error = OSError(f'Need approximately {required} free bytes for a new sharded checkpoint')
    _collective_error(error, group)
    dcp.save({'train': payload}, checkpoint_id=temporary / 'state', process_group=group)
    error = None
    try:
        atomic_save(rank_state, temporary / f'rank-{rank}.pt')
    except Exception as exc:  # noqa: BLE001 - synchronize all rank-local I/O errors
        error = exc
    _collective_error(error, group)
    # A manifest is the commit record. Retention must ignore incomplete directories.
    error = None
    if rank == 0:
        try:
            files = {p.relative_to(temporary).as_posix(): file_digest(p)
                     for p in temporary.rglob('*') if p.is_file()}
            manifest = {**metadata, 'format': 'modelchef-fsdp2-v1', 'world_size': world,
                        'files': files, 'torch_version': torch.__version__}
            with (temporary / 'manifest.json').open('w', encoding='utf-8') as stream:
                json.dump(manifest, stream, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.rename(path)
        except Exception as exc:  # noqa: BLE001 - synchronize commit errors
            error = exc
    _collective_error(error, group)
    dist.barrier(group=group)
    return path


def read_manifest(path, world_size=None, *, for_export=False):
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
    if manifest.get('format') != 'modelchef-fsdp2-v1':
        raise ValueError('Not a committed ModelChef FSDP2 checkpoint')
    if world_size is not None and manifest['world_size'] != world_size:
        raise ValueError('Exact FSDP2 resume requires the same world size (RNG and batch semantics)')
    version_matches = (manifest['torch_version'].split('+')[0] == torch.__version__.split('+')[0]
                       if for_export else manifest['torch_version'] == torch.__version__)
    if not version_matches:
        raise ValueError('FSDP2 resume requires the recorded PyTorch version; DCP is version-sensitive')
    from modelchef.recovery import sharded_files
    try:
        sharded_files(path)
    except (OSError, ValueError) as error:
        raise ValueError(f'Sharded checkpoint contents changed or are incomplete: {error}') from error
    return manifest


def load_sharded_model(checkpoint, architecture, vocab_size, *, run_folder=None):
    """Consolidate only native model tensors on CPU; do not load optimizer/RNG state."""
    if dist.is_initialized():
        raise ValueError('DCP inference/export must run outside an initialized distributed job')
    from modelchef.engines import create_model
    manifest = read_manifest(checkpoint, for_export=True)
    if run_folder is not None:
        identities = manifest.get('identities', {})
        for name in ['config.json', 'tokenizer.json', 'tokenizer-roles.json']:
            if name in identities and file_digest(Path(run_folder) / name) != identities[name]:
                raise ValueError('DCP inference configuration/tokenizer identity changed')
    with torch.device('meta'):
        model = create_model(architecture, vocab_size)
    model.to_empty(device='cpu')
    state = {'train': {'model': model.state_dict()}}
    dcp.load(state, checkpoint_id=Path(checkpoint) / 'state', no_dist=True)
    return state['train']['model']


def load_sharded(model, optimizer, path, identities):
    with checkpoint_io_group() as group:
        return _load_sharded(model, optimizer, path, identities, group)


def _load_sharded(model, optimizer, path, identities, group):
    error, manifest = None, None
    try:
        manifest = read_manifest(path, dist.get_world_size())
        if manifest['identities'] != identities:
            raise ValueError('Configuration/tokenizer/prepared snapshot identities changed')
    except Exception as exc:  # noqa: BLE001 - synchronize validation errors
        error = exc
    _collective_error(error, group)
    state = TrainingState(model, optimizer)
    dcp.load({'train': state}, checkpoint_id=Path(path) / 'state', process_group=group)
    rank_state = torch.load(Path(path) / f'rank-{dist.get_rank()}.pt',
                            map_location='cpu', weights_only=True)
    return manifest, rank_state


def publish_checkpoint_to_s3(root, folder, checkpoint):
    """All ranks wait for rank zero's complete S3 commit before acknowledging recovery."""
    if not os.environ.get('MODELCHEF_RECOVERY_S3_URI'):
        return
    # Bulk uploads may exceed the NCCL watchdog's ordinary training timeout.
    # Coordinate on CPU with a bounded I/O timeout; no NCCL operation waits on S3.
    with checkpoint_io_group() as group:
        error = None
        if dist.get_rank() == 0:
            try:
                from modelchef.recovery import checkpoint_to_s3
                checkpoint_to_s3(root, folder, checkpoint)
            except Exception as exc:  # noqa: BLE001 - every rank must receive upload failure
                error = exc
        _collective_error(error, group)


def publish_sharded_to_s3(root, folder, checkpoint):
    """Compatibility entry point for sharded callers; ordinary DDP files use the same commit."""
    return publish_checkpoint_to_s3(root, folder, checkpoint)


def prune_sharded(checkpoints, keep, protected=()):
    """Collectively prune only committed children; preserve protected inputs by absolute identity."""
    with checkpoint_io_group() as group:
        error = None
        if dist.get_rank() == 0:
            try:
                _prune_sharded(checkpoints, keep, protected)
            except Exception as exc:  # noqa: BLE001 - propagate filesystem failures to every rank
                error = exc
        _collective_error(error, group)


def _prune_sharded(checkpoints, keep, protected=()):
    if isinstance(keep, bool) or not isinstance(keep, int) or keep < 1:
        raise ValueError('MODELCHEF_LOCAL_CHECKPOINT_KEEP must be positive')
    checkpoints = Path(checkpoints).resolve()
    protected = {Path(path).resolve() for path in protected if path is not None}
    committed = sorted((path for path in checkpoints.glob('step-*.dcp')
                        if (path / 'manifest.json').is_file()),
                       key=lambda path: (int(path.name.split('-')[1].split('.')[0]),
                                         path.stat().st_mtime_ns, path.name))
    for path in committed[:-keep]:
        resolved = path.resolve()
        if path.is_symlink() or not resolved.is_relative_to(checkpoints) or resolved == checkpoints:
            raise ValueError('Refusing checkpoint retention outside the checkpoint directory')
        if resolved not in protected:
            shutil.rmtree(resolved)
