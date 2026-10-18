"""Durable JSONL and TensorBoard loss logging shared by training backends."""
import atexit
import json
import math
import os
import time
from pathlib import Path

_writers = {}


def tensorboard_info(folder):
    folder = Path(folder).resolve()
    logdir = folder / 'tensorboard' / 'training'
    return {'logdir': str(logdir), 'url': os.environ.get('MODELCHEF_TENSORBOARD_URL'),
            'command': f'tensorboard --logdir "{folder / "tensorboard"}" --port 6006',
            'validation_policy': 'Measured evaluations only; no interpolation or test metrics'}


def loss_scalars(record):
    train = record.get('training_loss', record.get('loss'))
    if isinstance(train, (int, float)) and not isinstance(train, bool) and math.isfinite(train):
        yield 'loss/train', train
    validation = record.get('validation', {})
    if not isinstance(validation, dict):
        return
    for name, value in validation.items():
        tag, loss = ('loss/validation', value) if name == 'loss' else (
            f'loss/validation/{name}', value.get('loss') if isinstance(value, dict) else None)
        if isinstance(loss, (int, float)) and not isinstance(loss, bool) and math.isfinite(loss):
            yield tag, loss


def record_metrics(folder, record):
    """Rank zero only. Flush actual optimizer-step measurements before returning."""
    from tensorboard.compat.proto.event_pb2 import Event, SessionLog
    from tensorboard.compat.proto.summary_pb2 import Summary
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    folder = Path(folder).resolve()
    key = (os.getpid(), folder)
    if key not in _writers:
        info = tensorboard_info(folder)
        _writers[key] = EventFileWriter(info['logdir'])
        _writers[key].add_event(Event(wall_time=time.time(), step=record['step'],
                                     session_log=SessionLog(status=SessionLog.START)))
        (folder / 'tensorboard.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
        from modelchef.progress import report
        report('TensorBoard: ' + (info['url'] or info['command']))
    with (folder / 'metrics.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record) + '\n')
    values = [Summary.Value(tag=tag, simple_value=value) for tag, value in loss_scalars(record)]
    if values:
        _writers[key].add_event(Event(wall_time=time.time(), step=record['step'],
                                     summary=Summary(value=values)))
    _writers[key].flush()


def close_metrics(folder):
    writer = _writers.pop((os.getpid(), Path(folder).resolve()), None)
    if writer:
        writer.close()


@atexit.register
def _close_all():
    for writer in list(_writers.values()):
        writer.close()
    _writers.clear()
