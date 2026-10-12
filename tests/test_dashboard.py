import json

import pytest

from modelchef.dashboard import MetricsMirror, scalars


def test_scalars_excludes_nonfinite_and_boolean_values():
    assert dict(scalars({'loss': 2.0, 'validation': {'loss': 3}, 'bad': float('nan'),
                         'flag': True, 'status': 'running'})) == {'loss': 2.0, 'validation/loss': 3}


def test_live_partial_records_and_final_sample_roundtrip(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.summary.writer.event_file_writer import EventFileWriter

    metrics = tmp_path / 'metrics.jsonl'
    metrics.write_bytes(b'{"step": 1, "training_loss": 2.0}\n{"step": 2')
    (tmp_path / 'events.jsonl').write_text(json.dumps({'stage': 'Train', 'message': 'Step 1'}) + '\n')
    writer = EventFileWriter(str(tmp_path / 'events'))
    mirror = MetricsMirror(tmp_path, writer)
    try:
        assert mirror.update() == 1
        assert mirror.update() == 0
        with metrics.open('ab') as output:
            output.write(b', "training_loss": 1.0}\n')
        assert mirror.update() == 1
        (tmp_path / 'report.json').write_text(json.dumps({'status': 'completed',
            'optimizer_steps': 2, 'generation': 'Hello', 'test': {'loss': 1.5}}))
        assert mirror.update() == 0
        assert mirror.final_written
    finally:
        writer.close()
    events = EventAccumulator(str(tmp_path / 'events')).Reload()
    assert [(event.step, event.value) for event in events.Scalars('training_loss')] == [(1, 2), (2, 1)]
    assert events.Scalars('final/test/loss')[0].value == 1.5
    assert events.Tensors('final/generated_sample')[0].tensor_proto.string_val == [b'Hello']
    status = events.Tensors('activity/current_status')
    assert status[0].step == 1
    assert b'Train' in status[0].tensor_proto.string_val[0]
    assert status[-1].step == 2


def test_preparation_events_are_not_optimizer_steps(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    messages = ['=== 1. Configuration and hardware ===', 'Device=cuda',
                '=== 2. Acquire a bounded corpus snapshot ===', 'Reading local shard']
    (tmp_path / 'events.jsonl').write_text(''.join(json.dumps({'message': text}) + '\n'
                                               for text in messages))
    writer = EventFileWriter(str(tmp_path / 'events'))
    mirror = MetricsMirror(tmp_path, writer)
    try:
        mirror.update()
    finally:
        writer.close()
    events = EventAccumulator(str(tmp_path / 'events')).Reload()
    status = events.Tensors('activity/current_status')
    assert len(status) == 1
    assert status[0].step == 0
    text = status[0].tensor_proto.string_val[0]
    assert b'2. Acquire' in text and b'Reading local shard' in text
    assert b'===' not in text


def test_explicit_chat_checkpoint_does_not_require_completed_report(tmp_path):
    from modelchef.chat import checkpoint_for
    folder = tmp_path / 'run'
    (folder / 'checkpoints').mkdir(parents=True)
    checkpoint = folder / 'checkpoints/step-002000.pt'
    checkpoint.write_bytes(b'fixture')
    assert checkpoint_for(folder, 'step-002000.pt') == checkpoint.resolve()
    with pytest.raises(ValueError, match='inside this run'):
        checkpoint_for(folder, '../../unrelated.pt')
