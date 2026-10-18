import json
import math

import pytest

from modelchef.training_metrics import close_metrics, loss_scalars, record_metrics, tensorboard_info


def test_loss_tags_include_actual_train_and_validation_only():
    assert dict(loss_scalars({'training_loss': 2., 'validation': {'loss': 3.},
                             'test': {'loss': .01}})) == {'loss/train': 2., 'loss/validation': 3.}
    assert dict(loss_scalars({'loss': 1., 'validation': {'source_a': {'loss': 2.},
        'source_b': {'loss': 3.}}})) == {
            'loss/train': 1., 'loss/validation/source_a': 2., 'loss/validation/source_b': 3.}
    assert dict(loss_scalars({'loss': math.nan, 'validation': {'loss': True}})) == {}
    assert dict(loss_scalars({'loss': 2.})) == {'loss/train': 2.}


def test_events_are_durable_and_resume_does_not_replace_prior_points(tmp_path, monkeypatch):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    monkeypatch.setenv('MODELCHEF_TENSORBOARD_URL', 'https://owned-6006.proxy.runpod.net/')
    try:
        record_metrics(tmp_path, {'step': 7, 'training_loss': 2.})
        record_metrics(tmp_path, {'step': 8, 'training_loss': 1., 'validation': {'loss': 3.},
                                  'test': {'loss': .1}})
    finally:
        close_metrics(tmp_path)
    record_metrics(tmp_path, {'step': 9, 'training_loss': .5})
    close_metrics(tmp_path)
    events = EventAccumulator(tensorboard_info(tmp_path)['logdir']).Reload()
    assert [(v.step, v.value) for v in events.Scalars('loss/train')] == [(7, 2.), (8, 1.), (9, .5)]
    assert [(v.step, v.value) for v in events.Scalars('loss/validation')] == [(8, 3.)]
    assert not any('test' in tag for tag in events.Tags()['scalars'])
    assert len((tmp_path / 'metrics.jsonl').read_text().splitlines()) == 3
    assert json.loads((tmp_path / 'tensorboard.json').read_text())['url'].startswith('https://owned-')


def test_local_access_is_a_command_without_inventing_a_running_server(tmp_path, monkeypatch):
    monkeypatch.delenv('MODELCHEF_TENSORBOARD_URL', raising=False)
    info = tensorboard_info(tmp_path)
    assert info['url'] is None
    assert '--port 6006' in info['command']


def test_rollback_resume_purges_superseded_points_in_tensorboard_view(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    record_metrics(tmp_path, {'step': 1, 'training_loss': 4.})
    record_metrics(tmp_path, {'step': 2, 'training_loss': 3.})
    close_metrics(tmp_path)
    record_metrics(tmp_path, {'step': 2, 'training_loss': 2.})
    close_metrics(tmp_path)
    events = EventAccumulator(tensorboard_info(tmp_path)['logdir']).Reload()
    assert [(v.step, v.value) for v in events.Scalars('loss/train')] == [(1, 4.), (2, 2.)]
    assert len((tmp_path / 'metrics.jsonl').read_text().splitlines()) == 3
