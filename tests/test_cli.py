from modelchef.cli import check_environment


def test_environment_shape():
    info = check_environment()
    assert "python" in info
    assert isinstance(info["cuda_available"], bool)


def test_cli_progress_preserves_json_stdout_and_quiet(monkeypatch, capsys):
    import json

    from modelchef.cli import main
    monkeypatch.setattr('sys.argv', ['modelchef', 'check-config', 'configs/local.yaml'])
    main()
    output = capsys.readouterr()
    assert json.loads(output.out)['project']['name'] == 'modelchef'
    assert 'Starting check-config' in output.err
    assert 'completed' in output.err
    monkeypatch.setattr('sys.argv', ['modelchef', 'check-config', 'configs/local.yaml', '--quiet'])
    main()
    output = capsys.readouterr()
    assert json.loads(output.out)['project']['name'] == 'modelchef'
    assert output.err == ''


def test_progress_reports_failure_without_background_worker(capsys):
    import threading

    import pytest

    from modelchef.progress import operation
    before = set(threading.enumerate())
    with pytest.raises(ValueError), operation('test'):
        raise ValueError('failure')
    assert 'did not complete' in capsys.readouterr().err
    assert set(threading.enumerate()) == before


def test_progress_waits_silently_between_specific_updates(capsys, monkeypatch):
    import threading

    from modelchef.progress import operation, report

    def forbidden_worker(*args, **kwargs):
        raise AssertionError('Progress must not create a periodic timer thread')

    monkeypatch.setattr(threading, 'Thread', forbidden_worker)
    with operation('doctor'):
        report('Importing PyTorch; the first import can take several seconds.')
        waiting = capsys.readouterr()
        assert 'Importing PyTorch' in waiting.err
        assert 'still running' not in waiting.err
        assert waiting.out == ''
        report('Checking CUDA/ROCm and Apple MPS availability.')
    completed = capsys.readouterr()
    assert 'Checking CUDA/ROCm' in completed.err
    assert 'doctor completed' in completed.err
    assert 'still running' not in completed.err


def test_quiet_suppresses_specific_stage_updates(capsys):
    from modelchef.progress import operation, report

    with operation('doctor', quiet=True):
        report('Importing PyTorch.')
    assert capsys.readouterr().err == ''
