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


def test_progress_reports_failure_and_stops_heartbeat(capsys):
    import threading

    import pytest

    from modelchef.progress import operation
    before = set(threading.enumerate())
    with pytest.raises(ValueError), operation('test'):
        raise ValueError('failure')
    assert 'did not complete' in capsys.readouterr().err
    assert set(threading.enumerate()) == before
