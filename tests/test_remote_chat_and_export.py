import hashlib
import io
import json
import tarfile
from pathlib import Path

import pytest

from modelchef.aws_export import recover_export
from modelchef.remote_chat import completion


def test_remote_chat_requires_https_and_does_not_send_without_key(monkeypatch):
    monkeypatch.delenv('MODELCHEF_API_KEY', raising=False)
    with pytest.raises(ValueError, match='HTTPS'):
        completion('http://example.com/v1', 'model', 'Hello')
    with pytest.raises(ValueError, match='MODELCHEF_API_KEY'):
        completion('https://example.com/v1', 'model', 'Hello')


def test_remote_chat_uses_authenticated_completion(monkeypatch):
    import modelchef.remote_chat as module

    seen = {}
    monkeypatch.setenv('MODELCHEF_API_KEY', 'a' * 40)

    class Response:
        def __enter__(self):
            return io.BytesIO(b'{"choices":[{"text":"Hello back"}]}')

        def __exit__(self, *_):
            pass

    def fake_open(request, **kwargs):
        seen['url'] = request.full_url
        seen['auth'] = request.get_header('Authorization')
        seen['body'] = json.loads(request.data)
        seen['context'] = kwargs['context']
        return Response()

    monkeypatch.setattr(module, 'urlopen', fake_open)
    assert completion('https://example.com/v1', 'our-model', 'Hello') == 'Hello back'
    assert seen['url'] == 'https://example.com/v1/completions'
    assert seen['auth'] == 'Bearer ' + 'a' * 40
    assert seen['body']['prompt'] == 'Hello'
    assert seen['context'].verify_mode.name == 'CERT_REQUIRED'


def test_remote_chat_discovers_sole_model_over_https(monkeypatch):
    import modelchef.remote_chat as module

    monkeypatch.setenv('MODELCHEF_API_KEY', 'a' * 40)
    requests = []

    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return io.BytesIO(self.body)

        def __exit__(self, *_):
            pass

    def fake_open(request, **_):
        requests.append(request)
        if request.full_url.endswith('/models'):
            return Response(b'{"data":[{"id":"served-model"}]}')
        return Response(b'{"choices":[{"text":"Ready"}]}')

    monkeypatch.setattr(module, 'urlopen', fake_open)
    assert completion('https://example.com/v1', 'auto', 'Hello') == 'Ready'
    assert requests[0].get_header('Authorization') == 'Bearer ' + 'a' * 40
    assert json.loads(requests[1].data)['model'] == 'served-model'


def _results(root: Path, run_id: str, files: dict[str, bytes], passed=True):
    directory = root / 'artifacts/aws-runs' / run_id
    directory.mkdir(parents=True)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode='w:gz') as archive:
        for name, content in files.items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))
    content = stream.getvalue()
    (directory / 'results.tar.gz').write_bytes(content)
    (directory / 'report.json').write_text(json.dumps({
        'run_id': run_id, 'test_passed': passed,
        'results_sha256': hashlib.sha256(content).hexdigest()}))
    return directory


def test_recover_completed_export_without_extracting_unrelated_files(tmp_path):
    run_id = 'modelchef-123abc'
    prefix = 'artifacts/training/assistant-1/export/'
    _results(tmp_path, run_id, {prefix + 'config.json': b'{"max_position_embeddings":64}',
                                prefix + 'tokenizer.json': b'{}',
                                prefix + 'model.safetensors': b'weights',
                                'artifacts/workflows/assistant-1/secret.txt': b'private'})
    result = recover_export(tmp_path, run_id)
    output = Path(result['export_dir'])
    assert (output / 'model.safetensors').read_bytes() == b'weights'
    assert not (output / 'secret.txt').exists()
    assert Path(result['inference_config']).is_file()


def test_recover_export_rejects_unconfirmed_or_tampered_run(tmp_path):
    run_id = 'modelchef-123abc'
    directory = _results(tmp_path, run_id, {}, passed=False)
    with pytest.raises(ValueError, match='confirmed'):
        recover_export(tmp_path, run_id)
    report = json.loads((directory / 'report.json').read_text())
    report['test_passed'] = True
    (directory / 'report.json').write_text(json.dumps(report))
    (directory / 'results.tar.gz').write_bytes(b'tampered')
    with pytest.raises(ValueError, match='checksum'):
        recover_export(tmp_path, run_id)
