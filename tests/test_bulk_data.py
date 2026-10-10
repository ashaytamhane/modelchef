import io
import json
from types import SimpleNamespace
from urllib.request import Request

import pytest

from modelchef.bulk_data import DownloadRedirect, download_file


def test_bulk_transfer_is_bounded_and_cache_is_verified(tmp_path, monkeypatch):
    calls = []

    def transfer(request, timeout):
        calls.append(request)
        return io.BytesIO(b'abc')

    monkeypatch.setattr('modelchef.bulk_data.build_opener', lambda *args: SimpleNamespace(open=transfer))
    path = tmp_path / 'file.parquet'
    saved = download_file('https://huggingface.co/file', path, 3, 30, {}, lambda msg: None)
    assert saved['size'] == 3
    download_file('https://huggingface.co/file', path, 3, 30, {}, lambda msg: None)
    assert len(calls) == 1
    path.write_bytes(b'xyz')
    with pytest.raises(ValueError, match='checksum'):
        download_file('https://huggingface.co/file', path, 3, 30, {}, lambda msg: None)


def test_bulk_transfer_oversize_removes_partial_file(tmp_path, monkeypatch):
    monkeypatch.setattr('modelchef.bulk_data.build_opener', lambda *args: SimpleNamespace(
        open=lambda *args, **kw: io.BytesIO(b'oversized')))
    with pytest.raises(ValueError, match='approved metadata size'):
        download_file('https://huggingface.co/file', tmp_path / 'file.parquet', 2, 30, {}, lambda msg: None)
    assert not list(tmp_path.glob('*.part'))


def test_redirect_does_not_forward_token_to_download_cdn():
    request = Request('https://huggingface.co/file', headers={'Authorization': 'Bearer test'})
    redirected = DownloadRedirect().redirect_request(request, None, 302, 'Found', {}, 'https://cdn.example/file')
    assert not redirected.has_header('Authorization')
    with pytest.raises(ValueError, match='HTTPS'):
        DownloadRedirect().redirect_request(request, None, 302, 'Found', {}, 'http://cdn.example/file')


def test_bulk_reader_selects_rows_and_reuses_complete_snapshot(tmp_path, monkeypatch):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq

    from modelchef.corpus import acquire
    from modelchef.run_config import RunConfig

    source = io.BytesIO()
    pq.write_table(pa.table({'text': [f'story {i}' for i in range(10)]}), source)
    payload = source.getvalue()
    metadata = json.dumps({'parquet_files': [{'config': 'default', 'split': 'train',
        'url': 'https://huggingface.co/datasets/test/stories/resolve/main/data.parquet',
        'size': len(payload)}]}).encode()
    calls = []
    monkeypatch.setattr('modelchef.bulk_data.viewer_headers', dict)
    monkeypatch.setattr('modelchef.bulk_data.viewer_page', lambda *args: metadata)
    monkeypatch.setattr('modelchef.bulk_data.build_opener', lambda *args: SimpleNamespace(
        open=lambda *args, **kw: calls.append('download') or io.BytesIO(payload)))
    config = RunConfig(data={'provider': 'hf_parquet', 'dataset': 'test/stories', 'subset': 'default',
                            'offset': 2, 'max_documents': 3})
    rows, manifest = acquire(config, tmp_path, lambda msg: None)
    assert [row['text'] for row in rows] == ['story 2', 'story 3', 'story 4']
    assert manifest['files'][0]['sha256']
    assert acquire(config, tmp_path, lambda msg: None)[0] == rows
    assert calls == ['download']
    config.data.max_download_bytes = 10
    with pytest.raises(ValueError, match='byte cap'):
        acquire(config, tmp_path, lambda msg: None)


def test_bulk_guest_dependencies_use_pipeline_extra():
    from modelchef.aws_guest import guest_plan
    from modelchef.run_config import RunConfig

    assert 'pipeline' in guest_plan(RunConfig(data={'provider': 'hf_parquet'}))['extras']
