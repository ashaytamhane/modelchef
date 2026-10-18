"""Prefetch overlaps required work without changing selection or byte limits."""
import io
import json
import threading

import pytest

from modelchef.bulk_data import acquire_parquet
from modelchef.corpus import digest
from modelchef.run_config import RunConfig


@pytest.fixture
def sources(monkeypatch):
    pa = pytest.importorskip('pyarrow')
    import pyarrow.parquet as pq
    payloads, files, calls = [], [], []
    for index in range(3):
        stream = io.BytesIO()
        pq.write_table(pa.table({'text': [f'article {2 * index}', f'article {2 * index + 1}']}), stream)
        payloads.append(stream.getvalue())
        files.append({'config': 'default', 'split': 'train',
                      'url': f'https://huggingface.co/datasets/test/articles/resolve/main/{index}.parquet',
                      'size': len(payloads[-1])})
    metadata = json.dumps({'parquet_files': files}).encode()
    monkeypatch.setattr('modelchef.bulk_data.viewer_headers', dict)
    monkeypatch.setattr('modelchef.bulk_data.viewer_page', lambda *args: metadata)

    def transfer(url, path, size, timeout, headers, explain):
        index = int(url.rsplit('/', 1)[1].split('.')[0])
        calls.append(index)
        path.write_bytes(payloads[index])
        return {'url': url, 'size': size, 'sha256': digest(payloads[index])}

    monkeypatch.setattr('modelchef.bulk_data.download_file', transfer)
    config = RunConfig(data={'provider': 'hf_parquet', 'dataset': 'test/articles',
                            'subset': 'default', 'max_documents': 6}).data
    return config, calls, transfer, metadata, payloads


@pytest.mark.parametrize('offset,cap,expected_files', [
    (0, 1, [0]), (0, 2, [0]), (1, 2, [0, 1]), (2, 3, [0, 1, 2]), (0, 6, [0, 1, 2]),
])
@pytest.mark.parametrize('use_sink', [False, True])
def test_prefetch_preserves_row_order_offsets_caps_and_manifests(
        tmp_path, sources, offset, cap, expected_files, use_sink):
    config, calls, _, metadata, payloads = sources
    config.offset, config.max_documents = offset, cap
    received = []
    rows, used, manifest = acquire_parquet(config, tmp_path, lambda _: None,
                                          received.append if use_sink else None)
    if use_sink:
        assert rows == cap
        rows = received
    assert rows == [{'text': f'article {i}', 'source_index': i} for i in range(offset, offset + cap)]
    assert calls == expected_files
    assert used == len(metadata) + sum(len(payloads[i]) for i in expected_files)
    assert [entry['sha256'] for entry in manifest['files']] == [
        digest(payloads[i]) for i in expected_files]


def test_next_download_overlaps_current_file_consumption(tmp_path, sources, monkeypatch):
    config, _, transfer, _, _ = sources
    downloading, consuming = threading.Event(), threading.Event()

    def concurrent_transfer(url, *args):
        if url.endswith('/1.parquet'):
            downloading.set()
            assert consuming.wait(5), 'Parsing did not overlap the next download'
        return transfer(url, *args)

    def sink(row):
        if row['source_index'] == 0:
            assert downloading.wait(5), 'Next file was not prefetched'
            consuming.set()

    monkeypatch.setattr('modelchef.bulk_data.download_file', concurrent_transfer)
    count, _, _ = acquire_parquet(config, tmp_path, lambda _: None, sink)
    assert count == 6


def test_prefetch_cannot_exceed_approved_source_byte_budget(tmp_path, sources):
    config, calls, _, metadata, payloads = sources
    config.max_download_bytes = len(metadata) + len(payloads[0])
    with pytest.raises(ValueError, match='max_download_bytes'):
        acquire_parquet(config, tmp_path, lambda _: None)
    assert calls == [0]


def test_prefetched_download_failure_is_propagated(tmp_path, sources, monkeypatch):
    config, _, transfer, _, _ = sources

    def fail(url, *args):
        if url.endswith('/1.parquet'):
            raise ConnectionError('simulated transfer failure')
        return transfer(url, *args)

    monkeypatch.setattr('modelchef.bulk_data.download_file', fail)
    received = []
    with pytest.raises(ConnectionError, match='simulated'):
        acquire_parquet(config, tmp_path, lambda _: None, received.append)
    assert [row['source_index'] for row in received] == [0, 1]


@pytest.mark.parametrize('cap', [2, 6])
def test_prefetch_validates_required_urls_and_ignores_unneeded_files(
        tmp_path, sources, monkeypatch, cap):
    config, calls, _, metadata, _ = sources
    config.max_documents = cap
    invalid = json.loads(metadata)
    invalid['parquet_files'][1]['url'] = 'https://unapproved.example/file.parquet'
    monkeypatch.setattr('modelchef.bulk_data.viewer_page', lambda *args: json.dumps(invalid).encode())
    if cap == 2:
        assert len(acquire_parquet(config, tmp_path, lambda _: None)[0]) == 2
    else:
        with pytest.raises(ValueError, match='Unexpected Parquet download URL'):
            acquire_parquet(config, tmp_path, lambda _: None)
    assert calls == [0]
