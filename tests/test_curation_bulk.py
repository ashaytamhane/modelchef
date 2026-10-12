import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from modelchef.curation import curate
from modelchef.curation_bulk import parquet_rows


def test_bulk_metadata_respects_remaining_budget(tmp_path, monkeypatch):
    source = SimpleNamespace(dataset='owner/data', subset='default', split='train')
    limits = []

    def metadata(request, timeout, limit, progress):
        limits.append(limit)
        return b'x' * limit

    monkeypatch.setattr('modelchef.curation_bulk.viewer_headers', dict)
    monkeypatch.setattr('modelchef.curation_bulk.viewer_page', metadata)
    with pytest.raises(ValueError, match='remaining download budget'):
        list(parquet_rows(source, tmp_path, 100, 10, lambda _: None))
    assert limits == [101]
    with pytest.raises(ValueError, match='exhausted'):
        list(parquet_rows(source, tmp_path, 0, 10, lambda _: None))
    assert limits == [101]


def test_bulk_mixture_replay_and_groups(tmp_path, monkeypatch):
    original = yaml.safe_load((Path(__file__).parents[1] /
        'configs/data-assistant-persona-sample.yaml').read_text())
    source = original['sources'][0]
    source.update(provider='hf_parquet', max_rows=100000, max_example_bytes=24000)
    original.update(output_dir='data/raw/bulk-test', sources=[source])
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(original))

    def batches(*args):
        yield [{'row_idx': i, 'row': {'text': f'This is a complete educational document {i}.'}}
               for i in range(8)], 1024, [{'sha256': 'pinned-shard', 'size': 1024}]

    monkeypatch.setattr('modelchef.curation_bulk.parquet_rows', batches)
    result = curate(config, tmp_path, execute=True, progress=lambda _: None)
    assert result['collections'][0]['examples'] == 8
    rows = [json.loads(line) for line in (tmp_path / 'data/raw/bulk-test/foundation.jsonl').read_text().splitlines()]
    assert len({row['split_group'] for row in rows}) == 8
    assert result['sources'][0]['parquet_files'][0]['sha256'] == 'pinned-shard'
    monkeypatch.setattr('modelchef.curation_bulk.parquet_rows', lambda *args: (_ for _ in ()).throw(
        AssertionError('Verified replay must not download')))
    assert curate(config, tmp_path, execute=True, progress=lambda _: None) == result
