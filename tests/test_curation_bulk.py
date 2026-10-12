import json

from modelchef.curation import CurationConfig, Ingredient
from modelchef.curation_bulk import curate_bulk


def test_bulk_curation_counts_oversized_source_records_without_aborting(tmp_path, monkeypatch):
    source = Ingredient(name='text', dataset='author/data', adapter='text', provider='hf_parquet',
        collection='text', license='apache-2.0', license_url='https://example.org/license',
        attribution='Authors', citations=[{'title': 'Paper', 'authors': ['Author'],
                                          'url': 'https://example.org/paper'}])
    rows = [{'row_idx': 0, 'row': {'text': 'x' * 2_000_001}}] + [
        {'row_idx': i, 'row': {'text': f'Example {i} about scientific reasoning.'}} for i in range(1, 4)]
    monkeypatch.setattr('modelchef.curation_bulk.parquet_rows',
                        lambda *args: iter([(rows, 100, [])]))
    c = CurationConfig(name='test', output_dir='data/test', sources=[source])
    result = curate_bulk(c, tmp_path, lambda message: None)
    assert result['sources'][0]['skipped']['oversized_source_record'] == 1
    assert result['sources'][0]['source_rows'] == 4
    assert result['sources'][0]['accepted_examples'] == 3
    assert len([json.loads(line) for line in (tmp_path/'data/test/text.jsonl').read_text().splitlines()]) == 3
