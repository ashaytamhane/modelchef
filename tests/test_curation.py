import json

import pytest
import yaml

pytest.importorskip('pydantic')
from modelchef.corpus import group_split, process
from modelchef.curation import Ingredient, adapt_rows, curate
from modelchef.run_config import RunConfig


def ingredient(**kwargs):
    return Ingredient(name='example', dataset='author/data', adapter='messages',
        collection='instruction', license='apache-2.0', license_url='https://example.org/license',
        attribution='Original authors', citations=[{'title': 'Original paper',
        'authors': ['Original Author'], 'url': 'https://example.org/paper'}], **kwargs)


def test_reasoning_preserves_complete_trace_and_system_instructions():
    source = ingredient(reasoning='require', min_user_words=3, system_prefix='Reasoning enabled')
    rows = [{'row_idx': 1, 'row': {'chat_template_kwargs': {'custom_instructions': 'Be reserved'},
        'messages': [{'role': 'user', 'content': 'Why might someone be angry?'},
                     {'role': 'assistant', 'content': '<think>Consider context.</think> We cannot be sure.'}]}}]
    records, _ = adapt_rows(rows, source)
    assert records[0]['prompt'] == 'System: Reasoning enabled\nBe reserved\nUser: Why might someone be angry?'
    assert records[0]['response'].endswith('We cannot be sure.')
    for answer in ['<think>unfinished', '<think>Only thought.</think>', 'A direct answer']:
        rows[0]['row']['messages'][1]['content'] = answer
        assert adapt_rows(rows, source)[0] == []


def test_curated_pretraining_text_retains_groups_and_attribution(tmp_path, monkeypatch):
    import modelchef.curation as module
    source = ingredient(pretraining_collection='social-text')
    payload = json.dumps({'rows': [{'row_idx': i, 'row': {'messages': [
        {'role': 'user', 'content': f'Situation {i}'},
        {'role': 'assistant', 'content': '<think>Consider uncertainty.</think> Ask gently.'}]}}
        for i in range(4)]}).encode()
    monkeypatch.setattr(module, 'viewer_headers', dict)
    calls = []
    monkeypatch.setattr(module, 'viewer_page', lambda *args: calls.append(1) or payload)
    config = {'name': 'test', 'output_dir': 'data/sample', 'sources': [source.model_dump()]}
    path = tmp_path / 'recipe.yaml'; path.write_text(yaml.safe_dump(config))
    assert curate(path, tmp_path)['name'] == 'test'
    assert not calls
    manifest = curate(path, tmp_path, True)
    text = [json.loads(line) for line in (tmp_path/'data/sample/social-text.jsonl').read_text().splitlines()]
    assert '<think>Consider uncertainty.</think>' in text[0]['text']
    assert text[0]['split_group'].startswith('example:')
    assert manifest['sources'][0]['citations'][0]['authors'] == ['Original Author']
    curate(path, tmp_path, True)
    assert len(calls) == 1
    (tmp_path/'data/sample/social-text.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='checksum'):
        curate(path, tmp_path, True)


@pytest.mark.parametrize('storage', ['memory', 'disk'])
def test_holdout_groups_stay_fixed_across_pretraining_and_sft(tmp_path, storage):
    c = RunConfig(data={'storage': storage, 'provider': 'local_jsonl', 'local_path': 'unused'},
        processing={'min_chars': 1, 'near_dedup': False,
                    'validation_fraction': .2, 'test_fraction': .2})
    records = [{'text': f'Experience {i}: uncertainty and emotion have contextual causes.',
                'source_index': i, 'split_group': f'conversation:{i//2}'} for i in range(200)]
    splits, info = process(records, c, tmp_path)
    assert info['split_policy'] == 'split_group-disjoint'
    locations = {}
    for split, rows in splits.items():
        for row in rows:
            assert group_split(row['split_group'], c) == split
            assert locations.setdefault(row['split_group'], split) == split
    # Expanded replay corpora keep the same assignment; no corpus-size-dependent shuffle.
    for i in range(100):
        assert group_split(f'conversation:{i}', c) in splits


def test_measurements_preserve_values_and_reject_false_scales():
    source = ingredient().model_copy(update={'adapter': 'measurements', 'text_field': 'word',
        'measurement_fields': {'touch': 'touch'},
        'measurement_description': 'Mean human concept ratings; scale 0 to 5.'})
    rows = [{'row_idx': 0, 'row': {'word': 'ache', 'touch': '1.23456789'}}]
    text = adapt_rows(rows, source)[0][0]['text']
    assert 'touch: 1.23456789' in text and 'human concept ratings' in text
    rows[0]['row']['touch'] = 'nan'
    with pytest.raises(ValueError, match='range'):
        adapt_rows(rows, source)


def test_csv_acquisition_is_bounded_and_stops_after_requested_rows(tmp_path, monkeypatch):
    import io

    import modelchef.curation as module
    source = ingredient().model_copy(update={'provider': 'https_csv',
        'source_url': 'https://example.org/data.csv', 'adapter': 'text', 'max_rows': 3})
    payload = b'text\nfirst complete record\nsecond complete record\nthird complete record\nextra record\n'
    streams = []
    def open_csv(*args, **kwargs):
        stream = io.BytesIO(payload)
        streams.append(stream)
        return stream
    monkeypatch.setattr(module, 'urlopen', open_csv)
    monkeypatch.setattr(module, 'viewer_headers', dict)
    path = tmp_path/'csv.yaml'
    path.write_text(yaml.safe_dump({'name':'csv', 'output_dir':'data/csv',
        'max_download_bytes': 80, 'sources':[source.model_dump()]}))
    result = curate(path, tmp_path, True)
    assert result['sources'][0]['accepted_examples'] == 3
    assert result['response_bytes'] < len(payload)
    assert result['sources'][0]['dataset_url'] == source.source_url
    config = yaml.safe_load(path.read_text());config['output_dir']='data/too-small'
    config['max_download_bytes']=10;path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='cap'):
        curate(path, tmp_path, True)


def test_empathy_adapter_does_not_leak_hidden_scenario_or_emotion():
    source = ingredient().model_copy(update={'adapter': 'empathy_turn'})
    rows = [{'row_idx': i, 'row': {'conv_id': 'one', 'utterance_idx': i+1,
        'speaker_idx': i, 'utterance': text, 'context': 'angry', 'prompt': 'hidden scenario'}}
        for i, text in enumerate(['I cancelled my plans.', 'What happened?'])]
    records, _ = adapt_rows(rows, source)
    assert 'hidden scenario' not in records[0]['prompt']
    assert 'angry' not in records[0]['prompt']
    assert records[0]['response'] == 'What happened?'
