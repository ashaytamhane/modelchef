import json

import pytest
import yaml

from modelchef.corpus import file_digest
from scripts.compose_corpus import compose


def test_composition_is_capped_attributed_and_preserves_unicode_jsonl(tmp_path):
    source = tmp_path/'data/ingredient'
    source.mkdir(parents=True)
    data = source/'text.jsonl'
    records = [{'source': 'example', 'text': f'User: Question {i}\nAssistant: <think>\u2028Work.</think> Answer {i}'}
               for i in range(5)]
    data.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    manifest = source/'attribution.json'
    manifest.write_text(json.dumps({'sources': [{'license': 'apache-2.0', 'dataset': 'author/data'}],
        'collections': [{'file': 'text.jsonl', 'sha256': file_digest(data)}]}))
    recipe = tmp_path/'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'name': 'test', 'output_dir': 'data/composed', 'seed': 42,
        'inputs': [{'manifest': str(manifest), 'collection': 'text.jsonl'}],
        'max_examples_by_source': {'example': 3}}))
    result = compose(recipe, tmp_path)
    assert result['source_counts'] == {'example': 3}
    assert result['sources'][0]['license'] == 'apache-2.0'
    assert compose(recipe, tmp_path) == result
    with (tmp_path/'data/composed/corpus.jsonl').open(encoding='utf-8') as stream:
        selected = [json.loads(line) for line in stream]
    assert len(selected) == 3 and all('\u2028' in r['text'] for r in selected)
    (tmp_path/'data/composed/corpus.jsonl').write_text('tampered')
    with pytest.raises(ValueError, match='differs'):
        compose(recipe, tmp_path)


def test_structured_composition_preserves_conversation_groups(tmp_path):
    folder = tmp_path / 'data/input'
    folder.mkdir(parents=True)
    data = folder / 'pairs.jsonl'
    data.write_text(''.join(json.dumps({'source': 'dialogue', 'prompt': f'Question {i}',
        'response': f'Reply {i}', 'split_group': 'same-conversation'}) + '\n' for i in range(3)))
    manifest = folder / 'attribution.json'
    manifest.write_text(json.dumps({'sources': [], 'collections': [
        {'file': 'pairs.jsonl', 'sha256': file_digest(data)}]}))
    recipe = tmp_path / 'recipe.yaml'
    recipe.write_text(yaml.safe_dump({'name': 'test', 'seed': 42, 'output_dir': 'data/out',
        'inputs': [{'manifest': str(manifest), 'collection': 'pairs.jsonl'}],
        'max_examples_by_source': {'dialogue': 3}}))
    compose(recipe, tmp_path)
    rows = [json.loads(line) for line in (tmp_path/'data/out/corpus.jsonl').read_text().splitlines()]
    assert len(rows) == 3
    assert all(r['split_group'] == 'same-conversation' and '\nAssistant: Reply' in r['text'] for r in rows)
