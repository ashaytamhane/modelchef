import io
import json
import tarfile
from pathlib import Path
from types import SimpleNamespace

import yaml

from modelchef.aws_training import build_bundle
from modelchef.objectives import pack_structured
from modelchef.pipeline import train_tokenizer
from modelchef.run_config import RunConfig, Tokenization


def test_bpe_fitting_document_limit_does_not_consume_full_corpus():
    def documents():
        for i in range(3):
            yield {'text': f'Training document {i} contains varied words.'}
        raise AssertionError('Tokenizer fitting exceeded its configured sample')
    tokenizer = train_tokenizer(documents(), Tokenization(vocab_size=260, training_max_documents=3))
    assert tokenizer.get_vocab_size() >= 260


def test_context_overflow_skips_whole_targets_and_reports_count():
    class Tokenizer:
        def encode(self, text, **kwargs):
            return SimpleNamespace(ids=[2] * (100 if 'oversized' in text else 2))

        def token_to_id(self, token):
            return 1 if token == '<eos>' else 0

    config = RunConfig.model_validate({'model': {'context_length': 8},
        'data': {'provider': 'local_jsonl', 'local_path': 'sample.jsonl'},
        'stage': {'kind': 'sft', 'parent_run': 'parent', 'oversized_policy': 'skip'}})
    messages = []
    blocks, tokens = pack_structured([{'prompt': 'oversized', 'response': 'whole target'},
        {'prompt': 'short', 'response': 'complete'}], Tokenizer(), config, 100, messages.append)
    assert len(blocks[0]) == 1 and tokens == 5
    assert 'skipped 1 oversized' in messages[0]
    assert blocks[1][0].tolist() == [-100, 2, 2, 1, -100, -100, -100, -100]


def test_full_workflow_packages_curation_without_local_full_dataset(tmp_path):
    root = tmp_path
    source_root = Path(__file__).parents[1]
    (root / 'curation.yaml').write_text(yaml.safe_dump({
        'name': 'fixture', 'output_dir': 'data/raw/fixture', 'max_download_bytes': 100000000000,
        'sources': [{'name': 'fixture', 'dataset': 'example/corpus', 'adapter': 'text',
                     'collection': 'foundation', 'license': 'test-only',
                     'license_url': 'https://example.com/license', 'attribution': 'Fixture author',
                     'citations': [{'title': 'Fixture source', 'authors': ['Fixture author'],
                                    'url': 'https://example.com/source'}]}]}))
    stage = RunConfig(data={'provider': 'local_jsonl',
                           'local_paths': ['data/raw/fixture/foundation.jsonl']})
    (root / 'stage.yaml').write_text(yaml.safe_dump(stage.model_dump(mode='json')))
    (root / 'workflow.yaml').write_text(yaml.safe_dump({'name': 'fixture',
        'curation_recipe': 'curation.yaml', 'stages': [{'name': 'foundation', 'recipe': 'stage.yaml'}]}))
    bundle = build_bundle(root, {'pipeline_config': 'workflow.yaml', '_source_root': source_root})
    with tarfile.open(fileobj=io.BytesIO(bundle)) as archive:
        workflow = yaml.safe_load(archive.extractfile('configs/job.yaml').read())
        assert workflow['curation_recipe'] == 'configs/curation.yaml'
        curation = yaml.safe_load(archive.extractfile('configs/curation.yaml').read())
        assert curation['max_download_bytes'] == 100_000_000_000
        stage = yaml.safe_load(archive.extractfile('inputs/stages/foundation/recipe.yaml').read())
        assert stage['data']['local_paths'][0].startswith('data/raw/fixture/')
        assert not any(name.endswith('data.jsonl') for name in archive.getnames())
        # Sources and credentials never enter a config's lineage as a fake pretrained parent.
        assert json.loads(json.dumps(stage))['stage']['kind'] == 'pretrain'
