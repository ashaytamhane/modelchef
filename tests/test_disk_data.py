import json
from pathlib import Path

import pytest

from modelchef.corpus import acquire, process
from modelchef.run_config import RunConfig


def recipe(paths):
    return RunConfig(data={'provider': 'local_jsonl', 'local_paths': paths,
                           'storage': 'disk', 'max_documents': 30, 'max_tokens': 5000},
        processing={'min_chars': 20, 'near_dedup': False}, tokenizer={'vocab_size': 300},
        model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4, 'kv_heads': 2,
               'intermediate_size': 64, 'context_length': 16, 'residual_dropout': 0.1},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 4, 'warmup_steps': 1,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 2},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 2})


def shards(root):
    paths = []
    for shard in range(2):
        path = root / f'shard-{shard}.jsonl'
        path.write_text(''.join(json.dumps({'text': f'Lesson {index}: ' +
            (f'Students study topic{index} and practice reading with teachers. ' * 8)}) + '\n'
            for index in range(shard * 15, (shard + 1) * 15)), encoding='utf-8')
        paths.append(str(path))
    return paths


def test_disk_shards_deterministic_disjoint_splits_and_budget(tmp_path):
    config = recipe(shards(tmp_path))
    rows, manifest = acquire(config, tmp_path, lambda _: None)
    assert len(rows) == 30 and manifest['storage'] == 'disk'
    first, info = process(rows, config, tmp_path)
    second, _ = process(rows, config, tmp_path)
    assert info['counts']['accepted'] == 30
    assert {k: list(v) for k, v in first.items()} == {k: list(v) for k, v in second.items()}
    identities = [{r['sha256'] for r in first[k]} for k in ['train', 'validation', 'test']]
    assert all(identities) and sum(map(len, identities)) == len(set.union(*identities))
    config.data.max_record_bytes = 20
    with pytest.raises(ValueError, match='record exceeds'):
        acquire(config, tmp_path, lambda _: None)
    config.data.max_record_bytes = 4000000
    config.data.max_download_bytes = 20
    with pytest.raises(ValueError, match='byte budget'):
        acquire(config, tmp_path, lambda _: None)


def test_disk_exact_and_near_duplicates(tmp_path):
    pytest.importorskip('numpy')
    config = recipe(shards(tmp_path))
    config.processing.near_dedup = True
    config.processing.near_dedup_jaccard = 0.8
    records = [{'text': ' '.join(f'word{i}' for i in range(100)), 'source_index': 0},
               {'text': ' '.join(f'word{i}' for i in range(100)), 'source_index': 1},
               {'text': ' '.join(f'word{i}' for i in range(99)) + ' changed', 'source_index': 2}]
    rows, _ = acquire(config, tmp_path, lambda _: None)
    _, report = process(records + list(rows), config, tmp_path)
    assert report['counts']['exact_duplicates'] == 1
    assert report['counts']['near_duplicates'] >= 1


def test_disk_pipeline_exact_resume_and_token_tamper(tmp_path):
    torch = pytest.importorskip('torch')
    pytest.importorskip('tokenizers')
    from modelchef.disk_data import load_blocks
    from modelchef.pipeline import run_pipeline
    torch.set_num_threads(1)
    config = recipe(shards(tmp_path))
    result = run_pipeline(config, tmp_path)
    assert result['status'] == 'completed'
    folder = Path(result['run_dir'])
    final = torch.load(result['checkpoint'], weights_only=True)['model']
    resumed = run_pipeline(config, tmp_path, folder / 'checkpoints/step-000002.pt')
    restored = torch.load(resumed['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(final[k], restored[k]) for k in final)
    blocks = load_blocks(folder)
    assert blocks['train'][0][torch.tensor([0, 1])].shape == (2, 16)
    token_file = folder / 'tokens-train.bin'
    with token_file.open('r+b') as stream:
        stream.write(b'xxxx')
    with pytest.raises(ValueError, match='checksum'):
        load_blocks(folder)
