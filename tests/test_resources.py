from collections import Counter

import pytest

from modelchef import resources


def test_cpu_quota_limits_host_and_affinity(tmp_path, monkeypatch):
    monkeypatch.setattr(resources.os, 'cpu_count', lambda: 128)
    monkeypatch.setattr(resources.os, 'sched_getaffinity', lambda _: set(range(48)), raising=False)
    monkeypatch.setattr(resources.os, 'process_cpu_count', lambda: 48, raising=False)
    (tmp_path / 'cpu.max').write_text('2720000 100000')
    result = resources.cpu_resources(tmp_path)
    assert result['usable_cpus'] == 27.2
    (tmp_path / 'cpu.max').write_text('50000 100000')
    assert resources.cpu_resources(tmp_path)['usable_cpus'] == .5


def test_workers_respect_memory_quota_and_override(monkeypatch):
    monkeypatch.setattr(resources, 'cpu_resources', lambda: {'usable_cpus': 27.2})
    monkeypatch.setattr(resources, 'available_memory', lambda: 2 * 1024**3)
    monkeypatch.delenv('MODELCHEF_PREPROCESS_WORKERS', raising=False)
    assert resources.preparation_plan()['workers'] == 4
    monkeypatch.setenv('MODELCHEF_PREPROCESS_WORKERS', '2')
    assert resources.preparation_plan()['workers'] == 2
    monkeypatch.setenv('MODELCHEF_PREPROCESS_WORKERS', '0')
    with pytest.raises(ValueError):
        resources.preparation_plan()


def test_parallel_cleaning_matches_sequential_order_and_counts(monkeypatch):
    from modelchef.preparation_workers import cleaned_records
    from modelchef.run_config import RunConfig
    config = RunConfig(processing={'min_chars': 10, 'exact_dedup': False, 'near_dedup': False})
    rows = [{'text': f'Document {i} has useful text and punctuation.', 'source_index': i}
            for i in range(150)] + [{'text': '', 'source_index': 150}]
    results = []
    for workers in [1, 2]:
        monkeypatch.setenv('MODELCHEF_PREPROCESS_WORKERS', str(workers))
        counts = Counter()
        results.append((list(cleaned_records(rows, config, set(), counts)), counts))
    assert results[0] == results[1]
    assert results[1][1]['input'] == 151


@pytest.mark.parametrize('packing', ['concatenate', 'document'])
def test_parallel_tokens_match_bytes_metadata_and_limit(tmp_path, monkeypatch, packing):
    from tokenizers import Tokenizer, models, pre_tokenizers

    from modelchef.disk_data import pack_disk
    from modelchef.run_config import Tokenization
    tokenizer = Tokenizer(models.WordLevel({'<unk>': 0, '<pad>': 1, '<bos>': 2,
                                           '<eos>': 3, 'hello': 4, 'world': 5}, unk_token='<unk>'))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenization = Tokenization(packing=packing, add_bos=True, append_eos=True)
    rows = [{'text': 'hello world'} for _ in range(150)]
    results = []
    for workers in [1, 2]:
        monkeypatch.setenv('MODELCHEF_PREPROCESS_WORKERS', str(workers))
        folder = tmp_path / str(workers)
        folder.mkdir()
        _, metadata = pack_disk(rows, tokenizer, tokenization, 8, 200, folder, 'train')
        results.append(((folder / 'tokens-train.bin').read_bytes(), metadata))
    assert results[0] == results[1]
