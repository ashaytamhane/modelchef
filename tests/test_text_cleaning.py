import hashlib
import json
import os
from collections import Counter

import pytest

from modelchef.corpus import clean_record, process
from modelchef.run_config import RunConfig
from modelchef.text_cleaning import clean_jsonl, collapse_duplicate_sentences

SENTENCE = 'Water evaporates from the surface into the air.'


def test_runs_preserve_first_sentence_and_intervening_content():
    text = SENTENCE + '  ' + SENTENCE + '\n' + SENTENCE + ' Birds fly. ' + SENTENCE
    cleaned, removed, chars = collapse_duplicate_sentences(text)
    assert cleaned == SENTENCE + '  Birds fly. ' + SENTENCE
    assert removed == 2 and chars > 0
    assert collapse_duplicate_sentences(cleaned)[1] == 0


@pytest.mark.parametrize('text', [
    'Very very cold. Very very cold.',
    'The result is 3.14 and Dr. Smith agrees. The result is 3.15 and Dr. Smith agrees.',
    'Values are 1. 2. 3. Values are 1. 2. 3.',
    SENTENCE + ' Water evaporates from the surface into the air!',
    SENTENCE + ' water evaporates from the surface into the air.',
    SENTENCE + ' Why? ' + SENTENCE,
    'This fragment has several words without ending punctuation',
])
def test_conservative_matching_preserves_valid_text(text):
    assert collapse_duplicate_sentences(text) == (text, 0, 0)


def test_whitespace_quotes_and_abbreviations():
    text = '“' + SENTENCE + '” ' + '“' + SENTENCE.replace(' ', '  ') + '”'
    assert collapse_duplicate_sentences(text)[1] == 1
    text = 'Dr. Smith studies water and its physical properties.'
    assert collapse_duplicate_sentences(text + ' ' + text)[0].strip() == text


def test_streaming_rewrite_metadata_hashes_and_document_boundaries(tmp_path):
    source, dest = tmp_path/'train.jsonl', tmp_path/'cleaned.jsonl'
    rows = [{'text': SENTENCE + ' ' + SENTENCE, 'source_index': 9,
             'split_group': 'original-train', 'sha256': 'old'},
            {'text': SENTENCE, 'source_index': 10}]
    original = ''.join(json.dumps(row) + '\n' for row in rows).encode()
    source.write_bytes(original)
    report = clean_jsonl(source, dest)
    cleaned = [json.loads(line) for line in dest.read_text().splitlines()]
    assert source.read_bytes() == original
    assert len(cleaned) == 2 and cleaned[1] == rows[1]
    assert cleaned[0]['text'].strip() == SENTENCE
    assert cleaned[0]['source_index'] == 9 and cleaned[0]['split_group'] == 'original-train'
    assert cleaned[0]['sha256'] == hashlib.sha256(cleaned[0]['text'].encode()).hexdigest()
    assert report['counts']['sentences_removed'] == 1
    assert report['output_sha256'] == hashlib.sha256(dest.read_bytes()).hexdigest()
    assert report['source_sha256'] == hashlib.sha256(original).hexdigest()
    with pytest.raises(ValueError, match='exists'):
        clean_jsonl(source, dest)


def test_atomic_inplace_preserves_other_hardlinks(tmp_path):
    source, linked = tmp_path/'train.jsonl', tmp_path/'protected.jsonl'
    original = json.dumps({'text': SENTENCE + ' ' + SENTENCE}) + '\n'
    source.write_text(original)
    try:
        os.link(source, linked)
    except OSError:
        pytest.skip('Filesystem does not support hard links')
    clean_jsonl(source, source, overwrite=True)
    assert linked.read_text() == original
    assert json.loads(source.read_text())['text'].strip() == SENTENCE


@pytest.mark.parametrize('bad', [b'{bad-json}\n', b'{"text": 10}\n', b'x'*500])
def test_failure_keeps_source_and_destination_and_removes_temporary(tmp_path, bad):
    source, dest = tmp_path/'input.jsonl', tmp_path/'output.jsonl'
    source.write_bytes(bad)
    dest.write_bytes(b'previous-output')
    with pytest.raises((ValueError, TypeError)):
        clean_jsonl(source, dest, overwrite=True, max_record_bytes=100)
    assert source.read_bytes() == bad and dest.read_bytes() == b'previous-output'
    assert not list(tmp_path.glob('.sentence-clean-*'))


def test_filter_default_off_and_reports_removed_sentences():
    config = RunConfig(processing={'min_chars': 1, 'pii': 'off'})
    row = {'text': SENTENCE + ' ' + SENTENCE, 'source_index': 0}
    assert clean_record(row, config, set(), Counter())['text'] == row['text']
    config.processing.collapse_consecutive_duplicate_sentences = True
    counts = Counter()
    cleaned = clean_record(row, config, set(), counts)
    assert cleaned['text'].strip() == SENTENCE
    assert counts['duplicate_sentences_removed'] == 1
    assert counts['sentence_cleanup_documents'] == 1


@pytest.mark.parametrize('storage,exact', [('memory', False), ('disk', False), ('disk', True)])
def test_processing_integration_and_counts(tmp_path, monkeypatch, storage, exact):
    monkeypatch.setenv('MODELCHEF_PREPROCESS_WORKERS', '1')
    config = RunConfig(data={'storage': storage, 'provider': 'local_jsonl', 'local_path': 'unused'},
        processing={'min_chars': 1, 'pii': 'off',
        'exact_dedup': exact, 'near_dedup': False, 'collapse_consecutive_duplicate_sentences': True})
    records = [{'text': f'Topic {i} explores the world around us. ' * 3, 'source_index': i}
               for i in range(12)]
    rows, report = process(iter(records), config, tmp_path)
    assert report['counts']['duplicate_sentences_removed'] == 24
    assert sum(len(v) for v in rows.values()) == 12
    for group in rows.values():
        for row in group:
            assert row['text'].count('explores') == 1


def test_invalid_settings_and_structured_stage_rejected():
    with pytest.raises(ValueError):
        RunConfig(processing={'minimum_sentence_words': 0})
    with pytest.raises(ValueError, match='raw-text'):
        RunConfig(stage={'kind': 'sft', 'parent_run': 'parent'},
                  processing={'collapse_consecutive_duplicate_sentences': True})


def test_cli_and_inplace_failure(tmp_path, monkeypatch, capsys):
    from modelchef.cli import main
    source, dest = tmp_path/'train.jsonl', tmp_path/'cleaned.jsonl'
    source.write_text(json.dumps({'text': SENTENCE + ' ' + SENTENCE}) + '\n')
    monkeypatch.setattr('sys.argv', ['modelchef', 'clean-text', str(source), '--output', str(dest)])
    main()
    report = json.loads(capsys.readouterr().out)
    assert report['counts']['sentences_removed'] == 1
    original = dest.read_bytes()
    clean_jsonl(dest, dest, overwrite=True)
    assert dest.read_bytes() == original
    source.write_bytes(b'{broken-json}\n')
    with pytest.raises(ValueError):
        clean_jsonl(source, source, overwrite=True)
    assert source.read_bytes() == b'{broken-json}\n'
