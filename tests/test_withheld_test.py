"""A tuning run must never score its reserved test split."""
import json
from pathlib import Path

import pytest

pytest.importorskip('torch')

from modelchef import pipeline
from modelchef.run_config import RunConfig


def test_tuning_run_withholds_test_scoring(tmp_path, monkeypatch):
    import torch
    torch.set_num_threads(1)
    source = tmp_path / 'corpus.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Lesson {i} explains observations and scientific evidence. ' * 8}) + '\n' for i in range(40)))
    names = {}
    original_process, original_pack = pipeline.process, pipeline.pack
    original_evaluate, original_stage = pipeline.evaluate, pipeline.evaluate_stage

    def process(*args):
        splits, info = original_process(*args)
        names.update({id(rows): name for name, rows in splits.items()})
        return splits, info

    def pack(rows, *args):
        pair, count = original_pack(rows, *args)
        pair[0]._split_name = names[id(rows)]
        return pair, count

    def evaluate(model, blocks, *args):
        assert blocks[0]._split_name != 'test', 'Test scoring leaked into tuning'
        return original_evaluate(model, blocks, *args)

    def stage(model, reference, blocks, *args):
        assert blocks[0]._split_name != 'test', 'Stage-specific test scoring leaked'
        return original_stage(model, reference, blocks, *args)

    monkeypatch.setattr(pipeline, 'process', process)
    monkeypatch.setattr(pipeline, 'pack', pack)
    monkeypatch.setattr(pipeline, 'evaluate', evaluate)
    monkeypatch.setattr(pipeline, 'evaluate_stage', stage)
    config = RunConfig(name='withheld-test', explain='brief',
        data={'provider':'local_jsonl','local_path':str(source),'max_tokens':4000},
        processing={'min_chars':1,'near_dedup':False}, tokenizer={'vocab_size':300},
        model={'layers':1,'hidden_size':16,'attention_heads':2,'kv_heads':1,'intermediate_size':32,'context_length':8},
        training={'accelerator':'cpu','precision':'fp32','steps':2,'warmup_steps':0,'micro_batch_size':2,'gradient_accumulation':1},
        evaluation={'benchmark':'none','evaluate_test':False,'batch_size':1,'batches':1,'generate_tokens':1})
    result = pipeline.run_pipeline(config, tmp_path)
    assert result['test'] is None and result['test_evaluation'] == 'withheld'
    assert 'test' not in result['stage_evaluation']
    assert result['checkpoint_recovery'] == 'passed'
    assert result['validation']['evaluated_examples'] == 1
    saved = json.loads((Path(result['run_dir']) / 'report.json').read_text())
    assert saved['test'] is None
