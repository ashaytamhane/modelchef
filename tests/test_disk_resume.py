import json
from pathlib import Path

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('tokenizers')

from modelchef.pipeline import run_pipeline
from modelchef.run_config import RunConfig


def test_disk_resume_skips_processing_and_reproduces_weights(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    source = tmp_path / 'data.jsonl'
    source.write_text(''.join(json.dumps({'text': f'Topic {i}: students study science. ' * 8})
                            + '\n' for i in range(30)))
    c = RunConfig(name='disk', explain='brief',
        data={'provider': 'local_jsonl', 'local_path': str(source), 'storage': 'disk', 'max_tokens': 6000},
        processing={'min_chars': 1, 'near_dedup': False}, tokenizer={'vocab_size': 300},
        model={'layers': 1, 'hidden_size': 32, 'attention_heads': 4, 'kv_heads': 2,
               'intermediate_size': 64, 'context_length': 32},
        training={'accelerator': 'cpu', 'precision': 'fp32', 'steps': 2, 'warmup_steps': 0,
                  'micro_batch_size': 1, 'gradient_accumulation': 1, 'checkpoint_every': 1,
                  'checkpoint_at_start': True},
        evaluation={'benchmark': 'none', 'batches': 1, 'generate_tokens': 1})
    result = run_pipeline(c, tmp_path)
    original = torch.load(result['checkpoint'], weights_only=True)['model']
    assert (Path(result['run_dir']) / 'checkpoints/step-000000.pt').is_file()

    def forbidden(*args, **kwargs):
        pytest.fail('Resume must not filter or fit a tokenizer again')

    monkeypatch.setattr('modelchef.pipeline.process', forbidden)
    monkeypatch.setattr('modelchef.pipeline.train_tokenizer', forbidden)
    checkpoint = Path(result['run_dir']) / 'checkpoints/step-000001.pt'
    resumed = run_pipeline(c, tmp_path, checkpoint)
    restored = torch.load(resumed['checkpoint'], weights_only=True)['model']
    assert all(torch.equal(original[key], restored[key]) for key in original)
    fresh = c.model_copy(deep=True)
    fresh.name = 'fresh'
    fresh.training.seed = 19
    fresh.data.prepared_run = result['run_dir']
    monkeypatch.setattr('modelchef.pipeline.acquire', forbidden)
    monkeypatch.setattr('modelchef.pipeline.pack_disk', forbidden)
    monkeypatch.setenv('MODELCHEF_LOCAL_CHECKPOINT_KEEP', '1')
    reused = run_pipeline(fresh, tmp_path)
    assert len(list((Path(reused['run_dir']) / 'checkpoints').glob('*.pt'))) == 1
    different = torch.load(reused['checkpoint'], weights_only=True)['model']
    assert any(not torch.equal(original[key], different[key]) for key in original)
    token_file = Path(result['run_dir']) / 'tokens-train.bin'
    with token_file.open('r+b') as target:
        target.write(b'\xff\xff\xff\xff')
    with pytest.raises(ValueError, match='checksum/size changed'):
        run_pipeline(c, tmp_path, checkpoint)
