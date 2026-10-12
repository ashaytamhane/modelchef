import sys
from types import SimpleNamespace

import pytest

from modelchef.run_config import RunConfig


@pytest.mark.parametrize('interactive', [False, True])
@pytest.mark.parametrize('transcript', [False, True])
def test_pretraining_chat_respects_declared_format(tmp_path, monkeypatch, interactive, transcript):
    import modelchef.chat as module
    from modelchef import engines

    config = RunConfig(stage={'kind': 'pretrain', 'prompt_format':
                             'chat_transcript' if transcript else 'user_assistant'},
                       tokenizer={'add_bos': True})
    (tmp_path / 'config.json').write_text(config.model_dump_json())
    seen = []

    class Tokenizer:
        def encode(self, text):
            return SimpleNamespace(ids=[ord(c) + 10 for c in text])

        def decode(self, ids):
            return ''.join(chr(i - 10) for i in ids)

        def token_to_id(self, _):
            return 1

    class Engine:
        def __init__(self, **_):
            pass

        def generate(self, prompts, params, **_):
            seen.extend(prompts)
            return [SimpleNamespace(outputs=[SimpleNamespace(text='<think>test</think>answer')])]

    monkeypatch.setattr(engines, 'load_tokenizer', lambda _: Tokenizer())
    monkeypatch.setattr(module, 'export_model', lambda *args: tmp_path)
    monkeypatch.setitem(sys.modules, 'vllm', SimpleNamespace(
        LLM=Engine, SamplingParams=lambda **kwargs: kwargs))
    messages = iter(['1+1', '/exit'])
    monkeypatch.setattr('builtins.input', lambda _: next(messages))
    module.chat(tmp_path, backend='vllm', prompt=None if interactive else '1+1',
                checkpoint='step-000001.pt')
    expected = 'User: 1+1\nAssistant:' if transcript else '1+1'
    assert seen[0]['prompt_token_ids'] == [1] + [ord(c) + 10 for c in expected]
