"""Exercise vLLM using tiny local random weights; no pretrained model download."""

import json
import os
import tempfile
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["VLLM_NO_USAGE_STATS"] = "1"


def main():
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    from vllm import LLM, SamplingParams

    torch.manual_seed(42)
    with tempfile.TemporaryDirectory() as directory:
        model_path = Path(directory) / "model"
        vocabulary = {"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "hello": 3, "world": 4}
        vocabulary.update({f"token{i}": i for i in range(5, 32)})
        tokenizer = Tokenizer(WordLevel(vocabulary, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        fast = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]",
                                      bos_token="[BOS]", eos_token="[EOS]")
        model = LlamaForCausalLM(LlamaConfig(
            vocab_size=32, hidden_size=128, intermediate_size=256,
            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
            max_position_embeddings=128, bos_token_id=1, eos_token_id=2))
        model.save_pretrained(model_path)
        fast.save_pretrained(model_path)
        del model
        engine = LLM(model=str(model_path), tokenizer=str(model_path),
                     dtype="float16", max_model_len=128, max_num_seqs=1,
                     gpu_memory_utilization=0.3, enforce_eager=True)
        results = engine.generate(["hello world"], SamplingParams(
            temperature=0, max_tokens=4, ignore_eos=True))
        token_ids = results[0].outputs[0].token_ids
        if len(token_ids) != 4:
            raise RuntimeError("vLLM did not generate the expected number of tokens")
        print(json.dumps({"vllm_generation": "passed", "generated_tokens": len(token_ids),
                          "model": "tiny random Llama", "downloaded_weights": False}))


if __name__ == "__main__":
    main()
