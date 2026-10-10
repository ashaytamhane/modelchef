"""Native and local Hugging Face model/tokenizer adapters; no implicit downloads."""
import json
from pathlib import Path

import torch
from tokenizers import Tokenizer


class TokenizerRoles:
    def __init__(self, tokenizer, roles=None):
        self.tokenizer, self.roles = tokenizer, roles or {}

    def token_to_id(self, token):
        return self.roles.get(token, self.tokenizer.token_to_id(token))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def load_tokenizer(folder):
    path = folder / 'tokenizer-roles.json'
    return TokenizerRoles(Tokenizer.from_file(str(folder / 'tokenizer.json')),
                          json.loads(path.read_text()) if path.exists() else {})


class HFDecoder(torch.nn.Module):
    def __init__(self, c, vocab_size):
        super().__init__()
        from transformers import AutoModelForCausalLM
        self.config = c
        if c.pretrained_config_sha256:
            from ashayllm.corpus import digest
            if digest((Path(c.pretrained_path) / 'config.json').read_bytes()) != c.pretrained_config_sha256:
                raise ValueError('Original pretrained model configuration changed')
        self.model = AutoModelForCausalLM.from_pretrained(c.pretrained_path,
            local_files_only=True, trust_remote_code=False, torch_dtype=torch.float32,
            attn_implementation='sdpa')
        if c.custom_swiglu_path:
            raise ValueError('Custom SwiGLU hooks currently target the native decoder only')
        if self.model.get_input_embeddings().num_embeddings != vocab_size:
            raise ValueError('Pretrained vocabulary size does not match tokenizer')
        if c.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        if c.peft.method == 'lora':
            from peft import LoraConfig, get_peft_model
            self.model = get_peft_model(self.model, LoraConfig(task_type='CAUSAL_LM',
                r=c.peft.rank, lora_alpha=c.peft.alpha, lora_dropout=c.peft.dropout,
                target_modules=c.peft.target_modules, use_rslora=c.peft.use_rslora,
                bias='none'))
            if c.gradient_checkpointing:
                self.model.enable_input_require_grads()

    def forward(self, ids):
        if ids.shape[1] > self.config.context_length:
            raise ValueError('Input exceeds configured context_length')
        from ashayllm.execution import attention_context
        with attention_context(self.config.attention_kernel):
            return self.model(input_ids=ids, use_cache=False).logits


def create_model(c, vocab_size):
    from ashayllm.model import Decoder
    model = HFDecoder(c, vocab_size) if c.backend == 'huggingface' else Decoder(c, vocab_size)
    if c.quantization == 'fp4_fake':
        from ashayllm.quantization import apply_fake_fp4
        apply_fake_fp4(model)
    return model


def load_parent(model, state, parent_config):
    if model.config.peft == parent_config.peft:
        model.load_state_dict(state)
    elif model.config.peft.method == 'lora' and parent_config.peft.method == 'none':
        # Restore parent weights before wrapping, since LoRA changes module/state names.
        from peft import get_peft_model
        adapter = model.model.peft_config['default']
        base = model.model.unload()
        if hasattr(base, 'peft_config'):
            del base.peft_config
        base.load_state_dict({key.removeprefix('model.'): value for key, value in state.items()})
        model.model = get_peft_model(base, adapter)
        if model.config.gradient_checkpointing:
            model.model.enable_input_require_grads()
    else:
        raise ValueError('Changing an existing adapter configuration requires merge/export and re-import')
