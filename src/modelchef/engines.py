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
        from modelchef.low_precision import prepare_bnb_libraries
        prepare_bnb_libraries()
        from transformers import AutoModelForCausalLM
        self.config = c
        options = {'torch_dtype': torch.float32}
        if c.peft.method == 'qlora':
            from transformers import BitsAndBytesConfig
            if not torch.cuda.is_available() or torch.version.hip:
                raise ValueError('The QLoRA integration requires an NVIDIA CUDA GPU')
            compute = getattr(torch, 'bfloat16' if c.peft.compute_dtype == 'bf16' else 'float16')
            if compute == torch.bfloat16 and not torch.cuda.is_bf16_supported():
                raise ValueError('Select model.peft.compute_dtype: fp16 on this GPU')
            options = {'torch_dtype': compute, 'device_map': {'': torch.cuda.current_device()},
                       'quantization_config': BitsAndBytesConfig(load_in_4bit=True,
                           bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=c.peft.double_quant,
                           bnb_4bit_compute_dtype=compute)}
        if c.pretrained_config_sha256:
            from modelchef.corpus import digest
            if digest((Path(c.pretrained_path) / 'config.json').read_bytes()) != c.pretrained_config_sha256:
                raise ValueError('Original pretrained model configuration changed')
        self.model = AutoModelForCausalLM.from_pretrained(c.pretrained_path,
            local_files_only=True, trust_remote_code=False, attn_implementation='sdpa', **options)
        if c.custom_swiglu_path:
            raise ValueError('Custom SwiGLU hooks currently target the native decoder only')
        if self.model.get_input_embeddings().num_embeddings != vocab_size:
            raise ValueError('Pretrained vocabulary size does not match tokenizer')
        if c.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        self.model.config.use_cache = False
        if c.peft.method in {'lora', 'qlora'}:
            from peft import LoraConfig, get_peft_model
            if c.peft.method == 'qlora':
                from peft import prepare_model_for_kbit_training
                self.model = prepare_model_for_kbit_training(self.model,
                    use_gradient_checkpointing=c.gradient_checkpointing)
            self.model = get_peft_model(self.model, LoraConfig(task_type='CAUSAL_LM',
                r=c.peft.rank, lora_alpha=c.peft.alpha, lora_dropout=c.peft.dropout,
                target_modules=c.peft.target_modules, use_rslora=c.peft.use_rslora,
                bias='none'))
            if c.gradient_checkpointing:
                self.model.enable_input_require_grads()

    def forward(self, ids):
        if ids.shape[1] > self.config.context_length:
            raise ValueError('Input exceeds configured context_length')
        from modelchef.execution import attention_context
        with attention_context(self.config.attention_kernel):
            return self.model(input_ids=ids, use_cache=False).logits

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if self.config.peft.method == 'qlora':
            import bitsandbytes as bnb
            state_dict = dict(state_dict)
            for name, layer in self.model.named_modules():
                if isinstance(layer, bnb.nn.Linear4bit):
                    key = 'model.' + name + '.weight'
                    statistics = {field.removeprefix(key + '.'): state_dict.pop(field)
                                  for field in list(state_dict) if field.startswith(key + '.')}
                    if not statistics:
                        raise ValueError(f'QLoRA checkpoint lacks NF4 metadata for {name}')
                    layer.weight = bnb.nn.Params4bit.from_prequantized(
                        data=state_dict[key], quantized_stats=statistics,
                        requires_grad=False, device=layer.weight.device, module=layer)
                    layer.quant_state = layer.weight.quant_state
        return super().load_state_dict(state_dict, strict=strict, assign=assign)


def create_model(c, vocab_size):
    from modelchef.model import Decoder
    model = HFDecoder(c, vocab_size) if c.backend == 'huggingface' else Decoder(c, vocab_size)
    if c.backend == 'native' and c.peft.method == 'lora':
        from modelchef.adapters import apply_lora
        apply_lora(model, c.peft)
    if c.quantization == 'fp4_fake':
        from modelchef.quantization import apply_fake_fp4
        apply_fake_fp4(model)
    elif c.quantization == 'nvfp4':
        from modelchef.low_precision import apply_nvfp4
        apply_nvfp4(model)
    return model


def load_parent(model, state, parent_config):
    if parent_config.quantization == 'nvfp4' and model.config.quantization != 'nvfp4':
        state = {key: value for key, value in state.items() if not key.endswith('_extra_state')}
    if model.config.peft == parent_config.peft:
        if model.config.quantization == 'nvfp4' and parent_config.quantization != 'nvfp4':
            missing, unexpected = model.load_state_dict(state, strict=False)
            if unexpected or any(not key.endswith('_extra_state') for key in missing):
                raise ValueError('Parent base weights differ from the NVFP4 model')
        else:
            model.load_state_dict(state)
    elif model.config.peft.method == 'qlora' and parent_config.peft.method == 'none':
        # run_pipeline resolves pretrained_path to the exact exported parent before quantization.
        return
    elif model.config.backend == 'native' and model.config.peft.method == 'lora' and parent_config.peft.method == 'none':
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected or any(not key.endswith(('.lora_A', '.lora_B')) for key in missing):
            raise ValueError('Parent base weights differ from the adapter model')
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
