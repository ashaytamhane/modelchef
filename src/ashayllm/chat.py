"""Local CLI conversation harness and Llama-compatible export for optional vLLM."""
import json
import os
from pathlib import Path


def checkpoint_for(folder):
    report = json.loads((folder / "report.json").read_text())
    if report["status"] != "completed":
        raise ValueError("Run is incomplete; resume/finish training before chatting")
    return folder / "checkpoints" / f"step-{report['optimizer_steps']:06d}.pt"


def export_model(folder):
    import torch
    from safetensors.torch import save_file

    from ashayllm.run_config import RunConfig

    folder = Path(folder).resolve()
    config = RunConfig.model_validate_json((folder / "config.json").read_text())
    c = config.model
    state = torch.load(checkpoint_for(folder), map_location="cpu", weights_only=True)["model"]
    output = folder / "export"
    output.mkdir(exist_ok=True)
    if c.quantization != 'none':
        (output / 'ashayllm-quantization.json').write_text(json.dumps({
            'training': c.quantization, 'export': 'full_precision_master_weights',
            'note': 'Export does not preserve fake-quantization forward behavior or produce packed FP4 weights.'}),
            encoding='utf-8')
    if c.backend == 'huggingface':
        from transformers import AutoTokenizer

        from ashayllm.engines import create_model, load_tokenizer
        tokenizer = load_tokenizer(folder)
        model = create_model(c, tokenizer.get_vocab_size())
        model.load_state_dict(state)
        if c.peft.method == 'lora':
            model.model.save_pretrained(folder / 'adapter', safe_serialization=True)
            model.model = model.model.merge_and_unload(safe_merge=True)
        model.model.save_pretrained(output, safe_serialization=True)
        AutoTokenizer.from_pretrained(c.pretrained_path, local_files_only=True,
                                      trust_remote_code=False).save_pretrained(output)
        return output
    weights = {"model.embed_tokens.weight": state["embedding.weight"].contiguous(),
               "model.norm.weight": state["norm.weight"].contiguous()}
    if not c.tie_embeddings:
        weights["lm_head.weight"] = state["head.weight"].contiguous()
    def rotary_layout(weight, heads):
        # Custom attention rotates adjacent pairs; HF Llama rotates two half-vectors.
        shape = weight.shape
        return weight.reshape(heads, c.hidden_size // c.attention_heads // 2, 2,
                              *shape[1:]).transpose(1, 2).reshape(shape).contiguous()
    mapping = {"attention_norm.weight": "input_layernorm.weight",
               "ffn_norm.weight": "post_attention_layernorm.weight",
               "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj",
               "attention.q": "self_attn.q_proj", "attention.k": "self_attn.k_proj",
               "attention.v": "self_attn.v_proj", "attention.out": "self_attn.o_proj"}
    for layer in range(c.layers):
        for original, target in mapping.items():
            for suffix in ([""] if original.endswith(".weight") else
                           ([".weight", ".bias"] if c.bias else [".weight"])):
                weight = state[f"blocks.{layer}.{original}{suffix}"]
                if original in {"attention.q", "attention.k"}:
                    weight = rotary_layout(weight, c.attention_heads if original.endswith("q")
                                            else c.kv_heads)
                weights[f"model.layers.{layer}.{target}{suffix}"] = weight.contiguous()
    save_file(weights, str(output / "model.safetensors"), metadata={"format": "pt"})
    from ashayllm.engines import load_tokenizer
    tokenizer = load_tokenizer(folder)
    (output / "tokenizer.json").write_bytes((folder / "tokenizer.json").read_bytes())
    hf_config = {"model_type": "llama", "architectures": ["LlamaForCausalLM"],
        "vocab_size": tokenizer.get_vocab_size(), "hidden_size": c.hidden_size,
        "intermediate_size": c.intermediate_size, "num_hidden_layers": c.layers,
        "num_attention_heads": c.attention_heads, "num_key_value_heads": c.kv_heads,
        "max_position_embeddings": c.context_length, "rope_theta": c.rope_theta,
        "rms_norm_eps": c.norm_eps, "hidden_act": "silu", "attention_bias": c.bias,
        "mlp_bias": c.bias, "attention_dropout": c.attention_dropout,
        "tie_word_embeddings": c.tie_embeddings, "torch_dtype": "float32",
        "bos_token_id": tokenizer.token_to_id("<bos>"),
        "eos_token_id": tokenizer.token_to_id("<eos>"),
        "pad_token_id": tokenizer.token_to_id("<pad>"), "initializer_range": c.init_std}
    (output / "config.json").write_text(json.dumps(hf_config, indent=2), encoding="utf-8")
    tokenizer_config = {"tokenizer_class": "PreTrainedTokenizerFast",
                        **{role + '_token': tokenizer.id_to_token(tokenizer.token_to_id('<' + role + '>'))
                           for role in ['bos', 'eos', 'unk', 'pad']},
                        "model_max_length": c.context_length, "clean_up_tokenization_spaces": False}
    (output / "tokenizer_config.json").write_text(json.dumps(tokenizer_config), encoding="utf-8")
    return output


def chat(folder, backend="torch", prompt=None, max_tokens=32, temperature=0.8, top_k=30):
    import torch

    from ashayllm.engines import create_model, load_tokenizer
    from ashayllm.hardware import detect_hardware, resolve_runtime
    from ashayllm.run_config import RunConfig

    if max_tokens < 1 or temperature <= 0 or top_k < 0:
        raise ValueError("max_tokens/temperature must be positive; top_k must be nonnegative")
    folder = Path(folder).resolve()
    config = RunConfig.model_validate_json((folder / "config.json").read_text())
    tokenizer = load_tokenizer(folder)
    torch.manual_seed(config.training.seed)
    if backend == "vllm":
        os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", VLLM_NO_USAGE_STATS="1")
        from vllm import LLM, SamplingParams
        export = export_model(folder)
        engine = LLM(model=str(export), tokenizer=str(export), dtype="float16",
                     max_model_len=config.model.context_length, max_num_seqs=1,
                     gpu_memory_utilization=0.3, enforce_eager=True)
        def generate(text):
            # Bound tokenized prompts ourselves; avoid version-specific sampling truncation flags.
            output_tokens = min(max_tokens, config.model.context_length - 1)
            ids = tokenizer.encode(text).ids[-(config.model.context_length - output_tokens):]
            ids = ids or [tokenizer.token_to_id("<bos>")]
            params = SamplingParams(temperature=temperature, top_k=top_k if top_k else -1,
                                    max_tokens=output_tokens,
                                    seed=config.training.seed)
            return engine.generate([{"prompt_token_ids": ids}], params,
                                   use_tqdm=False)[0].outputs[0].text
    else:
        runtime = resolve_runtime({"accelerator": "auto", "precision": "fp32"}, detect_hardware())
        device = runtime["device"]
        model = create_model(config.model, tokenizer.get_vocab_size()).to(device)
        model.load_state_dict(torch.load(checkpoint_for(folder), map_location=device,
                                        weights_only=True)["model"])
        model.eval()
        def generate(text):
            ids = tokenizer.encode(text).ids or [tokenizer.token_to_id("<bos>")]
            sequence = torch.tensor([ids], device=device)
            reply = []
            with torch.no_grad():
                for _ in range(max_tokens):
                    scores = model(sequence[:, -config.model.context_length:])[:, -1].float()
                    scores /= temperature
                    if top_k:
                        boundary = scores.topk(min(top_k, scores.shape[-1])).values[:, -1:]
                        scores = scores.masked_fill(scores < boundary, -float("inf"))
                    token = torch.multinomial(scores.softmax(-1), 1)
                    if token.item() == tokenizer.token_to_id("<eos>"):
                        break
                    reply.append(token.item())
                    sequence = torch.cat((sequence, token), -1)
            return tokenizer.decode(reply)
    print(f"Loaded checkpoint using {backend}; stage={config.stage.kind}. "
          "Short validation runs do not produce a useful assistant.", flush=True)
    if prompt is not None:
        if config.stage.kind in {'sft', 'dpo', 'rlvr', 'distill'}:
            from ashayllm.objectives import format_prompt
            prompt = format_prompt(prompt)
        print(generate(prompt), flush=True)
        return
    print("Enter a message; /reset clears history, /exit quits. History is kept only in memory.")
    history = ""
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if message == "/exit":
            return
        if message == "/reset":
            history = ""
            continue
        if not message:
            continue
        text = history + "User: " + message + "\nAssistant:"
        response = generate(text)
        print("Model: " + response, flush=True)
        # Keep at most a context window of tokenized history between turns.
        history = tokenizer.decode(tokenizer.encode(text + response + "\n").ids[
            -max(1, config.model.context_length // 2):])
