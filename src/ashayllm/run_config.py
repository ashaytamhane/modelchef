"""Strict, versioned configuration for the first complete pretraining workflow."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Data(Section):
    provider: Literal["hf_rows", "local_jsonl"] = "hf_rows"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    local_path: str | None = None
    offset: int = Field(default=0, ge=0)
    max_documents: int = Field(default=256, ge=3, le=10000)
    page_size: int = Field(default=50, ge=1, le=100)
    max_download_bytes: int = Field(default=16000000, gt=0)
    timeout_seconds: int = Field(default=30, gt=0, le=120)
    max_tokens: int = Field(default=200000, gt=0)
    domain: str = "general"
    domain_keywords: list[str] = Field(default_factory=list)


class Processing(Section):
    unicode_normalization: Literal["NFC", "NFKC"] = "NFKC"
    strip_html: bool = True
    min_chars: int = Field(default=200, gt=0)
    max_chars: int = Field(default=20000, gt=0)
    min_alnum_fraction: float = Field(default=0.5, ge=0, le=1)
    max_repeated_line_fraction: float = Field(default=0.4, ge=0, le=1)
    exact_dedup: bool = True
    near_dedup: bool = True
    near_dedup_jaccard: float = Field(default=0.9, gt=0, le=1)
    dedup_ngram_words: int = Field(default=5, gt=0)
    pii: Literal["redact", "drop", "off"] = "redact"
    benchmark_files: list[str] = Field(default_factory=list)
    decontamination_ngram_words: int = Field(default=13, gt=0)
    validation_fraction: float = Field(default=0.1, gt=0, lt=1)
    test_fraction: float = Field(default=0.1, gt=0, lt=1)


class Tokenization(Section):
    algorithm: Literal["byte_bpe", "pretrained"] = "byte_bpe"
    vocab_size: int = Field(default=4096, ge=260, le=262144)
    min_frequency: int = Field(default=2, gt=0)
    special_tokens: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]
    add_bos: bool = False
    append_eos: bool = True


class Architecture(Section):
    backend: Literal["native", "huggingface"] = "native"
    pretrained_path: str | None = None
    pretrained_config_sha256: str | None = None
    quantization: Literal["none", "fp4_fake"] = "none"
    attention_kernel: Literal['auto', 'math', 'flash', 'efficient', 'cudnn'] = 'auto'
    custom_swiglu_path: str | None = None
    type: Literal["decoder_rope"] = "decoder_rope"
    layers: int = Field(default=6, ge=1, le=48)
    hidden_size: int = Field(default=512, gt=0)
    attention_heads: int = Field(default=8, gt=0)
    kv_heads: int = Field(default=2, gt=0)
    intermediate_size: int = Field(default=1360, gt=0)
    context_length: int = Field(default=128, ge=8)
    norm: Literal["rmsnorm"] = "rmsnorm"
    norm_eps: float = Field(default=1e-5, gt=0)
    activation: Literal["swiglu"] = "swiglu"
    rope_theta: float = Field(default=10000, gt=0)
    attention_dropout: float = Field(default=0, ge=0, lt=1)
    residual_dropout: float = Field(default=0, ge=0, lt=1)
    bias: bool = False
    tie_embeddings: bool = True
    init_std: float = Field(default=0.02, gt=0)
    gradient_checkpointing: bool = False


class Training(Section):
    compile: bool = False
    compile_backend: Literal['inductor', 'aot_eager', 'eager'] = 'inductor'
    compile_mode: Literal['default', 'reduce-overhead', 'max-autotune'] = 'default'
    allow_tf32: bool = False
    optimizer_kernel: Literal['default', 'fused', 'foreach'] = 'default'
    accelerator: Literal["auto", "cpu", "cuda", "rocm", "mps"] = "auto"
    precision: Literal["auto", "fp32", "bf16", "fp16"] = "auto"
    seed: int = Field(default=42, ge=0)
    deterministic: bool = True
    steps: int = Field(default=40, gt=0)
    micro_batch_size: int = Field(default=2, gt=0)
    gradient_accumulation: int = Field(default=2, gt=0)
    optimizer: Literal["adamw"] = "adamw"
    learning_rate: float = Field(default=0.0005, gt=0)
    min_learning_rate_ratio: float = Field(default=0.1, ge=0, le=1)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0)
    optimizer_eps: float = Field(default=1e-8, gt=0)
    schedule: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = Field(default=4, ge=0)
    gradient_clip_norm: float = Field(default=1, gt=0)
    checkpoint_every: int = Field(default=20, gt=0)
    evaluate_every: int = Field(default=10, gt=0)
    log_every: int = Field(default=5, gt=0)
    max_runtime_seconds: int = Field(default=600, gt=0)


class Evaluation(Section):
    benchmark: Literal["auto", "none", "hellaswag", "financial_phrasebank"] = "auto"
    benchmark_examples: int = Field(default=8, ge=1, le=100)
    benchmark_max_download_bytes: int = Field(default=2000000, gt=0)
    batches: int = Field(default=4, gt=0)
    prompt: str = "Education helps people"
    generate_tokens: int = Field(default=24, ge=1)
    temperature: float = Field(default=0.8, gt=0)
    top_k: int = Field(default=30, ge=0)


class Stage(Section):
    kind: Literal["pretrain", "midtrain", "sft", "dpo", "rlvr", "distill"] = "pretrain"
    parent_run: str | None = None
    teacher_run: str | None = None
    distillation_temperature: float = Field(default=2, gt=0)
    distillation_alpha: float = Field(default=0.5, ge=0, le=1)
    dpo_beta: float = Field(default=0.1, gt=0)
    reward: Literal["exact_match", "integer"] = "integer"
    group_size: int = Field(default=4, ge=2, le=32)
    rollout_tokens: int = Field(default=16, ge=1)
    kl_coefficient: float = Field(default=0.02, ge=0)
    clip_ratio: float = Field(default=0.2, gt=0, lt=1)
    reasoning_format: Literal['plain', 'think_answer'] = 'plain'

    def data_fields(self):
        if self.kind == 'dpo':
            return ['prompt', 'chosen', 'rejected']
        if self.kind == 'rlvr':
            return ['prompt', 'answer']
        if self.reasoning_format == 'think_answer':
            return ['prompt', 'reasoning', 'answer']
        return ['prompt', 'response']


class RunConfig(Section):
    schema_version: Literal[1] = 1
    name: str = Field(default="first-local", pattern=r"^[a-zA-Z0-9_-]+$")
    output_dir: str = "artifacts/training"
    explain: Literal["detailed", "brief"] = "detailed"
    data: Data = Field(default_factory=Data)
    processing: Processing = Field(default_factory=Processing)
    tokenizer: Tokenization = Field(default_factory=Tokenization)
    model: Architecture = Field(default_factory=Architecture)
    training: Training = Field(default_factory=Training)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    stage: Stage = Field(default_factory=Stage)

    @model_validator(mode="after")
    def consistency(self):
        m, p, t = self.model, self.processing, self.training
        if m.hidden_size % m.attention_heads or m.attention_heads % m.kv_heads:
            raise ValueError("hidden_size must be divisible by heads; heads by kv_heads")
        if (m.hidden_size // m.attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if p.min_chars > p.max_chars or p.validation_fraction + p.test_fraction >= 1:
            raise ValueError("Invalid processing lengths or split fractions")
        if t.warmup_steps >= t.steps or any(not 0 <= b < 1 for b in t.betas):
            raise ValueError("Warmup must be shorter than training; betas must be in [0,1)")
        if len(set(self.tokenizer.special_tokens)) != len(self.tokenizer.special_tokens):
            raise ValueError("Special tokens must be unique")
        if not {"<pad>", "<unk>", "<bos>", "<eos>"} <= set(self.tokenizer.special_tokens):
            raise ValueError("Required pad/unk/bos/eos tokens missing")
        if self.data.provider == "local_jsonl" and not self.data.local_path:
            raise ValueError("local_jsonl requires local_path")
        if self.data.domain != "general" and not self.data.domain_keywords:
            raise ValueError("Domain selection requires explicit domain_keywords")
        if self.evaluation.benchmark == "auto" and self.data.domain not in {"general", "finance"}:
            raise ValueError("No automatic benchmark recipe for this domain; override or choose none")
        if self.stage.kind != "pretrain" and not self.stage.parent_run:
            raise ValueError("Post-training requires stage.parent_run")
        if self.stage.kind in {"sft", "dpo", "rlvr", "distill"} and self.data.provider != "local_jsonl":
            raise ValueError("Structured post-training requires a local JSONL dataset")
        if self.stage.kind == 'distill' and not self.stage.teacher_run:
            raise ValueError('Token distillation requires a teacher_run')
        if m.backend == 'huggingface' and not m.pretrained_path:
            raise ValueError('Hugging Face backend requires a local pretrained_path')
        return self
