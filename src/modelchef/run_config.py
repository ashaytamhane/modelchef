"""Strict, versioned configuration for the first complete pretraining workflow."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Data(Section):
    provider: Literal["hf_rows", "hf_parquet", "local_jsonl"] = "hf_rows"
    dataset: str = "HuggingFaceFW/fineweb-edu"
    subset: str = "sample-10BT"
    split: str = "train"
    text_field: str = "text"
    prepared_run: str | None = None
    local_path: str | None = None
    local_paths: list[str] = Field(default_factory=list)
    storage: Literal["memory", "disk"] = "memory"
    max_record_bytes: int = Field(default=4000000, gt=0)
    offset: int = Field(default=0, ge=0)
    max_documents: int = Field(default=256, ge=3)
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
    collapse_consecutive_duplicate_sentences: bool = False
    minimum_sentence_words: int = Field(default=6, ge=1)
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
    packing: Literal['concatenate', 'document'] = 'concatenate'
    algorithm: Literal["byte_bpe", "pretrained"] = "byte_bpe"
    vocab_size: int = Field(default=4096, ge=260, le=262144)
    min_frequency: int = Field(default=2, gt=0)
    training_max_documents: int | None = Field(default=None, gt=0)
    special_tokens: list[str] = ["<pad>", "<unk>", "<bos>", "<eos>"]
    add_bos: bool = False
    append_eos: bool = True


class PEFT(Section):
    method: Literal['none', 'lora', 'qlora'] = 'none'
    rank: int = Field(default=8, gt=0)
    alpha: int = Field(default=16, gt=0)
    dropout: float = Field(default=0, ge=0, lt=1)
    target_modules: list[str] = Field(default_factory=lambda: ['q_proj', 'v_proj'])
    use_rslora: bool = False
    compute_dtype: Literal['bf16', 'fp16'] = 'bf16'
    double_quant: bool = True


class Architecture(Section):
    peft: PEFT = Field(default_factory=PEFT)
    backend: Literal["native", "huggingface"] = "native"
    pretrained_path: str | None = None
    pretrained_config_sha256: str | None = None
    quantization: Literal["none", "fp4_fake", "nvfp4"] = "none"
    attention_kernel: Literal['auto', 'math', 'flash', 'efficient', 'cudnn'] = 'auto'
    custom_swiglu_path: str | None = None
    type: Literal["decoder_rope", "decoder_moe"] = "decoder_rope"
    num_experts: int = Field(default=4, ge=2)
    experts_per_token: int = Field(default=2, ge=1)
    router_aux_loss_coef: float = Field(default=0.01, ge=0)
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


class PerformanceGate(Section):
    probe_steps: int = Field(default=50, gt=1)
    warmup_steps: int = Field(default=10, ge=0)
    safety_multiplier: float = Field(default=2.0, ge=1)
    overhead_seconds: int = Field(default=600, ge=0)
    max_allocated_memory_fraction: float = Field(default=0.9, gt=0, le=1)

    @model_validator(mode='after')
    def window(self):
        if self.warmup_steps >= self.probe_steps:
            raise ValueError('Performance-gate warmup must be shorter than its probe')
        return self


class Training(Section):
    performance_gate: PerformanceGate | None = None
    distributed: Literal['none', 'ddp', 'fsdp2'] = 'none'
    profile_phases: bool = False
    distributed_timeout_seconds: int = Field(default=300, gt=0)
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
    optimizer: Literal['adamw', 'adamw8bit', 'adamw_in_backward', 'adamw8bit_in_backward'] = 'adamw'
    learning_rate: float = Field(default=0.0005, gt=0)
    min_learning_rate_ratio: float = Field(default=0.1, ge=0, le=1)
    betas: tuple[float, float] = (0.9, 0.95)
    weight_decay: float = Field(default=0.1, ge=0)
    optimizer_eps: float = Field(default=1e-8, gt=0)
    schedule: Literal["cosine", "constant"] = "cosine"
    warmup_steps: int = Field(default=4, ge=0)
    gradient_clip_norm: float = Field(default=1, ge=0)
    checkpoint_every: int = Field(default=20, gt=0)
    checkpoint_at_start: bool = False
    evaluate_every: int = Field(default=10, gt=0)
    log_every: int = Field(default=5, gt=0)
    max_runtime_seconds: int = Field(default=600, gt=0)


class Evaluation(Section):
    batch_size: int | None = Field(default=None, gt=0)
    evaluate_test: bool = True
    benchmark: Literal["auto", "none", "hellaswag", "financial_phrasebank"] = "auto"
    benchmark_examples: int = Field(default=8, ge=1, le=100)
    benchmark_max_download_bytes: int = Field(default=2000000, gt=0)
    batches: Annotated[int, Field(gt=0)] | Literal['all'] = 4
    prompt: str = "Education helps people"
    generate_tokens: int = Field(default=24, ge=1)
    temperature: float = Field(default=0.8, gt=0)
    top_k: int = Field(default=30, ge=0)


class HuggingFaceModel(Section):
    repo_id: str = Field(pattern=r'^[A-Za-z0-9][A-Za-z0-9_.-]*(/[A-Za-z0-9][A-Za-z0-9_.-]*)?$')
    revision: str = Field(default='main', min_length=1)
    max_download_bytes: int = Field(default=500000000, gt=0)
    context_length: int | None = Field(default=None, ge=8)


class Stage(Section):
    oversized_policy: Literal['error', 'skip'] = 'error'
    kind: Literal["pretrain", "midtrain", "sft", "dpo", "rlvr", "distill"] = "pretrain"
    parent_run: str | HuggingFaceModel | None = None
    teacher_run: str | HuggingFaceModel | None = None
    distillation_temperature: float = Field(default=2, gt=0)
    distillation_alpha: float = Field(default=0.5, ge=0, le=1)
    dpo_beta: float = Field(default=0.1, gt=0)
    reward: Literal["exact_match", "integer"] = "integer"
    group_size: int = Field(default=4, ge=2, le=32)
    rollout_tokens: int = Field(default=16, ge=1)
    kl_coefficient: float = Field(default=0.02, ge=0)
    clip_ratio: float = Field(default=0.2, gt=0, lt=1)
    reasoning_format: Literal['plain', 'think_answer'] = 'plain'
    prompt_format: Literal['user_assistant', 'chat_transcript'] = 'user_assistant'

    @property
    def uses_distillation(self):
        return self.kind == 'distill' or self.teacher_run is not None

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
        model_config, processing, training = self.model, self.processing, self.training
        if model_config.hidden_size % model_config.attention_heads or model_config.attention_heads % model_config.kv_heads:
            raise ValueError("hidden_size must be divisible by heads; heads by kv_heads")
        if (model_config.hidden_size // model_config.attention_heads) % 2:
            raise ValueError("RoPE head dimension must be even")
        if processing.min_chars > processing.max_chars or processing.validation_fraction + processing.test_fraction >= 1:
            raise ValueError("Invalid processing lengths or split fractions")
        if processing.collapse_consecutive_duplicate_sentences and self.stage.kind not in {'pretrain', 'midtrain'}:
            raise ValueError('Sentence cleanup requires raw-text pretraining or midtraining')
        if training.warmup_steps >= training.steps or any(not 0 <= b < 1 for b in training.betas):
            raise ValueError("Warmup must be shorter than training; betas must be in [0,1)")
        if training.performance_gate and training.performance_gate.probe_steps >= training.steps:
            raise ValueError('Performance probe must be shorter than training')
        if training.performance_gate and training.distributed != 'none':
            raise ValueError('Performance gates currently require single-process training')
        if training.optimizer != 'adamw' and training.optimizer_kernel != 'default':
            raise ValueError('Optional optimizer backends require optimizer_kernel: default')
        if training.optimizer.endswith('_in_backward') and (
                training.gradient_accumulation != 1 or training.gradient_clip_norm != 0 or
                training.distributed != 'none' or training.compile or training.precision == 'fp16' or
                self.stage.kind in {'dpo', 'rlvr'}):
            raise ValueError('In-backward AdamW requires accumulation=1, gradient_clip_norm=0, '
                                 'single process, no compile/FP16, and a supervised objective')
        if len(set(self.tokenizer.special_tokens)) != len(self.tokenizer.special_tokens):
            raise ValueError("Special tokens must be unique")
        if not {"<pad>", "<unk>", "<bos>", "<eos>"} <= set(self.tokenizer.special_tokens):
            raise ValueError("Required pad/unk/bos/eos tokens missing")
        if self.data.provider == "local_jsonl" and not (self.data.local_path or self.data.local_paths):
            raise ValueError("local_jsonl requires local_path or local_paths")
        if self.tokenizer.packing == 'document' and (self.data.storage != 'disk' or
                self.stage.kind not in {'pretrain', 'midtrain'}):
            raise ValueError('Document packing requires disk-based next-token training')
        if self.data.storage == 'disk' and self.stage.kind not in {'pretrain', 'midtrain'}:
            raise ValueError('Disk-backed storage currently supports pretrain and midtrain')
        if self.data.storage == 'disk' and self.data.provider not in {'local_jsonl', 'hf_parquet'}:
            raise ValueError('Disk-backed acquisition requires local JSONL or bulk Parquet')
        if self.data.local_path and self.data.local_paths:
            raise ValueError('Choose local_path or local_paths, not both')
        if self.data.domain != "general" and not self.data.domain_keywords:
            raise ValueError("Domain selection requires explicit domain_keywords")
        if self.evaluation.benchmark == "auto" and self.data.domain not in {"general", "finance"}:
            raise ValueError("No automatic benchmark recipe for this domain; override or choose none")
        if self.stage.kind != "pretrain" and not self.stage.parent_run:
            raise ValueError("Post-training requires stage.parent_run")
        if self.stage.kind == 'pretrain' and self.stage.parent_run:
            raise ValueError('From-scratch pretraining cannot load parent weights; choose midtrain to continue a parent')
        if self.stage.kind in {"sft", "dpo", "rlvr", "distill"} and self.data.provider != "local_jsonl":
            raise ValueError("Structured post-training requires a local JSONL dataset")
        if self.stage.kind == 'distill' and not self.stage.teacher_run:
            raise ValueError('Token distillation requires a teacher_run')
        if self.stage.teacher_run and self.stage.kind not in {'pretrain', 'midtrain', 'sft', 'distill'}:
            raise ValueError('Online token distillation supports pretrain, midtrain, sft and distill only')
        if model_config.backend == 'huggingface' and not model_config.pretrained_path:
            raise ValueError('Hugging Face backend requires a local pretrained_path')
        if model_config.peft.method != 'none':
            if self.stage.kind == 'pretrain':
                raise ValueError('PEFT requires a trained parent; use a continuation/post-training stage')
            if not model_config.peft.target_modules or any(not name.strip() for name in model_config.peft.target_modules):
                raise ValueError('LoRA requires explicit nonempty target_modules')
            if model_config.quantization != 'none':
                raise ValueError('PEFT and FP4 projection quantization cannot be combined')
        if (model_config.backend == 'native' and model_config.peft.method == 'lora' and model_config.tie_embeddings and
                any(name == 'head' or name.endswith('.head') for name in model_config.peft.target_modules)):
            raise ValueError('LoRA on a tied output head cannot be merged; set tie_embeddings: false')
        if model_config.peft.method == 'qlora' and model_config.backend != 'huggingface':
            raise ValueError('QLoRA requires the Hugging Face backend')
        if model_config.experts_per_token > model_config.num_experts:
            raise ValueError('experts_per_token must not exceed num_experts')
        if model_config.custom_swiglu_path and (model_config.backend != 'native' or model_config.type != 'decoder_rope'):
            raise ValueError('Custom SwiGLU requires a native dense decoder')
        if model_config.type == 'decoder_moe' and model_config.bias:
            raise ValueError('Native MoE requires bias: false for Mixtral-compatible export')
        if model_config.quantization == 'nvfp4' and (model_config.backend != 'native' or model_config.type != 'decoder_rope'):
            raise ValueError('NVFP4 currently requires a native dense decoder')
        if model_config.quantization == 'nvfp4' and (model_config.gradient_checkpointing or model_config.custom_swiglu_path):
            raise ValueError('NVFP4 currently excludes gradient checkpointing and custom SwiGLU')
        if training.distributed == 'ddp' and (training.compile or model_config.peft.method == 'qlora' or
                                      model_config.quantization == 'nvfp4' or self.stage.kind == 'rlvr'):
            raise ValueError('DDP currently excludes compile, QLoRA, NVFP4 and RLVR')
        if training.distributed != 'none' and training.optimizer != 'adamw':
            raise ValueError('Distributed training currently requires standard AdamW')
        if training.distributed == 'fsdp2' and (
                model_config.backend != 'native' or model_config.type != 'decoder_rope' or model_config.peft.method != 'none' or
                model_config.quantization != 'none' or model_config.custom_swiglu_path or training.compile or
                training.precision == 'fp16' or training.accelerator not in {'auto', 'cuda'} or
                self.stage.kind not in {'pretrain', 'midtrain'} or self.stage.uses_distillation):
            raise ValueError('FSDP2 requires native dense unquantized pretrain/midtrain, standard AdamW, '
                             'CUDA FP32/BF16, no PEFT, teacher, compile or custom kernels')
        return self

    def validate_execution(self):
        """Stored imports/students share the schema but are not from-scratch recipes."""
        if self.stage.kind == 'pretrain' and self.model.backend == 'huggingface':
            raise ValueError('Import the pretrained model, then choose a continuation/post-training stage')
        if self.stage.kind == 'pretrain' and self.stage.uses_distillation:
            if self.tokenizer.algorithm != 'pretrained':
                raise ValueError('Distilled pretraining requires tokenizer.algorithm: pretrained to reuse the teacher vocabulary')
        elif self.stage.kind == 'pretrain' and self.tokenizer.algorithm != 'byte_bpe':
            raise ValueError('From-scratch pretraining requires tokenizer.algorithm: byte_bpe')
        return self
