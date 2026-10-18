"""Strict recipes for local vision-language SFT and Whisper speech recognition."""
from typing import Literal

from pydantic import Field, model_validator

from modelchef.run_config import Section, Training


class MediaData(Section):
    manifest: str
    max_records: int = Field(default=128, ge=3)
    max_manifest_bytes: int = Field(default=2000000, gt=0)
    max_record_bytes: int = Field(default=64000, gt=0)
    max_media_bytes: int = Field(default=16000000, gt=0)
    max_total_media_bytes: int = Field(default=256000000, gt=0)
    max_image_pixels: int = Field(default=4000000, gt=0)
    max_audio_seconds: float = Field(default=30, gt=0, le=30)
    validation_fraction: float = Field(default=0.1, gt=0, lt=0.5)
    test_fraction: float = Field(default=0.1, gt=0, lt=0.5)


class MediaModel(Section):
    path: str
    max_model_bytes: int = Field(default=4000000000, gt=0)
    vision_format: Literal['chat', 'suffix'] = 'chat'
    freeze_encoders: bool = False
    encoder_modules: list[str] = Field(default_factory=list)
    attention_kernel: Literal['auto', 'math', 'flash', 'efficient', 'cudnn'] = 'auto'
    gradient_checkpointing: bool = False
    language: str | None = None
    speech_task: Literal['transcribe', 'translate'] = 'transcribe'


class MediaEvaluation(Section):
    max_examples: int = Field(default=8, ge=1)
    max_new_tokens: int = Field(default=64, ge=1, le=1024)


class MultimodalConfig(Section):
    schema_version: Literal[1] = 1
    name: str = 'multimodal'
    modality: Literal['vision', 'speech']
    output_dir: str = 'artifacts/multimodal'
    data: MediaData
    model: MediaModel
    max_text_tokens: int = Field(default=512, ge=8)
    training: Training = Field(default_factory=lambda: Training(
        micro_batch_size=1, gradient_accumulation=1, warmup_steps=0))
    evaluation: MediaEvaluation = Field(default_factory=MediaEvaluation)

    @model_validator(mode='after')
    def supported(self):
        if self.training.profile_phases:
            raise ValueError('Phase profiling currently supports text training only; set profile_phases: false')
        if self.training.performance_gate:
            raise ValueError('Performance gates currently apply to text training stages')
        if self.training.distributed != 'none':
            raise ValueError('Multimodal training currently uses a single process')
        if self.training.micro_batch_size != 1:
            raise ValueError('Multimodal micro_batch_size is 1; use gradient_accumulation')
        if self.training.warmup_steps >= self.training.steps:
            raise ValueError('warmup_steps must be smaller than steps')
        if self.model.freeze_encoders and not self.model.encoder_modules:
            raise ValueError('freeze_encoders requires explicit encoder_modules to avoid guessing architecture')
        return self
