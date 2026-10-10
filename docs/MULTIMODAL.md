# Vision and speech recipes

ModelChef fine-tunes local vision-language models on image/prompt/response triples and
Whisper models on audio/transcript pairs. It exports weights and processors and runs
image-to-text or audio-to-text inference. These are dedicated multimodal recipes;
the native scratch decoder and its text recipes remain text-only.

Vision support uses Hugging Face's [multimodal processors and chat templates](https://huggingface.co/docs/transformers/v4.55.4/chat_templating_multimodal).
Speech support uses the [Whisper encoder-decoder implementation](https://huggingface.co/docs/transformers/model_doc/whisper).
An encoder turns pixels or audio features into representations the decoder can use.
The processor supplies image transformations or log-mel features and the pretrained
tokenizer; training a replacement text tokenizer would break that model's vocabulary.

## 1. Install and select a local model

Use the normal hardware setup, then install the optional processor dependencies:

```bash
uv pip install -e '.[multimodal]'
python -m modelchef.cli doctor
```

Obtain a compatible model directory containing safetensors, config, tokenizer and
processor files. The existing `download-model REPO` command previews a revision-pinned
download; `--run` explicitly downloads it under `artifacts/pretrained`. Set a deliberate
`--max-bytes` cap and put the returned local path into `model.path`. No multimodal training
or inference command downloads missing files or executes remote repository code.

Choose a model that fits your hardware with room for gradients and optimizer state.
Compatibility depends on the installed Transformers version; a model requiring remote
Python code is outside this workflow. Start with a small model and a short test run.

## 2. Prepare ingredients

Image records use one local image per example:

```json
{"image":"images/example.png","prompt":"Describe the chart.","response":"Revenue increased over the three periods."}
```

Speech records use local uncompressed **mono PCM16 WAV at 16000 Hz**, at most 30 seconds:

```json
{"audio":"clips/example.wav","text":"The meeting starts at nine."}
```

Paths resolve relative to the JSONL manifest. Supply at least three distinct media
assets so train, validation and test splits are nonempty. Records sharing identical
media bytes stay in the same split, even if their questions or transcripts differ.
Image dimensions, audio duration, individual file sizes and cumulative selected media
bytes are checked before loading model weights. Text/image-token overflow fails clearly
rather than silently truncating a supervised answer. Media are loaded one example at a
time; the manifest itself is bounded by the recipe.

Use licensed, consented examples and inspect transcript/image contents before training.
This path does not automatically redact faces, spoken personal information or text
inside images, and asset grouping does not establish benchmark decontamination.

## 3. Write and preview a recipe

```bash
cp configs/multimodal-vision.yaml configs/vision.local.yaml
# For speech, copy configs/multimodal-speech.yaml instead.
python -m modelchef.cli check-config configs/vision.local.yaml
python -m modelchef.cli multimodal configs/vision.local.yaml
```

Edit `data.manifest` and `model.path` to point to your inputs. Preview verifies selected
files, hashes and split sizes without importing model weights, launching AWS or training.

| Field | Requirement and meaning |
| --- | --- |
| `modality` | **Required:** `vision` or `speech` |
| `data.manifest` | **Required:** local JSONL path |
| `model.path` | **Required:** local model/processor directory |
| `schema_version`, `name`, `output_dir` | Optional: version 1, `multimodal`, `artifacts/multimodal` |
| `data.max_records` | Optional: 128 selected examples, minimum 3 |
| `data.max_manifest_bytes`, `max_record_bytes` | Optional: 2 MB manifest, 64 KB per JSONL line |
| `data.max_media_bytes`, `max_total_media_bytes` | Optional: 16 MB per media file, 256 MB unique selected assets |
| `data.max_image_pixels` | Optional: 4 million pixels before decoding |
| `data.max_audio_seconds` | Optional: 30 seconds; cannot exceed Whisper's supported short-clip limit |
| `data.validation_fraction`, `test_fraction` | Optional: 0.1 each; splits group by media checksum |
| `model.max_model_bytes` | Optional: 4 GB local model/processor files |
| `model.vision_format` | Optional: `chat`; use `suffix` for processors such as PaliGemma that generate target labels |
| `model.freeze_encoders`, `encoder_modules` | Optional: false and []; freezing requires explicit architecture-specific module paths |
| `model.attention_kernel` | Optional: `auto`; also `math`, `flash`, `efficient`, `cudnn` |
| `model.gradient_checkpointing` | Optional: false; use a model that implements this feature |
| `model.language`, `speech_task` | Optional: null and `transcribe`; multilingual Whisper also supports `translate`. English-only models require null language and transcription |
| `max_text_tokens` | Optional: 512, including vision image tokens; example vision recipe uses 2048 |
| `training` | Optional: shared optimizer/runtime/schedule fields from text recipes; single process, microbatch 1, larger effective batches via gradient accumulation |
| `evaluation.max_examples`, `max_new_tokens` | Optional: 8 held-out examples and 64 generated tokens |

The complete strict schema, including every training default and required field, is
[multimodal.schema.json](../configs/multimodal.schema.json). Unknown settings are rejected.
Multimodal DDP, PEFT, native FP4, scratch multimodal pretraining, video and speech synthesis
are not implemented by this trainer. Text workflows retain their separate capabilities.

## 4. Train, evaluate and recover

```bash
python -m modelchef.cli multimodal configs/vision.local.yaml --run
python -m modelchef.cli multimodal configs/vision.local.yaml --run --resume artifacts/multimodal/YOUR_RUN/checkpoints/step-000010.pt
```

Vision chat training masks user/image-prefix tokens and supervises the assistant answer.
Suffix mode uses the processor's own target labels. Speech training learns transcript
tokens from audio features. Encoder freezing, gradient checkpointing, autocast, compilation,
SDPA policy and optimizer kernels follow the declared recipe and available hardware.

Runs record config, media/model checksums, grouped splits, environment, metrics and
atomic checkpoints with optimizer, scaler, sampler and Torch RNG state. Resume rejects
changed recipes, selected media or original model files. Keep those inputs for recovery.
The runtime cap is checked between optimizer steps; a single compile/model call can take
longer than that cap. Managed AWS also has its independent shutdown deadlines.

Held-out results include loss and exact match; speech includes word error rate (word edits
divided by reference word count, which can exceed 1). Vision exact match is only a simple
diagnostic, not a replacement for VQA/captioning suites. Small local samples establish
pipeline behavior, not general model quality. Completed runs save a self-contained
`export/` with model weights, generation configuration, tokenizer and processor.

## 5. Run inference, use AWS or publish

```bash
python -m modelchef.cli predict-media configs/vision.local.yaml --model-path artifacts/multimodal/YOUR_RUN/export --media data/vision/images/example.png --prompt "Describe this image."
python -m modelchef.cli predict-media configs/speech.local.yaml --model-path artifacts/multimodal/YOUR_RUN/export --media data/speech/clips/example.wav
```

For managed GPU training, set `pipeline_config` in your AWS training configuration to the
multimodal recipe. The runner bundles bounded media and local safetensors/processor files,
runs a two-step CPU preflight, installs multimodal guest dependencies and launches the same
trainer. Budget, result-transfer caps and termination safeguards apply. Increase bundle/result
caps only after checking model size and cost; the local preflight still loads the real model.
Live AWS multimodal training has not been exercised by the local integration tests.

Publish the export with the [Hugging Face publication workflow](PUBLISHING.md), selecting
`image-text-to-text` or `automatic-speech-recognition` as the pipeline tag. Multimodal API
requests are not enabled by the existing text-only API middleware; use CLI media inference
for this increment. Whisper exports use Transformers inference rather than the vLLM text server.
