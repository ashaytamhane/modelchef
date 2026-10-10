# Share a model on Hugging Face

Publishing uploads an exported model, its tokenizer/processor and a generated model card.
It uses the Hugging Face [repository](https://huggingface.co/docs/huggingface_hub/guides/repository)
and [commit APIs](https://huggingface.co/docs/huggingface_hub/guides/upload).
Preview runs offline; upload requires `--run` and your cached authentication token.

## 1. Prepare the export

For a text run:

```bash
python -m modelchef.cli export artifacts/training/YOUR_RUN
```

Completed multimodal runs already contain `export/`. Confirm that local inference works
before sharing. Publishing expects safetensors, model configuration, tokenizer configuration
and the processor metadata required by your modality. It accepts complete model exports,
including merged fine-tunes; publishing standalone adapters is not implemented here.

## 2. Write the publication recipe

```bash
cp configs/publish-model.yaml configs/publish.local.yaml
```

Edit every placeholder. Choose a license compatible with the model, source weights and
data permissions; the example license is illustrative and is not an automatic license grant.
Describe actual training data, evaluation and limitations. No scores or licensing rights
are inferred from the weights. For multimodal exports select the corresponding pipeline tag.

```yaml
schema_version: 1
export_dir: artifacts/multimodal/YOUR_RUN/export
repo_id: YOUR_USERNAME/YOUR_MODEL
private: true
license: apache-2.0
summary: A vision-language model fine-tuned for chart descriptions.
training_data: Describe the actual authorized image/prompt/response data and processing.
evaluation: Report the actual held-out protocol and results.
limitations: Describe known errors and intended deployment boundaries.
base_model: null
pipeline_tag: image-text-to-text
max_upload_bytes: 8000000000
```

**Required:** `export_dir`, `repo_id`, `license`, `summary`, `training_data`, `evaluation`,
`limitations`. **Optional:** schema version 1, private visibility (default true), `base_model`
(default null), `pipeline_tag` (default `text-generation`), `max_upload_bytes` (8 GB),
`max_text_file_bytes` (64 MB), and `commit_message`.
See [publish.schema.json](../configs/publish.schema.json) for the complete strict schema.

## 3. Review the exact upload

```bash
python -m modelchef.cli check-config configs/publish.local.yaml
python -m modelchef.cli publish-model configs/publish.local.yaml
```

Preview lists names, sizes, SHA256 checksums, visibility and the full generated model card.
It does not contact the Hub or read a token. The allowlist covers safetensors, tokenizer,
generation configuration and processor files. It excludes raw datasets, optimizer
checkpoints, logs, environment files and arbitrary Python code. Symlinks, missing indexed
shards and recognizable credential strings in text metadata fail validation.
This credential check is a heuristic; review metadata and model-card text yourself.

## 4. Authenticate locally and upload explicitly

```bash
uv pip install -e '.[pretrained,pipeline]'
hf auth login
python -m modelchef.cli publish-model configs/publish.local.yaml --run
```

Use a token with write permission to the selected repository. Authentication uses the
Hugging Face local token cache or `HF_TOKEN` environment variable. Do not put tokens in
recipes, command-line arguments, Git or model cards. ModelChef does not print or save them.
See the [official CLI authentication instructions](https://huggingface.co/docs/huggingface_hub/guides/cli).

New repositories default to private. To publish publicly, explicitly set `private: false`
and review the preview again. An existing repository must already match the requested
visibility; ModelChef refuses to silently change it. Upload creates one model commit and
returns its URL and revision. Rerunning uploads a new commit to the same repository;
unrelated existing Hub files are not deleted. Use immutable export files while uploading.

The generated README uses standard [model-card metadata](https://huggingface.co/docs/huggingface_hub/guides/model-cards).
It replaces the Hub README as part of the explicit commit. Add fuller documentation on the
Hub afterward if required. Tests simulate Hub operations; no real model is uploaded during CI.
