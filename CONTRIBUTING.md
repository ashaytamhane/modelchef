# Working on ModelChef

Start with the [offline starter recipe](configs/pretrain-demo.yaml). It exercises data preparation, tokenizer training, checkpoints and evaluation using included text. The [starter workflow](configs/workflow-demo.yaml) adds SFT and export. Neither needs datasets, model downloads or AWS after dependency installation.

## Prepare the environment

From the repository root, follow the README's hardware setup, then install the checks and any feature extras you need:

```bash
uv pip install -e '.[dev,pipeline,pretrained,peft,aws,multimodal]'
python -m modelchef.cli doctor
python -m ruff check src tests scripts/check_docs.py
python scripts/check_docs.py
python -m pytest -q
```

Choose the PyTorch wheel using `setup`/`setup --apply`; feature extras do not choose a GPU driver or wheel for you. CPU-only CI covers the complete small pipeline, native/imported fine-tuning, export and two-process distributed recovery. Windows CI checks the controller without installing GPU libraries. CUDA, MPS, ROCm, vLLM and Blackwell behavior need tests on the corresponding hardware; a skipped test is not evidence of success there.

## Make a change reviewable

Describe the user-visible problem, the changed behavior and relevant verification. Add a regression test for a bug or new execution path. Prefer small local fixtures over network downloads, and mock cloud operations instead of provisioning resources. Keep recipe fields, generated schemas and the annotated README recipe in agreement; `scripts/check_docs.py` checks that agreement.

Run artifacts, caches, credentials, certificates and personal recipes stay out of Git. Use `configs/*.local.yaml` for personal settings. Include commands and limitations in a PR, not private run logs or account identifiers. Hardware tests skip when their device or optional toolchain is unavailable; spending and model downloads require explicit opt-in. Ordinary tests must not download models or create cloud resources.

Multimodal tests create tiny random PaliGemma/Whisper models and synthetic media locally, then exercise training, recovery, export and inference. Publishing tests mock Hub calls and check the exact file allowlist, visibility and authentication behavior; they never upload a model. Keep the [multimodal](docs/MULTIMODAL.md) and [publishing](docs/PUBLISHING.md) tutorials and their schemas synchronized with new settings.

## Report a problem

For ordinary bugs, include the command, relevant recipe fragment, Python/package versions, device and a sanitized error. Use `--debug` locally for a traceback. Report security problems through the private route described in [SECURITY.md](SECURITY.md).
