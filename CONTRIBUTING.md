# Working on ModelChef

Start with the [offline starter recipe](configs/pretrain-demo.yaml). It exercises data preparation, tokenizer training, checkpoints and evaluation using included text. The [starter workflow](configs/workflow-demo.yaml) adds SFT and export. Neither needs datasets, model downloads or AWS after dependency installation.

## Prepare the environment

From the repository root, follow the README's hardware setup, then install the checks and any feature extras you need:

```bash
uv pip install -e '.[dev,pipeline,pretrained,peft,aws]'
python -m modelchef.cli doctor
python -m ruff check src tests scripts/check_docs.py
python scripts/check_docs.py
python -m pytest -q
```

Choose the PyTorch wheel using `setup`/`setup --apply`; feature extras do not choose a GPU driver or wheel for you. CPU-only CI covers the complete small pipeline, native/imported fine-tuning, export and two-process distributed recovery. Windows CI checks the controller without installing GPU libraries. CUDA, MPS, ROCm, vLLM and Blackwell behavior need tests on the corresponding hardware; a skipped test is not evidence of success there.

## Make a change reviewable

Describe the user-visible problem, the changed behavior and relevant verification. Add a regression test for a bug or new execution path. Prefer small local fixtures over network downloads, and mock cloud operations instead of provisioning resources. Keep recipe fields, generated schemas and the annotated README recipe in agreement; `scripts/check_docs.py` checks that agreement.

Run artifacts, caches, credentials, certificates and personal recipes stay out of Git. Use `configs/*.local.yaml` for personal settings. Include commands and limitations in a PR, not private run logs or account identifiers. Hardware tests and spending require explicit opt-in; ordinary tests must not download models or create cloud resources.

## Report a problem

For ordinary bugs, include the command, relevant recipe fragment, Python/package versions, device and a sanitized error. Use `--debug` locally for a traceback. Report security problems through the private route described in [SECURITY.md](SECURITY.md).
