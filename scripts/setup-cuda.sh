#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
uv pip install 'torch==2.13.0' --torch-backend=cu130
uv pip install 'numpy==2.3.5'
mkdir -p artifacts/environment
uv run --no-sync python -m ashayllm.cli doctor | tee artifacts/environment/doctor.json
uv run --no-sync python scripts/cuda_smoke.py | tee artifacts/environment/cuda-smoke.json
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv pip check
uv pip freeze > requirements-cuda.snapshot.txt
