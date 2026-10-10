#!/usr/bin/env bash
set -euo pipefail
if [ "$(id -u)" -eq 0 ]; then
  echo 'Run this setup as your regular Linux user.' >&2
  exit 1
fi
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl --proto '=https' --tlsv1.2 --connect-timeout 15 --max-time 120 -LsSf https://astral.sh/uv/install.sh -o /tmp/loomlm-uv-install.sh
  sh /tmp/loomlm-uv-install.sh
fi
mkdir -p "$HOME/projects"
project_dir="$HOME/projects/loomlm"
if [ ! -e "$project_dir" ]; then
  source_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  mkdir -p "$project_dir"
  tar -C "$source_dir" --exclude='.venv*' --exclude='.git' --exclude='.aws*' \
    --exclude='artifacts' --exclude='*.local.yaml' -cf - . | tar -C "$project_dir" -xf -
fi
cd "$project_dir"
uv python install 3.12
uv venv --python 3.12
uv pip install -e '.[dev]'
uv pip freeze > requirements-dev.lock.txt
uv run --no-sync python -m loomlm.cli check-config configs/local.yaml
uv run --no-sync python -m pytest -q
uv run --no-sync ruff check .
uv run --no-sync python -m loomlm.cli doctor
