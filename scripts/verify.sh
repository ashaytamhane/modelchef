#!/usr/bin/env bash
set -euo pipefail
python -m ashayllm.cli check-config configs/local.yaml
python -m ashayllm.cli doctor
python -m pytest -q
