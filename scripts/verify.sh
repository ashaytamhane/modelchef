#!/usr/bin/env bash
set -euo pipefail
python -m loomlm.cli check-config configs/local.yaml
python -m loomlm.cli doctor
python -m pytest -q
