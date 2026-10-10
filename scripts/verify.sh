#!/usr/bin/env bash
set -euo pipefail
python -m modelchef.cli check-config configs/local.yaml
python -m modelchef.cli doctor
python -m pytest -q
