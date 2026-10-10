#!/usr/bin/env bash
set -euo pipefail
linux_user="${1:?Usage: sudo bash scripts/setup-wsl-user.sh USERNAME}"
if [[ ! "$linux_user" =~ ^[a-z_][a-z0-9_-]*$ ]]; then
  echo 'Use a valid Linux username.' >&2
  exit 1
fi
if ! id "$linux_user" >/dev/null 2>&1; then
  useradd --create-home --shell /bin/bash "$linux_user"
fi
usermod --append --groups sudo "$linux_user"
if ! grep -q '^\[user\]' /etc/wsl.conf; then
  printf '\n[user]\ndefault=%s\n' "$linux_user" >> /etc/wsl.conf
else
  echo 'Existing WSL user section found; left unchanged.'
fi
id "$linux_user"
