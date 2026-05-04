#!/usr/bin/env bash
# Local wrapper: run training via uv run.
set -euo pipefail

CONFIG_PATH="${1:-configs/train/open_single_door_ppo.yaml}"
SEED="${2:-0}"
RESUME_FROM="${3:-${RESUME_FROM:-}}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

if [ -n "${RESUME_FROM}" ]; then
  exec uv run python -m robocasa_telecom.train \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --resume-from "${RESUME_FROM}"
fi

exec uv run python -m robocasa_telecom.train --config "${CONFIG_PATH}" --seed "${SEED}"
