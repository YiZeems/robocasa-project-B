#!/usr/bin/env bash
# Local wrapper: evaluate checkpoint via uv run.
set -euo pipefail

CHECKPOINT_PATH="${1:-}"
CONFIG_PATH="${2:-configs/train/open_single_door_ppo.yaml}"
NUM_EPISODES="${3:-20}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Usage: $0 <checkpoint_path> [config_path] [num_episodes]"
  exit 1
fi

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

exec uv run python -m robocasa_telecom.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --deterministic
