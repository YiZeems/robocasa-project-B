#!/usr/bin/env bash
# Local wrapper: render the best checkpoint from a run directory as a 4-view MP4.
set -euo pipefail

CHECKPOINT_PATH="${1:-}"
CONFIG_PATH="${2:-configs/train/open_single_door_sac.yaml}"
SEED="${3:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Usage: $0 <checkpoint_or_run_dir> [config_path] [seed]"
  exit 1
fi

cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

exec uv run python -m robocasa_telecom.render_best_run \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --seed "${SEED}"
