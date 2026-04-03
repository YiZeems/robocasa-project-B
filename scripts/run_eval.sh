#!/usr/bin/env bash
# Local wrapper: evaluate checkpoint via with_env (cross-shell / OS-friendly).
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

exec scripts/with_env.sh python -m robocasa_telecom.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --deterministic
