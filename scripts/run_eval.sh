#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_PATH="${1:-}"
CONFIG_PATH="${2:-configs/train/open_single_door_ppo.yaml}"
NUM_EPISODES="${3:-20}"
ENV_NAME="${ENV_NAME:-robocasa_telecom}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Usage: $0 <checkpoint_path> [config_path] [num_episodes]"
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

PYTHONPATH="${REPO_ROOT}" python -m robocasa_telecom.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --deterministic
