#!/usr/bin/env bash
# Local wrapper: run training with the same env activation logic as SLURM jobs.
set -euo pipefail

CONFIG_PATH="${1:-configs/train/open_single_door_ppo.yaml}"
SEED="${2:-0}"
ENV_NAME="${ENV_NAME:-robocasa_telecom}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"
cd "${REPO_ROOT}"

python -m robocasa_telecom.train --config "${CONFIG_PATH}" --seed "${SEED}"
