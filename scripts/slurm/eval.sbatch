#!/usr/bin/env bash
#SBATCH --job-name=robocasa_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/%x-%A.out
#SBATCH --error=logs/slurm/%x-%A.err

# SLURM evaluation job for a provided checkpoint path.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm outputs/eval
REPO_ROOT="$(pwd)"

CONFIG_PATH="${CONFIG_PATH:-configs/train/open_single_door_ppo.yaml}"
NUM_EPISODES="${NUM_EPISODES:-20}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Please provide CHECKPOINT_PATH, e.g.:"
  echo "sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run>/final_model.zip scripts/slurm/eval.sbatch"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${SLURM_SUBMIT_DIR:-$PWD}/.venv"
uv sync --python "${PYTHON_VERSION:-3.11}" >/dev/null

uv run python -m robocasa_telecom.evaluate \
  --config "${CONFIG_PATH}" \
  --checkpoint "${CHECKPOINT_PATH}" \
  --num-episodes "${NUM_EPISODES}" \
  --deterministic
