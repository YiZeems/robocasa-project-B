#!/usr/bin/env bash
#SBATCH --job-name=robocasa_train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=logs/slurm/%x-%A_%a.out
#SBATCH --error=logs/slurm/%x-%A_%a.err

# SLURM array launcher: one seed per array index.
# Algorithm is selected via the YAML pointed to by CONFIG_PATH (PPO or SAC).
# Examples:
#   sbatch scripts/slurm/train_array.sbatch
#   sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac.yaml \
#          scripts/slurm/train_array.sbatch
#   sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac_tuned.yaml,SEEDS_CSV=0,1,2 \
#          scripts/slurm/train_array.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

CONFIG_PATH="${CONFIG_PATH:-configs/train/open_single_door_ppo.yaml}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"
RESUME_FROM="${RESUME_FROM:-}"
echo "Config: ${CONFIG_PATH} | Seeds: ${SEEDS_CSV}"

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${SLURM_SUBMIT_DIR:-$PWD}/.venv"
UV_SYNC_ARGS=(sync --python "${PYTHON_VERSION:-3.11}")
if [ -n "${UV_EXTRA:-cuda}" ]; then
  UV_SYNC_ARGS+=(--extra "${UV_EXTRA:-cuda}")
fi
uv "${UV_SYNC_ARGS[@]}" >/dev/null

# Convert comma-separated seed list into indexed bash array.
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${IDX}" -ge "${#SEEDS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range for seeds: ${SEEDS_CSV}"
  exit 1
fi

SEED="${SEEDS[$IDX]}"
echo "Running seed ${SEED} (array index ${IDX})"

if [ -n "${RESUME_FROM}" ]; then
  echo "Resuming from ${RESUME_FROM}"
  uv run python -m robocasa_telecom.train \
    --config "${CONFIG_PATH}" \
    --seed "${SEED}" \
    --resume-from "${RESUME_FROM}"
else
  uv run python -m robocasa_telecom.train --config "${CONFIG_PATH}" --seed "${SEED}"
fi
