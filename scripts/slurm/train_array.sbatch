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

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm

ENV_NAME="${ENV_NAME:-robocasa_telecom}"
CONFIG_PATH="${CONFIG_PATH:-configs/train/open_single_door_ppo.yaml}"
SEEDS_CSV="${SEEDS_CSV:-0,1,2}"

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV}"
IDX="${SLURM_ARRAY_TASK_ID:-0}"

if [ "${IDX}" -ge "${#SEEDS[@]}" ]; then
  echo "SLURM_ARRAY_TASK_ID=${IDX} out of range for seeds: ${SEEDS_CSV}"
  exit 1
fi

SEED="${SEEDS[$IDX]}"
echo "Running seed ${SEED} (array index ${IDX})"

PYTHONPATH=src python -m robocasa_telecom.train --config "${CONFIG_PATH}" --seed "${SEED}"
