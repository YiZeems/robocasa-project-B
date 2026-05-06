#!/usr/bin/env bash
#SBATCH --job-name=robocasa_render
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/%x-%A.out
#SBATCH --error=logs/slurm/%x-%A.err

# SLURM job to render a 4-view MP4 from best_model.zip.
# Examples:
#   sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run>/ scripts/slurm/render_best_run.sbatch
#   sbatch --export=ALL,CONFIG_PATH=configs/train/open_single_door_sac.yaml,\
# CHECKPOINT_PATH=checkpoints/<run>/best_model.zip,SEED=0 scripts/slurm/render_best_run.sbatch
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs/slurm outputs/eval/videos

CONFIG_PATH="${CONFIG_PATH:-configs/train/open_single_door_sac.yaml}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"
SEED="${SEED:-0}"
MAX_STEPS="${MAX_STEPS:-500}"
OUTPUT_PATH="${OUTPUT_PATH:-}"

if [ -z "${CHECKPOINT_PATH}" ]; then
  echo "Please provide CHECKPOINT_PATH, e.g.:"
  echo "sbatch --export=ALL,CHECKPOINT_PATH=checkpoints/<run>/ scripts/slurm/render_best_run.sbatch"
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

export UV_PROJECT_ENVIRONMENT="${SLURM_SUBMIT_DIR:-$PWD}/.venv"
uv sync --python "${PYTHON_VERSION:-3.11}" >/dev/null
echo "Config: ${CONFIG_PATH} | Checkpoint: ${CHECKPOINT_PATH} | Seed: ${SEED}"

CMD=(
  uv run python -m robocasa_telecom.render_best_run
  --config "${CONFIG_PATH}"
  --checkpoint "${CHECKPOINT_PATH}"
  --seed "${SEED}"
  --max-steps "${MAX_STEPS}"
)

if [ -n "${OUTPUT_PATH}" ]; then
  CMD+=(--output "${OUTPUT_PATH}")
fi

"${CMD[@]}"
