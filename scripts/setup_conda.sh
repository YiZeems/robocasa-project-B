#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-robocasa_telecom}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ROBOCASA_COMMIT="${ROBOCASA_COMMIT:-9a3a78680443734786c9784ab661413edb87067b}"
ROBOSUITE_COMMIT="${ROBOSUITE_COMMIT:-aaa8b9b214ce8e77e82926d677b4d61d55e577ab}"

RUN_SETUP_MACROS="${RUN_SETUP_MACROS:-1}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
DATASET_TASK="${DATASET_TASK:-OpenSingleDoor}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel

mkdir -p external logs/slurm outputs checkpoints

if [ ! -d external/robosuite/.git ]; then
  git clone https://github.com/ARISE-Initiative/robosuite.git external/robosuite
fi

if [ ! -d external/robocasa/.git ]; then
  git clone https://github.com/robocasa/robocasa.git external/robocasa
fi

git -C external/robosuite fetch --all --tags --prune
git -C external/robosuite checkout "${ROBOSUITE_COMMIT}"

git -C external/robocasa fetch --all --tags --prune
git -C external/robocasa checkout "${ROBOCASA_COMMIT}"

python -m pip install -e external/robosuite
python -m pip install -e external/robocasa
python -m pip install -r requirements-project.txt

if [ "${RUN_SETUP_MACROS}" = "1" ]; then
  yes y | python -m robocasa.scripts.setup_macros || true
fi

if [ "${DOWNLOAD_ASSETS}" = "1" ]; then
  yes y | python -m robocasa.scripts.download_kitchen_assets --type all || true
fi

if [ "${DOWNLOAD_DATASETS}" = "1" ]; then
  yes y | python -m robocasa.scripts.download_datasets --tasks "${DATASET_TASK}" --source human --split target --overwrite || true
fi

if [ "${DOWNLOAD_ASSETS}" != "1" ]; then
  echo "WARNING: RoboCasa assets were not downloaded (DOWNLOAD_ASSETS=${DOWNLOAD_ASSETS})."
  echo "Runtime env reset may fail with missing XML files."
fi

echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
