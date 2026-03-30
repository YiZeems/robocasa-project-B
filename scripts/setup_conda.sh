#!/usr/bin/env bash
# Bootstrap reproducible RoboCasa training environment (Conda + pinned repos).
set -euo pipefail

# Environment settings (overridable at invocation time).
ENV_NAME="${ENV_NAME:-robocasa_telecom}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
ROBOCASA_COMMIT="${ROBOCASA_COMMIT:-9a3a78680443734786c9784ab661413edb87067b}"
ROBOSUITE_COMMIT="${ROBOSUITE_COMMIT:-aaa8b9b214ce8e77e82926d677b4d61d55e577ab}"

# Optional setup actions for assets/datasets/macros.
RUN_SETUP_MACROS="${RUN_SETUP_MACROS:-1}"
DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-1}"
DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}"
DATASET_TASK="${DATASET_TASK:-OpenSingleDoor}"
VERIFY_ASSETS="${VERIFY_ASSETS:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create project env only if absent (idempotent script behavior).
if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

# Upgrade packaging tools before editable installs.
python -m pip install --upgrade pip setuptools wheel

mkdir -p external logs/slurm outputs checkpoints

# Clone robosuite if missing, otherwise keep existing checkout and refresh.
if [ ! -d external/robosuite/.git ]; then
  git clone https://github.com/ARISE-Initiative/robosuite.git external/robosuite
fi

# Clone robocasa if missing, otherwise keep existing checkout and refresh.
if [ ! -d external/robocasa/.git ]; then
  git clone https://github.com/robocasa/robocasa.git external/robocasa
fi

# Pin robosuite revision for deterministic reproducibility.
git -C external/robosuite fetch --all --tags --prune
git -C external/robosuite checkout "${ROBOSUITE_COMMIT}"

# Pin robocasa revision for deterministic reproducibility.
git -C external/robocasa fetch --all --tags --prune
git -C external/robocasa checkout "${ROBOCASA_COMMIT}"

# Editable installs make local debugging and patching easier.
python -m pip install -e external/robosuite
python -m pip install -e external/robocasa
python -m pip install -r requirements-project.txt

# RoboCasa setup scripts can be interactive; `yes y` keeps CI/cluster non-interactive.
if [ "${RUN_SETUP_MACROS}" = "1" ]; then
  if ! yes y | python -m robocasa.scripts.setup_macros; then
    echo "WARNING: robocasa.scripts.setup_macros failed. Continuing."
  fi
fi

if [ "${DOWNLOAD_ASSETS}" = "1" ]; then
  # Asset download is critical for runtime: fail fast if this step fails.
  yes y | python -m robocasa.scripts.download_kitchen_assets --type all

  if [ "${VERIFY_ASSETS}" = "1" ]; then
    python - <<'PY'
from pathlib import Path
import sys
import robocasa

assets_root = Path(robocasa.__path__[0]) / "models" / "assets"
sample_files = [
    assets_root / "objects" / "lightwheel" / "stool" / "Stool013" / "model.xml",
    assets_root / "objects" / "lightwheel" / "stool" / "Stool019" / "model.xml",
]
for file in sample_files:
    if not file.exists():
        print(f"ERROR: Expected asset file is missing: {file}")
        sys.exit(1)

# Also validate that key asset directories are populated.
required_dirs = [
    assets_root / "fixtures",
    assets_root / "objects" / "objaverse",
    assets_root / "objects" / "lightwheel",
]
for directory in required_dirs:
    if not directory.exists():
        print(f"ERROR: Missing asset directory: {directory}")
        sys.exit(1)

    xml_count = sum(1 for _ in directory.rglob("model.xml"))
    if xml_count == 0:
        print(f"ERROR: No model.xml found under asset directory: {directory}")
        sys.exit(1)

print("Asset validation passed.")
PY
  fi
fi

if [ "${DOWNLOAD_DATASETS}" = "1" ]; then
  # If datasets were explicitly requested, surface failures.
  yes y | python -m robocasa.scripts.download_datasets --tasks "${DATASET_TASK}" --source human --split target --overwrite
fi

if [ "${DOWNLOAD_ASSETS}" != "1" ]; then
  echo "WARNING: RoboCasa assets were not downloaded (DOWNLOAD_ASSETS=${DOWNLOAD_ASSETS})."
  echo "Runtime env reset may fail with missing XML files."
fi

echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
