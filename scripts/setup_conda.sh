#!/usr/bin/env bash
# Bootstrap reproducible RoboCasa training environment (Conda + pinned repos).
set -euo pipefail

# Always execute from repository root (script is safe to call from any cwd).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

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

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
case "${UNAME_S}" in
  Linux*) OS_FAMILY="linux" ;;
  Darwin*) OS_FAMILY="macos" ;;
  MINGW*|MSYS*|CYGWIN*) OS_FAMILY="windows" ;;
  *) OS_FAMILY="unknown" ;;
esac

echo "Detected OS: ${OS_FAMILY} (${UNAME_S})"
if [ "${OS_FAMILY}" = "windows" ]; then
  echo "Info: use Git Bash or WSL for this script (not plain cmd.exe)."
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: neither 'python' nor 'python3' is available in PATH."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"

# Create project env only if absent (idempotent script behavior).
if ! conda env list --json | "${PYTHON_BIN}" -c "import json,sys; d=json.load(sys.stdin); names={p.replace('\\\\','/').rstrip('/').split('/')[-1] for p in d.get('envs', [])}; sys.exit(0 if '${ENV_NAME}' in names else 1)"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"

# Adapt existing env if Python version does not match expected major.minor.
CURRENT_PY_MM="$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
if [ "${CURRENT_PY_MM}" != "${PYTHON_VERSION}" ]; then
  conda install -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
  conda activate "${ENV_NAME}"
fi

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
# Install this repository package so `python -m robocasa_telecom.*` works from any cwd.
python -m pip install -e .

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

# Verify critical runtime imports in the target environment.
python - <<'PY'
import importlib
required = [
    "robocasa_telecom",
    "robosuite",
    "robosuite_models",
    "robocasa",
    "gymnasium",
    "stable_baselines3",
]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"ERROR: Missing required imports after setup: {missing}")
print("Import validation passed.")
PY

echo "Setup complete."
echo "Activate with: conda activate ${ENV_NAME}"
echo "Or run without activate: scripts/with_env.sh python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20"
