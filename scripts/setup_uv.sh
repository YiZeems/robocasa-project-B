#!/usr/bin/env bash
# Bootstrap reproducible RoboCasa training environment with uv.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
UV_EXTRA="${UV_EXTRA:-}"
ROBOCASA_COMMIT="${ROBOCASA_COMMIT:-9a3a78680443734786c9784ab661413edb87067b}"
ROBOSUITE_COMMIT="${ROBOSUITE_COMMIT:-aaa8b9b214ce8e77e82926d677b4d61d55e577ab}"
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
  echo "Info: use Git Bash or WSL for this script, not plain cmd.exe."
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not available in PATH."
  exit 1
fi

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

UV_SYNC_ARGS=(sync --python "${PYTHON_VERSION}")
if [ -n "${UV_EXTRA}" ]; then
  UV_SYNC_ARGS+=(--extra "${UV_EXTRA}")
  echo "Syncing optional uv extra: ${UV_EXTRA}"
fi
uv "${UV_SYNC_ARGS[@]}"

INSTALLED_ROBOCASA_DIR="$(uv run python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

spec = find_spec("robocasa")
if spec is None or spec.origin is None:
    raise SystemExit("ERROR: robocasa is not importable after uv sync")
print(Path(spec.origin).resolve().parent)
PY
)"
EXTERNAL_ROBOCASA_DIR="${REPO_ROOT}/external/robocasa/robocasa"
INSTALLED_ASSETS_DIR="${INSTALLED_ROBOCASA_DIR}/models/assets"
EXTERNAL_ASSETS_DIR="${EXTERNAL_ROBOCASA_DIR}/models/assets"

if [ ! -e "${INSTALLED_ASSETS_DIR}" ] && [ -d "${EXTERNAL_ASSETS_DIR}" ]; then
  mkdir -p "$(dirname "${INSTALLED_ASSETS_DIR}")"
  ln -s "${EXTERNAL_ASSETS_DIR}" "${INSTALLED_ASSETS_DIR}"
  echo "Linked RoboCasa assets from external checkout."
fi

if [ ! -e "${INSTALLED_ASSETS_DIR}" ]; then
  echo "WARNING: RoboCasa assets are still missing at ${INSTALLED_ASSETS_DIR}"
fi

if [ "${RUN_SETUP_MACROS}" = "1" ]; then
  if ! yes y | uv run python -m robocasa.scripts.setup_macros; then
    echo "WARNING: robocasa.scripts.setup_macros failed. Continuing."
  fi
fi

if [ "${DOWNLOAD_ASSETS}" = "1" ]; then
  yes y | uv run python -m robocasa.scripts.download_kitchen_assets --type all
fi

if [ "${VERIFY_ASSETS}" = "1" ]; then
  uv run python - <<'PY'
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

if [ "${DOWNLOAD_DATASETS}" = "1" ]; then
  yes y | uv run python -m robocasa.scripts.download_datasets --tasks "${DATASET_TASK}" --source human --split target --overwrite
fi

uv run python - <<'PY'
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
echo "Run with: uv run python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20"
