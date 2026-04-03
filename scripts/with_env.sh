#!/usr/bin/env bash
# Run any command inside the project Conda environment without `conda activate`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

ENV_NAME="${ENV_NAME:-robocasa_telecom}"
AUTO_SETUP="${AUTO_SETUP:-1}"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda is not available in PATH."
  exit 1
fi

PYTHON_BIN="$(command -v python || command -v python3 || true)"
if [ -z "${PYTHON_BIN}" ]; then
  echo "Error: neither 'python' nor 'python3' is available in PATH."
  exit 1
fi

if [ "$#" -eq 0 ]; then
  echo "Usage: scripts/with_env.sh <command> [args...]"
  echo "Example: scripts/with_env.sh python -m robocasa_telecom.sanity --config configs/env/open_single_door.yaml --steps 20"
  exit 1
fi

if ! conda env list --json | "${PYTHON_BIN}" -c "import json,sys; d=json.load(sys.stdin); names={p.rstrip('/').split('/')[-1] for p in d.get('envs', [])}; sys.exit(0 if '${ENV_NAME}' in names else 1)"; then
  if [ "${AUTO_SETUP}" = "1" ]; then
    echo "Conda env '${ENV_NAME}' not found. Running setup..."
    ENV_NAME="${ENV_NAME}" DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}" VERIFY_ASSETS="${VERIFY_ASSETS:-0}" RUN_SETUP_MACROS="${RUN_SETUP_MACROS:-0}" DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}" bash scripts/setup_conda.sh
  else
    echo "Error: conda env '${ENV_NAME}' not found. Run: ENV_NAME=${ENV_NAME} bash scripts/setup_conda.sh"
    exit 1
  fi
fi

exec conda run -n "${ENV_NAME}" "$@"
