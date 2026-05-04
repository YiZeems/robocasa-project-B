#!/usr/bin/env bash
# Run any command inside the project uv environment without manual activation.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

AUTO_SETUP="${AUTO_SETUP:-1}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"

UNAME_S="$(uname -s 2>/dev/null || echo unknown)"
case "${UNAME_S}" in
  Linux*) OS_FAMILY="linux" ;;
  Darwin*) OS_FAMILY="macos" ;;
  MINGW*|MSYS*|CYGWIN*) OS_FAMILY="windows" ;;
  *) OS_FAMILY="unknown" ;;
esac

if ! command -v uv >/dev/null 2>&1; then
  echo "Error: uv is not available in PATH."
  if [ "${OS_FAMILY}" = "windows" ]; then
    echo "Hint: run from Git Bash or WSL where uv is installed and available."
  fi
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

if [ ! -x ".venv/bin/python" ] && [ ! -x ".venv/Scripts/python.exe" ]; then
  if [ "${AUTO_SETUP}" = "1" ]; then
    echo "uv environment not found. Running setup..."
    DOWNLOAD_ASSETS="${DOWNLOAD_ASSETS:-0}" VERIFY_ASSETS="${VERIFY_ASSETS:-0}" RUN_SETUP_MACROS="${RUN_SETUP_MACROS:-0}" DOWNLOAD_DATASETS="${DOWNLOAD_DATASETS:-0}" PYTHON_VERSION="${PYTHON_VERSION}" bash scripts/setup_uv.sh
  else
    echo "Error: uv environment not found. Run: bash scripts/setup_uv.sh"
    exit 1
  fi
fi

exec uv run --python "${PYTHON_VERSION}" "$@"
