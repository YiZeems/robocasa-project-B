#!/usr/bin/env bash
# Local wrapper: run training via with_env (cross-shell / OS-friendly).
set -euo pipefail

CONFIG_PATH="${1:-configs/train/open_single_door_ppo.yaml}"
SEED="${2:-0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "${REPO_ROOT}"

exec scripts/with_env.sh python -m robocasa_telecom.train --config "${CONFIG_PATH}" --seed "${SEED}"
