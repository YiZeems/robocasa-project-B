from __future__ import annotations

from typing import Any


def infer_success(info: dict[str, Any] | None, env: Any | None = None) -> bool:
    payload = dict(info or {})
    for key in ("success", "task_success", "is_success"):
        if key in payload:
            return bool(payload[key])

    probe = env
    for _ in range(10):
        if probe is None:
            break

        raw_env = getattr(probe, "raw_env", None)
        if raw_env is not None and hasattr(raw_env, "_check_success"):
            try:
                return bool(raw_env._check_success())
            except Exception:
                pass

        if hasattr(probe, "_check_success"):
            try:
                return bool(probe._check_success())
            except Exception:
                pass

        probe = getattr(probe, "env", None)

    return False
