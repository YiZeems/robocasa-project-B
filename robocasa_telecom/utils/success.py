"""Utilities to infer task success across heterogeneous wrapper stacks."""

from __future__ import annotations

from typing import Any


def infer_success(info: dict[str, Any] | None, env: Any | None = None) -> bool:
    """Infer episode success from `info` and wrapped env internals.

    Priority:
    1) explicit success keys in info dict,
    2) `_check_success()` on raw env,
    3) `_check_success()` by traversing wrapper chain.
    """

    payload = dict(info or {})
    for key in ("success", "task_success", "is_success"):
        if key in payload:
            return bool(payload[key])

    probe = env
    for _ in range(10):
        if probe is None:
            break

        # First try raw env handle if adapter exposes it.
        raw_env = getattr(probe, "raw_env", None)
        if raw_env is not None and hasattr(raw_env, "_check_success"):
            try:
                return bool(raw_env._check_success())
            except Exception:
                pass

        # Then try on the currently probed wrapper itself.
        if hasattr(probe, "_check_success"):
            try:
                return bool(probe._check_success())
            except Exception:
                pass

        # Move one wrapper deeper (gym Monitor / wrapper stacks).
        probe = getattr(probe, "env", None)

    return False
