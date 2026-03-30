"""Shared IO helpers for configuration and filesystem setup."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML file and enforce mapping-like root object."""

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def ensure_dir(path: str | Path) -> Path:
    """Create directory recursively if needed and return normalized Path."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
