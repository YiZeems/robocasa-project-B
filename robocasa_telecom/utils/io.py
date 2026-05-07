
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _resolve_existing_path(path: str | Path) -> Path:

    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate

    if not candidate.is_absolute():
                                                                 
        repo_root = Path(__file__).resolve().parents[2]
        repo_relative = repo_root / candidate
        if repo_relative.exists():
            return repo_relative

    raise FileNotFoundError(f"Path not found: {candidate}")


def load_yaml(path: str | Path) -> dict[str, Any]:

    path = _resolve_existing_path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def ensure_dir(path: str | Path) -> Path:

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
