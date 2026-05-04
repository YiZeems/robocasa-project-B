"""Helpers for saving and resuming SB3 checkpoint artifacts."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CHECKPOINT_STEP_RE = re.compile(r"^(?P<prefix>.+)_(?P<step>\d+)_steps\.zip$")


@dataclass(frozen=True)
class CheckpointArtifact:
    """Resolved checkpoint files for model and optional replay buffer."""

    model_path: Path
    replay_buffer_path: Path
    metadata_path: Path
    step: int | None = None


def checkpoint_step(path: Path) -> int | None:
    """Extract the training step from a checkpoint filename."""

    match = CHECKPOINT_STEP_RE.match(path.name)
    if match is None:
        return None
    try:
        return int(match.group("step"))
    except ValueError:
        return None


def checkpoint_sidecars(model_path: Path) -> tuple[Path, Path]:
    """Return the replay-buffer and metadata sidecar paths for a checkpoint."""

    stem = model_path.with_suffix("")
    return stem.with_name(f"{stem.name}_replay_buffer.pkl"), stem.with_suffix(".json")


def resolve_checkpoint_path(path: str | Path) -> Path:
    """Resolve a checkpoint file or directory to the latest usable checkpoint."""

    candidate = Path(path).expanduser().resolve()
    if candidate.is_file():
        if candidate.suffix == ".json":
            try:
                with candidate.open("r", encoding="utf-8") as f:
                    payload = json.load(f)
                checkpoint_path = payload.get("checkpoint_path")
                if checkpoint_path:
                    resolved = Path(checkpoint_path).expanduser().resolve()
                    if resolved.exists():
                        return resolved
            except Exception:
                pass

            sibling_zip = candidate.with_suffix(".zip")
            if sibling_zip.exists():
                return sibling_zip.resolve()
        return candidate

    if not candidate.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {candidate}")

    if candidate.is_dir():
        checkpoints = list(candidate.glob("*_steps.zip"))
        if not checkpoints:
            for fallback in ("final_model.zip", "best_model.zip"):
                fallback_path = candidate / fallback
                if fallback_path.exists():
                    checkpoints.append(fallback_path)

        if not checkpoints:
            raise FileNotFoundError(
                f"No checkpoint zip found in directory: {candidate}"
            )

        def sort_key(item: Path) -> tuple[int, float, str]:
            step = checkpoint_step(item)
            step_key = step if step is not None else -1
            return (step_key, item.stat().st_mtime, item.name)

        return max(checkpoints, key=sort_key)

    raise FileNotFoundError(f"Unsupported checkpoint path: {candidate}")


def resolve_checkpoint_artifact(path: str | Path) -> CheckpointArtifact:
    """Resolve a checkpoint zip plus its sidecars."""

    model_path = resolve_checkpoint_path(path)
    replay_buffer_path, metadata_path = checkpoint_sidecars(model_path)
    return CheckpointArtifact(
        model_path=model_path,
        replay_buffer_path=replay_buffer_path,
        metadata_path=metadata_path,
        step=checkpoint_step(model_path),
    )


def load_checkpoint_metadata(path: str | Path) -> dict[str, Any]:
    """Load checkpoint metadata when a JSON sidecar exists."""

    artifact = resolve_checkpoint_artifact(path)
    if not artifact.metadata_path.exists():
        return {}

    with artifact.metadata_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, dict) else {}


def save_checkpoint_metadata(
    model_path: str | Path,
    *,
    run_id: str,
    algorithm: str,
    num_timesteps: int,
    source: str,
    save_replay_buffer: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a JSON sidecar next to a checkpoint zip."""

    artifact = resolve_checkpoint_artifact(model_path)
    payload: dict[str, Any] = {
        "run_id": run_id,
        "algorithm": algorithm,
        "num_timesteps": int(num_timesteps),
        "checkpoint_path": str(artifact.model_path),
        "replay_buffer_path": str(artifact.replay_buffer_path),
        "metadata_path": str(artifact.metadata_path),
        "save_replay_buffer": bool(save_replay_buffer),
        "source": source,
    }
    if artifact.step is not None:
        payload["step"] = artifact.step
    if extra:
        payload.update(extra)

    with artifact.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return artifact.metadata_path
