"""Video export helpers for RoboCasa evaluation runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np


def ensure_uint8_frame(frame: Any) -> np.ndarray:
    """Normalize a rendered frame to uint8 HxWx3."""

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    if arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def grid_2x2(frames: list[np.ndarray]) -> np.ndarray:
    """Compose four frames into a 2x2 mosaic."""

    if len(frames) != 4:
        raise ValueError(f"Expected exactly 4 frames, got {len(frames)}")

    top = np.concatenate([frames[0], frames[1]], axis=1)
    bottom = np.concatenate([frames[2], frames[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def save_mp4(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> Path:
    """Write frames to an MP4 file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)
    return output_path

