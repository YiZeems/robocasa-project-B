"""Cross-platform device resolution for PyTorch / Stable-Baselines3.

SB3 v2.3.x get_device("auto") returns "cpu" on macOS even when MPS is
available. This helper picks cuda > mps > cpu in "auto" mode and falls
back gracefully when an explicit choice is unavailable on the host.
"""

from __future__ import annotations

import platform
import warnings

import torch


def resolve_device(preference: str | None = "auto") -> str:
    """Return a torch device string usable by SB3 across macOS/Linux/Windows."""

    pref = (preference or "auto").lower()

    if pref == "cpu":
        return "cpu"

    cuda_ok = torch.cuda.is_available()
    mps_backend = getattr(torch.backends, "mps", None)
    mps_ok = (
        platform.system() == "Darwin"
        and mps_backend is not None
        and mps_backend.is_available()
    )

    if pref == "cuda":
        if cuda_ok:
            return "cuda"
        warnings.warn("CUDA requested but unavailable; falling back to CPU.")
        return "cpu"

    if pref == "mps":
        if mps_ok:
            return "mps"
        warnings.warn("MPS requested but unavailable; falling back to CPU.")
        return "cpu"

    if cuda_ok:
        return "cuda"
    if mps_ok:
        return "mps"
    return "cpu"
