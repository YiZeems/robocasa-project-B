"""Cross-platform smoke tests — no GPU, no MuJoCo required.

Run on every OS before pushing:
    pytest tests/test_platform_smoke.py -v
"""

from __future__ import annotations

import os
import platform
import sys


def test_resolve_device_returns_string() -> None:
    from robocasa_telecom.utils.device import resolve_device

    for pref in ("auto", "cpu", "cuda", "mps"):
        result = resolve_device(pref)
        assert isinstance(result, str), f"resolve_device({pref!r}) returned {type(result)}"
        assert result in ("cpu", "cuda", "mps"), f"unexpected device: {result}"


def test_resolve_device_auto_never_crashes() -> None:
    from robocasa_telecom.utils.device import resolve_device

    device = resolve_device("auto")
    assert device in ("cpu", "cuda", "mps")


def test_resolve_device_cpu_always_works() -> None:
    from robocasa_telecom.utils.device import resolve_device

    assert resolve_device("cpu") == "cpu"


def test_mujoco_gl_set_after_factory_import() -> None:
    # Import factory to trigger the MUJOCO_GL auto-detection block.
    # The env var must be set to a non-empty string after import.
    os.environ.pop("MUJOCO_GL", None)
    import importlib
    import robocasa_telecom.envs.factory as factory_mod
    importlib.reload(factory_mod)

    assert "MUJOCO_GL" in os.environ
    assert os.environ["MUJOCO_GL"] != ""


def test_mujoco_gl_correct_for_platform() -> None:
    os.environ.pop("MUJOCO_GL", None)
    import importlib
    import robocasa_telecom.envs.factory as factory_mod
    importlib.reload(factory_mod)

    gl = os.environ.get("MUJOCO_GL", "")
    if sys.platform == "linux":
        assert gl == "egl", f"expected egl on Linux, got {gl}"
    elif sys.platform == "darwin":
        assert gl == "cgl", f"expected cgl on macOS, got {gl}"
    elif sys.platform.startswith("win"):
        assert gl == "wgl", f"expected wgl on Windows, got {gl}"


def test_mujoco_gl_not_overridden_if_set() -> None:
    os.environ["MUJOCO_GL"] = "osmesa"
    import importlib
    import robocasa_telecom.envs.factory as factory_mod
    importlib.reload(factory_mod)

    assert os.environ["MUJOCO_GL"] == "osmesa"
    os.environ.pop("MUJOCO_GL", None)


def test_pathlib_paths_portable() -> None:
    from pathlib import Path
    from robocasa_telecom.utils.io import ensure_dir
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        p = ensure_dir(Path(tmp) / "subdir" / "nested")
        assert p.exists()
        assert p.is_dir()


def test_mlflow_uri_is_absolute() -> None:
    """Verify the mlruns URI produced by Path.resolve().as_uri() is absolute."""
    from pathlib import Path

    uri = Path("mlruns").resolve().as_uri()
    assert uri.startswith("file:///") or uri.startswith("file://"), f"not absolute: {uri}"
    # Must not be a bare relative path
    assert "mlruns" in uri


def test_imageio_ffmpeg_importable() -> None:
    import imageio_ffmpeg  # noqa: F401


def test_stable_baselines3_importable() -> None:
    from stable_baselines3 import PPO, SAC  # noqa: F401


def test_torch_importable() -> None:
    import torch  # noqa: F401
    assert hasattr(torch, "tensor")
