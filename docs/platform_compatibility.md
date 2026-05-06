# Platform Compatibility

## Support matrix

| Feature | macOS Apple Silicon (M1/M2/M3) | Windows 11 CUDA | WSL2 / Linux CUDA |
|---|---|---|---|
| Training (SAC/PPO) | ✅ MPS or CPU | ✅ CUDA | ✅ CUDA |
| Device auto-detection | ✅ `resolve_device("auto")` → mps | ✅ → cuda | ✅ → cuda |
| Parallel workers (`SubprocVecEnv`) | ✅ spawn | ✅ spawn | ✅ spawn |
| MuJoCo rendering | ✅ `MUJOCO_GL=cgl` | ⚠️ `MUJOCO_GL=wgl` (see note) | ✅ `MUJOCO_GL=egl` |
| Video export (MP4) | ✅ imageio-ffmpeg | ✅ imageio-ffmpeg | ✅ imageio-ffmpeg |
| MLflow UI | ✅ | ✅ | ✅ |
| `uv sync` | ✅ | ⚠️ needs extra steps (see below) | ✅ |

---

## macOS Apple Silicon

### Prerequisites

```bash
brew install cmake   # if not already present
pip install uv       # or: curl -Ls https://astral.sh/uv/install.sh | sh
```

### Install

```bash
git clone --recurse-submodules <repo-url>
cd robocasa-project-B
uv sync
```

### Train

```bash
uv run robocasa-telecom-train --config configs/train/open_single_door_sac.yaml
```

Device resolution: `resolve_device("auto")` checks `torch.backends.mps.is_available()` and returns `"mps"` on Apple Silicon. SB3's built-in `get_device("auto")` does **not** pick MPS — this project works around that.

### Notes

- `MUJOCO_GL` is auto-set to `cgl` (CoreGL) by `factory.py` at import time.
- To use CPU instead of MPS: `--device cpu` is not yet a CLI flag, but you can set `device: cpu` in your YAML or `export PYTORCH_ENABLE_MPS_FALLBACK=1` for unsupported ops.
- 12 parallel workers (`n_envs: 12`) work on Apple Silicon via `SubprocVecEnv(start_method="spawn")`.

---

## Windows 11 (NVIDIA CUDA)

### Prerequisites

1. **Visual C++ Build Tools 2022** (required by MuJoCo): [download](https://visualstudio.microsoft.com/visual-cpp-build-tools/)
2. **CUDA Toolkit 12.x**: [download](https://developer.nvidia.com/cuda-downloads)
3. **Python 3.11 or 3.12**: [download](https://www.python.org/downloads/windows/)
4. **uv**: `pip install uv`

### Install

```powershell
git clone --recurse-submodules <repo-url>
cd robocasa-project-B
# uv.lock was generated on Linux/macOS — regenerate for Windows:
uv lock
uv sync
```

### Train (PowerShell)

```powershell
uv run robocasa-telecom-train --config configs/train/open_single_door_sac.yaml
```

### Debug (single process, no spawn)

```powershell
uv run robocasa-telecom-train `
  --config configs/train/open_single_door_sac_debug.yaml `
  --vec-env dummy
```

### Notes

- **MuJoCo GL**: `MUJOCO_GL=wgl` is set automatically. If rendering fails (no display), try:
  ```powershell
  $env:MUJOCO_GL = "osmesa"
  uv run robocasa-telecom-train ...
  ```
- **SubprocVecEnv on Windows**: `start_method="spawn"` is the Windows default and works correctly. If you encounter `PicklingError`, use `--vec-env dummy` to debug.
- **MLflow UI**:
  ```powershell
  uv run mlflow ui --backend-store-uri (Resolve-Path mlruns).Path
  ```

---

## WSL2 / Linux (NVIDIA CUDA)

### Prerequisites

```bash
# NVIDIA drivers + CUDA toolkit installed on the host
nvidia-smi  # verify GPU visible
sudo apt install cmake libgl1-mesa-glx   # EGL/osmesa deps
pip install uv
```

### Install

```bash
git clone --recurse-submodules <repo-url>
cd robocasa-project-B
uv sync
```

### Train

```bash
uv run robocasa-telecom-train --config configs/train/open_single_door_sac.yaml
```

### Headless rendering

`MUJOCO_GL=egl` is set automatically by `factory.py`. EGL requires a NVIDIA driver with EGL support. If unavailable:

```bash
MUJOCO_GL=osmesa uv run robocasa-telecom-train ...
```

### MLflow UI

```bash
uv run mlflow ui --backend-store-uri $(realpath mlruns)
```

---

## Smoke tests (all platforms)

Run before any experiment to verify the environment is correctly set up:

```bash
pytest tests/test_platform_smoke.py -v
```

Expected: **11 passed** in under 5 seconds. No GPU or MuJoCo scene required.

---

## CLI flags for portability

| Flag | Purpose | Example |
|---|---|---|
| `--vec-env dummy` | Single-process VecEnv (debug, Windows) | `--vec-env dummy` |
| `--vec-env subproc` | Parallel workers (default) | `--vec-env subproc` |
| `--n-envs 2` | Override worker count (must be even) | `--n-envs 2` |

The `device` is set in the YAML (`device: auto`) and resolved automatically — no CLI flag needed.

---

## Environment variables

| Variable | Values | Set by |
|---|---|---|
| `MUJOCO_GL` | `egl`, `cgl`, `wgl`, `osmesa` | `factory.py` (auto) or user |
| `PYTORCH_ENABLE_MPS_FALLBACK` | `1` | User (macOS, for ops not supported on MPS) |
| `ROBOCASA_RENDER_BEST_RUN_VIDEO` | `1` | User (enable post-training best-run video) |
