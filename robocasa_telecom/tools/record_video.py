
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import A2C, PPO, SAC

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.io import load_yaml
from ..utils.success import infer_success

ALGO_MAP = {"PPO": PPO, "SAC": SAC, "A2C": A2C}

                                                                
_CAMERA_FALLBACKS = [
    "robot0_frontview",
    "robot0_agentview_center",
    "robot0_agentview_left",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record evaluation video of a trained RoboCasa agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to record")
    parser.add_argument("--output", type=str, default="eval_video.mp4", help="Output .mp4 path")
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=None,
        metavar="CAMERA",
        help=(
            "One or more camera names. Single camera → normal video. "
            "Multiple cameras → tiled side-by-side. "
            "Default: robot0_frontview"
        ),
    )
    parser.add_argument("--width", type=int, default=512, help="Width per camera tile (pixels)")
    parser.add_argument("--height", type=int, default=512, help="Height per camera tile (pixels)")
    parser.add_argument("--fps", type=int, default=20, help="Output video FPS")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _get_sim(env: Any) -> Any:
    for _ in range(10):
        sim = getattr(env, "sim", None)
        if sim is not None:
            return sim
                                                                
        inner = getattr(env, "_env", None) or getattr(env, "raw_env", None)
        if inner is None:
            break
        env = inner
    raise RuntimeError("Cannot reach env.sim. Ensure has_offscreen_renderer=True.")


def _resolve_cameras(sim: Any, requested: list[str]) -> list[str]:
    try:
        available = {sim.model.camera(i).name for i in range(sim.model.ncam)}
    except Exception:
                             
        try:
            available = {sim.model.camera_names[i] for i in range(sim.model.ncam)}
        except Exception:
            return requested                            

    resolved = [c for c in requested if c in available]
    if not resolved:
        for fallback in _CAMERA_FALLBACKS:
            if fallback in available:
                print(f"[record_video] Requested cameras not found. Falling back to {fallback!r}.")
                return [fallback]
        return [next(iter(available))]
    missing = set(requested) - set(resolved)
    if missing:
        print(f"[record_video] Cameras not found (skipped): {sorted(missing)}")
    return resolved


def _render_tile(sim: Any, camera_name: str, width: int, height: int) -> np.ndarray:
    frame = sim.render(camera_name=camera_name, width=width, height=height, depth=False)
    return frame[::-1].copy()                              


def _render_frame(sim: Any, cameras: list[str], width: int, height: int) -> np.ndarray:
    tiles = [_render_tile(sim, cam, width, height) for cam in cameras]
    return np.concatenate(tiles, axis=1)               


def main() -> None:
    try:
        import imageio
    except ImportError:
        raise ImportError("imageio is required. pip install 'imageio[ffmpeg]'")

    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})
    algorithm = train_cfg.get("algorithm", "PPO").upper()
    device = train_cfg.get("device", "auto")

    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)

    env_cfg.has_offscreen_renderer = True
    env_cfg.has_renderer = False
    env_cfg.use_camera_obs = False

                                            
    requested_cameras = args.cameras or [env_cfg.render_camera or "robot0_frontview"]

    algo_cls = ALGO_MAP[algorithm]
    model = algo_cls.load(args.checkpoint, device=device)

                                                                                 
    ckpt_path = Path(args.checkpoint)
    obs_config_path = ckpt_path.parent / "obs_config.json"
    if obs_config_path.exists():
        with obs_config_path.open(encoding="utf-8") as f:
            obs_config = json.load(f)
        env_cfg.obs_keys = tuple(obs_config["obs_keys"])
        env_cfg.obs_shapes = {k: tuple(v) for k, v in obs_config["obs_shapes"].items()}
        n_keys = len(env_cfg.obs_keys)
        base_size = sum(int(np.prod(s)) for s in env_cfg.obs_shapes.values())
        print(f"[record_video] obs_config loaded: {n_keys} keys, base_size={base_size} (+4 augmented)")
    else:
                                                                                                     
        print("[record_video] obs_config.json not found — probing env to lock canonical obs shape.")
        _probe = make_env_from_config(env_cfg, seed=args.seed)
        _adapter = _probe._env                                                
        env_cfg.obs_keys = _adapter._obs_keys
        env_cfg.obs_shapes = {k: tuple(v.shape) for k, v in _adapter._dict_space.spaces.items()}
        _probe.close()
        expected = model.observation_space.shape[0]
        probe_base = sum(int(np.prod(s)) for s in env_cfg.obs_shapes.values())
        probe_total = probe_base + 4                                                
        print(f"[record_video] probe obs size={probe_total}, model expects={expected}")
        if probe_total != expected:
            print(
                f"[record_video] WARNING: obs size mismatch ({probe_total} vs {expected}). "
                "Recording may fail. Run a new training run to generate obs_config.json."
            )

    env = make_env_from_config(env_cfg, seed=args.seed)

                                                                  
    sim = _get_sim(env)
    cameras = _resolve_cameras(sim, requested_cameras)
    print(f"Recording cameras: {cameras}  |  resolution per tile: {args.width}×{args.height}")

    frames: list[np.ndarray] = []
    results = []

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False

        while not done:
            frames.append(_render_frame(sim, cameras, args.width, args.height))

            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

        results.append({"episode": ep + 1, "return": round(ep_return, 3), "success": ep_success})
        status = "SUCCESS" if ep_success else "fail"
        print(f"  ep {ep + 1}/{args.episodes}  return={ep_return:.2f}  [{status}]")

    env.close()

    out_path = Path(args.output)
    imageio.mimwrite(str(out_path), frames, fps=args.fps, quality=8)
    total_w = args.width * len(cameras)
    print(f"\nVideo saved → {out_path}  ({len(frames)} frames, {total_w}×{args.height}px)")

    summary = {
        "checkpoint": args.checkpoint,
        "cameras": cameras,
        "resolution": f"{total_w}x{args.height}",
        "episodes": results,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "return_mean": float(np.mean([r["return"] for r in results])),
    }
    json_path = out_path.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
