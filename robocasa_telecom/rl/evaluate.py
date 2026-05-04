"""Evaluation entrypoint for trained PPO checkpoints on RoboCasa."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success
from ..utils.video import ensure_uint8_frame, grid_2x2, save_mp4


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on RoboCasa")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--num-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic policy")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--video-every",
        type=int,
        default=5,
        help="Save a video every N episodes (set to 1 to save all episodes)",
    )
    parser.add_argument(
        "--video-output-dir",
        default=None,
        help="Optional directory for MP4 exports (default: outputs/eval/videos)",
    )
    parser.add_argument(
        "--video-cameras",
        default="robot0_agentview_center,robot0_eye_in_hand,frontview,sideview",
        help="Comma-separated list of 4 camera names used for the MP4 mosaic",
    )
    parser.add_argument("--video-fps", type=int, default=20, help="FPS for exported MP4 files")
    return parser.parse_args()


def _render_camera_frame(env, camera_name: str) -> np.ndarray:
    """Best-effort camera render across RoboSuite versions."""

    candidates = [
        {"camera_name": camera_name, "width": 256, "height": 256},
        {"camera_name": camera_name},
        {},
    ]
    for kwargs in candidates:
        try:
            frame = env.raw_env.render(mode="rgb_array", **kwargs)
        except TypeError:
            try:
                frame = env.raw_env.render(**kwargs)
            except Exception:
                continue
        except Exception:
            continue
        if frame is not None:
            return ensure_uint8_frame(frame)
    raise RuntimeError(f"Unable to render camera '{camera_name}'")


def _resolve_video_cameras(env_cfg) -> tuple[str, str, str, str]:
    """Pick four cameras with fallbacks across RoboSuite versions."""

    requested = [
        [env_cfg.render_camera, "robot0_agentview_center", "agentview"],
        ["robot0_eye_in_hand", "eye_in_hand", "robot0_wristcam"],
        ["frontview", "agentview"],
        ["sideview", "robot0_agentview_right", "robot0_agentview_left"],
    ]
    resolved: list[str] = []
    for choices in requested:
        for candidate in choices:
            if candidate and candidate not in resolved:
                resolved.append(candidate)
                break
    while len(resolved) < 4:
        resolved.append(env_cfg.render_camera)
    return tuple(resolved[:4])


def main() -> None:
    """Evaluate a PPO checkpoint and export aggregate metrics."""

    args = parse_args()

    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)
    out_root = ensure_dir(Path(cfg.get("paths", {}).get("output_root", "outputs")) / "eval")
    video_root = ensure_dir(
        Path(args.video_output_dir)
        if args.video_output_dir
        else out_root / "videos"
    )
    explicit_cameras = [
        camera.strip()
        for camera in str(args.video_cameras).split(",")
        if camera.strip()
    ]
    video_cameras = tuple(explicit_cameras) if len(explicit_cameras) == 4 else _resolve_video_cameras(env_cfg)

    env = make_env_from_config(env_cfg, seed=args.seed)
    model = PPO.load(args.checkpoint, device=cfg.get("train", {}).get("device", "auto"))

    returns = []
    successes = []

    for ep in range(int(args.num_episodes)):
        # Offset seed by episode index to avoid replaying the exact same reset.
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        ep_ret = 0.0
        ep_success = False
        episode_frames: list[np.ndarray] = []

        while not done:
            action, _state = model.predict(obs, deterministic=bool(args.deterministic))

            if int(args.video_every) > 0 and ep % int(args.video_every) == 0:
                camera_frames = [_render_camera_frame(env, camera) for camera in video_cameras]
                episode_frames.append(grid_2x2(camera_frames))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

        returns.append(ep_ret)
        successes.append(float(ep_success))

        if episode_frames:
            video_path = video_root / f"eval_ep{ep:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            save_mp4(episode_frames, video_path, fps=int(args.video_fps))

    metrics = {
        "task": env_cfg.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_episodes": int(args.num_episodes),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "video_output_dir": str(video_root.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = out_root / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    env.close()
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
