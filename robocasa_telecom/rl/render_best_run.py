"""Render a best-checkpoint rollout as a 4-view MP4 mosaic."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.checkpoints import resolve_run_checkpoint_path
from ..utils.device import resolve_device
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success
from ..utils.video import (
    ensure_uint8_frame,
    grid_2x2,
    resolve_arm_video_cameras,
    save_mp4,
)


SUPPORTED_ALGOS = ("PPO", "SAC")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for best-run rendering."""

    parser = argparse.ArgumentParser(
        description="Render a best RoboCasa checkpoint into a 4-view MP4"
    )
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a checkpoint zip or a run directory containing best_model.zip",
    )
    parser.add_argument(
        "--algorithm",
        choices=list(SUPPORTED_ALGOS),
        default=None,
        help="Override algorithm declared in YAML (PPO or SAC). Defaults to the config.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for the rendered rollout"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output MP4 path (default: outputs/eval/videos/<run>_best_arm_views_<timestamp>.mp4)",
    )
    parser.add_argument(
        "--video-fps", type=int, default=20, help="FPS for the exported MP4 file"
    )
    parser.add_argument(
        "--video-cameras",
        default=None,
        help="Optional comma-separated list of 4 camera names. Defaults to arm-focused views.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Safety cap on rollout length to avoid infinite videos",
    )
    parser.add_argument(
        "--min-seconds",
        type=int,
        default=12,
        help="Minimum video duration in seconds; additional episodes are appended until this target is reached",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=5,
        help="Maximum number of episodes to append when extending the video",
    )
    return parser.parse_args()


def _resolve_algorithm(cfg: dict[str, Any], override: str | None) -> str:
    """Pick algorithm from CLI override, then YAML, defaulting to PPO."""

    raw = override or cfg.get("train", {}).get("algorithm") or "PPO"
    algo = str(raw).upper()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{algo}'. Choose one of: {SUPPORTED_ALGOS}"
        )
    return algo


def _load_model(algorithm: str, checkpoint: str, device: str) -> BaseAlgorithm:
    """Load a checkpoint with the matching SB3 class."""

    if algorithm == "PPO":
        return PPO.load(checkpoint, device=device)
    if algorithm == "SAC":
        return SAC.load(checkpoint, device=device)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _render_camera_frame(env, camera_name: str) -> np.ndarray:
    """Best-effort camera render across RoboSuite versions."""

    sim = getattr(getattr(env, "raw_env", None), "sim", None)
    if sim is not None:
        try:
            frame = sim.render(height=256, width=256, camera_name=camera_name)
        except Exception:
            frame = None
        if frame is not None:
            return ensure_uint8_frame(frame)

    candidates = [({}, "render"), ({"camera_name": camera_name}, "render")]
    for kwargs, _method in candidates:
        try:
            frame = env.raw_env.render(**kwargs)
        except Exception:
            continue
        if frame is not None:
            return ensure_uint8_frame(frame)
    raise RuntimeError(f"Unable to render camera '{camera_name}'")


def _resolve_video_cameras(
    env_cfg, explicit_cameras: list[str] | None = None
) -> tuple[str, str, str, str]:
    """Resolve four cameras, defaulting to arm-focused views."""

    if explicit_cameras and len(explicit_cameras) == 4:
        return tuple(explicit_cameras)
    return resolve_arm_video_cameras(env_cfg)


def render_best_checkpoint_video(
    env_cfg,
    checkpoint: str | Path,
    *,
    algorithm: str,
    device: str = "auto",
    seed: int = 0,
    output: str | Path | None = None,
    video_fps: int = 20,
    video_cameras: list[str] | None = None,
    max_steps: int = 500,
    min_seconds: int = 12,
    max_episodes: int = 5,
) -> dict[str, Any]:
    """Render a best checkpoint as a 4-view MP4 and return metadata."""

    render_env_cfg = replace(env_cfg, has_offscreen_renderer=True, has_renderer=False)
    checkpoint_path = resolve_run_checkpoint_path(checkpoint, preference="best")
    output_path = Path(output) if output is not None else None
    if output_path is None:
        out_root = ensure_dir(Path("outputs") / "eval" / "videos")
        run_name = checkpoint_path.parent.name
        output_path = (
            out_root
            / f"{run_name}_best_arm_views_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    explicit_cameras = [
        camera.strip() for camera in (video_cameras or []) if camera.strip()
    ]
    resolved_cameras = _resolve_video_cameras(
        render_env_cfg, explicit_cameras or None
    )

    model = _load_model(algorithm, str(checkpoint_path), device=device)
    env = make_env_from_config(
        render_env_cfg,
        seed=int(seed),
        reference_obs_space=getattr(model, "observation_space", None),
    )

    obs, _ = env.reset(seed=int(seed))
    frames: list[np.ndarray] = []
    episode_rewards: list[float] = []
    episode_successes: list[bool] = []
    total_reward = 0.0
    min_frames = max(1, int(min_seconds) * int(video_fps))

    for episode_index in range(max(1, int(max_episodes))):
        if episode_index > 0:
            obs, _ = env.reset(seed=int(seed) + episode_index)

        episode_reward = 0.0
        episode_success = False
        step_count = 0
        terminated = False
        truncated = False

        while step_count < int(max_steps):
            camera_frames = [
                _render_camera_frame(env, camera) for camera in resolved_cameras
            ]
            frames.append(grid_2x2(camera_frames))

            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_success = episode_success or infer_success(info, env)
            step_count += 1

            if bool(terminated or truncated):
                break

        total_reward += episode_reward
        episode_rewards.append(float(episode_reward))
        episode_successes.append(bool(episode_success))

        if len(frames) >= min_frames and bool(terminated or truncated):
            break
        if len(frames) >= min_frames and step_count >= int(max_steps):
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames were rendered for the best-run video")

    save_mp4(frames, output_path, fps=int(video_fps))

    metadata = {
        "checkpoint": str(checkpoint_path),
        "algorithm": algorithm,
        "seed": int(seed),
        "video_path": str(output_path.resolve()),
        "video_cameras": list(resolved_cameras),
        "num_frames": len(frames),
        "num_episodes_rendered": len(episode_rewards),
        "min_seconds_requested": int(min_seconds),
        "min_frames_target": int(min_frames),
        "episode_reward": float(total_reward),
        "episode_reward_mean": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "episode_success": bool(any(episode_successes)),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    metadata_path = output_path.with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    metadata["metadata_path"] = str(metadata_path.resolve())
    return metadata


def main() -> None:
    """Render a single best-checkpoint rollout as a 4-view MP4."""

    args = parse_args()

    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get(
        "config_path", "configs/env/open_single_door.yaml"
    )
    env_cfg = load_env_config(env_cfg_path)
    algorithm = _resolve_algorithm(cfg, args.algorithm)

    metadata = render_best_checkpoint_video(
        env_cfg,
        args.checkpoint,
        algorithm=algorithm,
        device=resolve_device(cfg.get("train", {}).get("device", "auto")),
        seed=int(args.seed),
        output=args.output,
        video_fps=int(args.video_fps),
        video_cameras=[
            camera.strip()
            for camera in str(args.video_cameras).split(",")
            if camera.strip()
        ],
        max_steps=int(args.max_steps),
        min_seconds=int(args.min_seconds),
        max_episodes=int(args.max_episodes),
    )

    metadata["config"] = str(Path(args.config).resolve())
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
