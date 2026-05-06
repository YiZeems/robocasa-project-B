"""Evaluation entrypoint for trained PPO/SAC checkpoints on RoboCasa.

The script loads a checkpoint (algorithm auto-detected if not specified),
runs `--num-episodes` rollouts, and exports aggregate metrics. It supports
splitting evaluation seeds into validation / test buckets so the rapport can
report the train/validation/test gap recommended in docs/EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.device import resolve_device
from ..utils.metrics import (
    DOOR_ANGLE_KEYS,
    action_magnitude,
    extract_scalar_metric,
    prefixed_metrics,
)
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success
from ..utils.video import ensure_uint8_frame, grid_2x2, save_mp4


SUPPORTED_ALGOS = ("PPO", "SAC")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate RL checkpoint on RoboCasa")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument(
        "--num-episodes", type=int, default=20, help="Evaluation episodes"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Base seed for evaluation rollouts"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="Deterministic policy"
    )
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    parser.add_argument(
        "--algorithm",
        choices=list(SUPPORTED_ALGOS),
        default=None,
        help="Override algorithm declared in YAML (PPO or SAC). Defaults to the value in the config.",
    )
    parser.add_argument(
        "--split",
        choices=("validation", "test", "custom"),
        default="custom",
        help="Use a predeclared seed bucket from the YAML eval section instead of --seed",
    )
    parser.add_argument(
        "--video-every",
        type=int,
        default=5,
        help="Save a video every N episodes (set to 1 to save all episodes, 0 to disable)",
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
    parser.add_argument(
        "--video-fps", type=int, default=20, help="FPS for exported MP4 files"
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


def _resolve_seed_for_split(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    """Map --split to the seed buckets declared in the YAML eval section."""

    if args.split == "custom":
        return int(args.seed)
    eval_cfg = cfg.get("eval", {})
    key = "validation_seed" if args.split == "validation" else "test_seed"
    seed_value = eval_cfg.get(key)
    if seed_value is None:
        raise ValueError(
            f"Config does not declare 'eval.{key}'. Provide it in the YAML or use --split custom."
        )
    return int(seed_value)


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
    """Evaluate an RL checkpoint and export aggregate metrics."""

    args = parse_args()

    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get(
        "config_path", "configs/env/open_single_door.yaml"
    )
    env_cfg = load_env_config(env_cfg_path)
    out_root = ensure_dir(
        Path(cfg.get("paths", {}).get("output_root", "outputs")) / "eval"
    )
    video_root = ensure_dir(
        Path(args.video_output_dir) if args.video_output_dir else out_root / "videos"
    )
    explicit_cameras = [
        camera.strip()
        for camera in str(args.video_cameras).split(",")
        if camera.strip()
    ]
    video_cameras = (
        tuple(explicit_cameras)
        if len(explicit_cameras) == 4
        else _resolve_video_cameras(env_cfg)
    )

    algorithm = _resolve_algorithm(cfg, args.algorithm)
    seed = _resolve_seed_for_split(args, cfg)

    # Anchor MLflow at an absolute file:// URI (works on Windows, any cwd).
    mlruns_dir = Path(cfg.get("paths", {}).get("mlruns_dir", "mlruns")).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())
    mlflow.set_experiment(f"RoboCasa-{env_cfg.task}-Eval")
    eval_run_name = (
        f"eval_{algorithm}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    mlflow.start_run(run_name=eval_run_name)

    # Log evaluation parameters
    mlflow.log_param("algorithm", algorithm)
    mlflow.log_param("split", args.split)
    mlflow.log_param("seed", seed)
    mlflow.log_param("num_episodes", args.num_episodes)
    mlflow.log_param("checkpoint", str(Path(args.checkpoint).resolve()))
    mlflow.log_param("deterministic", args.deterministic)

    device = resolve_device(cfg.get("train", {}).get("device", "auto"))
    mlflow.log_param("resolved_device", device)
    model = _load_model(algorithm, args.checkpoint, device=device)
    env = make_env_from_config(
        env_cfg,
        seed=seed,
        reference_obs_space=getattr(model, "observation_space", None),
    )

    metrics = {
        "task": env_cfg.task,
        "algorithm": algorithm,
        "split": args.split,
        "seed": int(seed),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "video_output_dir": str(video_root.resolve()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    returns: list[float] = []
    successes: list[float] = []
    episode_lengths: list[float] = []
    action_magnitudes: list[float] = []
    door_angles: list[float] = []

    for ep in range(int(args.num_episodes)):
        # Offset seed by episode index to avoid replaying the exact same reset.
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        ep_len = 0
        ep_success = False
        episode_frames: list[np.ndarray] = []
        ep_action_magnitudes: list[float] = []
        ep_door_angle: float | None = None

        while not done:
            action, _state = model.predict(obs, deterministic=bool(args.deterministic))
            ep_action_magnitudes.append(action_magnitude(action))

            if int(args.video_every) > 0 and ep % int(args.video_every) == 0:
                camera_frames = [
                    _render_camera_frame(env, camera) for camera in video_cameras
                ]
                episode_frames.append(grid_2x2(camera_frames))

            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_len += 1
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

            door_angle = extract_scalar_metric(info if isinstance(info, dict) else None, env, DOOR_ANGLE_KEYS)
            if door_angle is not None:
                ep_door_angle = door_angle

        returns.append(ep_ret)
        successes.append(float(ep_success))
        episode_lengths.append(float(ep_len))
        action_magnitudes.append(
            float(np.mean(ep_action_magnitudes)) if ep_action_magnitudes else 0.0
        )
        if ep_door_angle is not None:
            door_angles.append(ep_door_angle)

        if episode_frames:
            video_path = (
                video_root
                / f"eval_ep{ep:03d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            save_mp4(episode_frames, video_path, fps=int(args.video_fps))

    metrics.update(
        {
            "num_episodes": int(args.num_episodes),
            "return_mean": float(np.mean(returns)) if returns else 0.0,
            "return_std": float(np.std(returns)) if returns else 0.0,
            "success_rate": float(np.mean(successes)) if successes else 0.0,
            "episode_length_mean": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
            "episode_length_std": float(np.std(episode_lengths)) if episode_lengths else 0.0,
            "action_magnitude_mean": float(np.mean(action_magnitudes)) if action_magnitudes else 0.0,
            "action_magnitude_std": float(np.std(action_magnitudes)) if action_magnitudes else 0.0,
        }
    )
    if door_angles:
        metrics["door_angle_final_mean"] = float(np.mean(door_angles))
        metrics["door_angle_final_std"] = float(np.std(door_angles))

    base_metrics = dict(metrics)
    split_alias = "validation" if args.split == "validation" else "test" if args.split == "test" else "evaluation"
    metrics[f"{split_alias}_return_mean"] = metrics["return_mean"]
    metrics[f"{split_alias}_return_std"] = metrics["return_std"]
    metrics[f"{split_alias}_success_rate"] = metrics["success_rate"]
    metrics[f"{split_alias}_episode_length_mean"] = metrics["episode_length_mean"]
    metrics[f"{split_alias}_episode_length_std"] = metrics["episode_length_std"]
    metrics[f"{split_alias}_action_magnitude_mean"] = metrics["action_magnitude_mean"]
    metrics[f"{split_alias}_action_magnitude_std"] = metrics["action_magnitude_std"]
    if "door_angle_final_mean" in metrics:
        metrics[f"{split_alias}_door_angle_final_mean"] = metrics["door_angle_final_mean"]
        metrics[f"{split_alias}_door_angle_final_std"] = metrics["door_angle_final_std"]

    # Log metrics to MLflow
    mlflow.log_metrics(prefixed_metrics(base_metrics, "eval_"))
    mlflow.log_metrics(prefixed_metrics(metrics, f"{split_alias}_"))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = (
            out_root
            / f"eval_{algorithm}_{args.split}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Log evaluation results as artifact
    mlflow.log_artifact(str(output_path))

    # End MLflow run
    mlflow.end_run()

    env.close()
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics to: {output_path}")


if __name__ == "__main__":
    main()
