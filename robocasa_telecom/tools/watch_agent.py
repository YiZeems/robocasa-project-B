"""Interactive live viewer: load a checkpoint and watch the agent in the MuJoCo 3D viewer.

Requires a real display (local machine or SSH with X11 forwarding).
For headless clusters, use record_video.py instead.

Usage:
    # Watch trained agent
    python -m robocasa_telecom.watch_agent \\
        --config configs/train/open_single_door_ppo.yaml \\
        --checkpoint checkpoints/<run_id>/final_model.zip

    # Watch random policy (no checkpoint needed)
    python -m robocasa_telecom.watch_agent \\
        --config configs/train/open_single_door_ppo.yaml \\
        --random

    # Slow down playback for easier observation
    python -m robocasa_telecom.watch_agent \\
        --config configs/train/open_single_door_ppo.yaml \\
        --checkpoint checkpoints/<run_id>/final_model.zip \\
        --sleep 0.05
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.io import load_yaml
from ..utils.success import infer_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch a trained RoboCasa agent in the interactive 3D viewer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to .zip checkpoint. Omit to use random policy.",
    )
    parser.add_argument("--episodes", type=int, default=10, help="Number of episodes to watch")
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.0,
        help="Sleep (seconds) between steps to slow down rendering (e.g. 0.05)",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use random policy instead of a trained checkpoint",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_model(checkpoint: str, algorithm: str, device: str) -> Any:
    from stable_baselines3 import A2C, PPO, SAC

    algo_map = {"PPO": PPO, "SAC": SAC, "A2C": A2C}
    algo_cls = algo_map[algorithm.upper()]
    return algo_cls.load(checkpoint, device=device)


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    train_cfg = cfg.get("train", {})

    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)

    # Enable live 3D viewer
    env_cfg.has_renderer = True
    env_cfg.has_offscreen_renderer = False
    env_cfg.use_camera_obs = False

    model = None
    if not args.random:
        if args.checkpoint is None:
            raise ValueError("Provide --checkpoint or use --random for random policy.")
        model = _load_model(
            args.checkpoint,
            algorithm=train_cfg.get("algorithm", "PPO"),
            device=train_cfg.get("device", "auto"),
        )
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using random policy.")

    env = make_env_from_config(env_cfg, seed=args.seed)
    print(f"Task: {env_cfg.task}  |  Horizon: {env_cfg.horizon} steps")
    print("Press Ctrl+C to quit.\n")

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            done = False
            ep_return = 0.0
            ep_success = False
            step = 0

            while not done:
                if model is not None:
                    action, _ = model.predict(obs, deterministic=True)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                ep_return += float(reward)
                ep_success = ep_success or infer_success(info, env)
                done = bool(terminated or truncated)
                step += 1

                # Render the 3D viewer frame
                env.render()

                if args.sleep > 0:
                    time.sleep(args.sleep)

            status = "SUCCESS" if ep_success else "fail"
            print(f"Episode {ep:3d}/{args.episodes}  steps={step:4d}  return={ep_return:7.2f}  [{status}]")

    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
