#!/usr/bin/env python3
"""Quick environment visualization runner for smoke testing observation/render stack."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Ensure direct script execution can import the project package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robocasa_telecom.env_factory import load_env_config, make_env_from_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for random-policy environment rollout."""

    parser = argparse.ArgumentParser(description="Visualize environment with offscreen multi-cameras")
    parser.add_argument("--config", default="configs/env/open_single_door.yaml")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    """Run random actions while forcing camera observations for debug visibility."""

    args = parse_args()
    env_cfg = load_env_config(args.config)

    # Force camera settings so the script exercises offscreen rendering pipeline.
    env_cfg.use_camera_obs = True
    env_cfg.has_offscreen_renderer = True
    env_cfg.has_renderer = False

    env = make_env_from_config(env_cfg, seed=args.seed)
    obs, _ = env.reset(seed=args.seed)

    print("Observation shape:", np.asarray(obs).shape)
    for step in range(args.steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if step % 20 == 0:
            print(f"step={step} reward={reward:.4f}")

        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)

    env.close()
    print("Visualization run completed.")


if __name__ == "__main__":
    main()
