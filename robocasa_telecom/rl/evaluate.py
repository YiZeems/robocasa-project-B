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


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for evaluation."""

    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoint on RoboCasa")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to .zip checkpoint")
    parser.add_argument("--num-episodes", type=int, default=20, help="Evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Seed for evaluation")
    parser.add_argument("--deterministic", action="store_true", help="Deterministic policy")
    parser.add_argument("--output", default=None, help="Optional output JSON path")
    return parser.parse_args()


def main() -> None:
    """Evaluate a PPO checkpoint and export aggregate metrics."""

    args = parse_args()

    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)
    out_root = ensure_dir(Path(cfg.get("paths", {}).get("output_root", "outputs")) / "eval")

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

        while not done:
            action, _state = model.predict(obs, deterministic=bool(args.deterministic))
            obs, reward, terminated, truncated, info = env.step(action)
            ep_ret += float(reward)
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

        returns.append(ep_ret)
        successes.append(float(ep_success))

    metrics = {
        "task": env_cfg.task,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "num_episodes": int(args.num_episodes),
        "return_mean": float(np.mean(returns)) if returns else 0.0,
        "return_std": float(np.std(returns)) if returns else 0.0,
        "success_rate": float(np.mean(successes)) if successes else 0.0,
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
