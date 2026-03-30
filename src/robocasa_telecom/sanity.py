from __future__ import annotations

import argparse

from .env_factory import load_env_config, make_env_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick sanity check for RoboCasa environment")
    parser.add_argument("--config", required=True, help="Path to env YAML config")
    parser.add_argument("--steps", type=int, default=20, help="Number of random steps")
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_cfg = load_env_config(args.config)
    env = make_env_from_config(env_cfg, seed=args.seed)

    obs, _ = env.reset(seed=args.seed)
    print(f"Initial obs shape: {obs.shape}")

    total_reward = 0.0
    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            obs, _ = env.reset(seed=args.seed)
        if step % 5 == 0:
            print(f"step={step} reward={reward:.4f} cum_reward={total_reward:.4f}")

    env.close()
    print("Sanity check completed successfully.")


if __name__ == "__main__":
    main()
