from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO on RoboCasa task")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    return parser.parse_args()


def _evaluate_policy(model: PPO, env: Monitor, episodes: int) -> dict[str, float]:
    returns = []
    success_flags = []

    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        ep_success = False

        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += float(reward)
            ep_success = ep_success or infer_success(info, env)
            done = bool(terminated or truncated)

        returns.append(ep_return)
        success_flags.append(float(ep_success))

    return {
        "eval_return_mean": float(np.mean(returns)) if returns else 0.0,
        "eval_return_std": float(np.std(returns)) if returns else 0.0,
        "eval_success_rate": float(np.mean(success_flags)) if success_flags else 0.0,
    }


def _export_training_curve(monitor_path: Path, out_csv: Path) -> None:
    if not monitor_path.exists():
        return

    rows: list[dict[str, float]] = []
    with monitor_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(line for line in f if not line.startswith("#"))
        for idx, row in enumerate(reader):
            rows.append(
                {
                    "episode": float(idx),
                    "reward": float(row.get("r", 0.0)),
                    "length": float(row.get("l", 0.0)),
                    "time": float(row.get("t", 0.0)),
                }
            )

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["episode", "reward", "length", "time"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)

    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    total_timesteps = int(
        args.total_timesteps
        if args.total_timesteps is not None
        else train_cfg.get("total_timesteps", 200000)
    )

    run_id = f"{env_cfg.task}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_root = ensure_dir(paths_cfg.get("output_root", "outputs"))
    checkpoint_root = ensure_dir(paths_cfg.get("checkpoint_root", "checkpoints"))
    tensorboard_root = ensure_dir(paths_cfg.get("tensorboard_root", "logs/tensorboard"))

    run_output_dir = ensure_dir(output_root / run_id)
    run_checkpoint_dir = ensure_dir(checkpoint_root / run_id)

    env = make_env_from_config(env_cfg, seed=seed)
    monitor_path = run_output_dir / "monitor.csv"
    env = Monitor(env, filename=str(monitor_path))

    model = PPO(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        n_steps=int(train_cfg.get("n_steps", 2048)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
        clip_range=float(train_cfg.get("clip_range", 0.2)),
        ent_coef=float(train_cfg.get("ent_coef", 0.0)),
        vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        tensorboard_log=str(tensorboard_root),
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=1,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(train_cfg.get("save_freq_steps", 50000))),
        save_path=str(run_checkpoint_dir),
        name_prefix="ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_cb])
    final_model_path = run_checkpoint_dir / "final_model"
    model.save(str(final_model_path))

    eval_metrics = _evaluate_policy(
        model,
        env,
        episodes=int(train_cfg.get("eval_episodes", 5)),
    )

    _export_training_curve(monitor_path, run_output_dir / "training_curve.csv")

    summary = {
        "run_id": run_id,
        "task": env_cfg.task,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "final_model": f"{final_model_path}.zip",
        **eval_metrics,
    }

    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with (run_output_dir / "resolved_train_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
