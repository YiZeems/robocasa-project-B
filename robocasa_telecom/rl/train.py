"""RL training entrypoint for RoboCasa tasks.

Supports PPO, SAC, and A2C via the ``algorithm`` key in the train YAML config.
This module is intentionally self-contained so it can be called from local scripts,
SLURM batch jobs, or direct `python -m` invocations with the same behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from ..envs.factory import load_env_config, make_env_from_config
from ..utils.io import ensure_dir, load_yaml
from ..utils.success import infer_success

# Supported algorithms — extend here to add new ones.
ALGO_MAP = {
    "PPO": PPO,
    "SAC": SAC,
    "A2C": A2C,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training."""

    parser = argparse.ArgumentParser(description="Train RL agent on RoboCasa task")
    parser.add_argument("--config", required=True, help="Path to train YAML config")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=None,
        help="Override total timesteps",
    )
    return parser.parse_args()


def _build_model(
    algorithm: str,
    train_cfg: dict[str, Any],
    env: Monitor,
    tensorboard_root: Path,
    seed: int,
) -> Any:
    """Instantiate the right SB3 model based on ``algorithm``.

    Each algorithm only receives the hyperparameters it understands — passing
    PPO-specific keys (e.g. ``clip_range``) to SAC would raise an error.
    """
    algo_cls = ALGO_MAP[algorithm]
    common = dict(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        tensorboard_log=str(tensorboard_root),
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=1,
    )

    if algorithm in ("PPO", "A2C"):
        common.update(
            n_steps=int(train_cfg.get("n_steps", 2048)),
            gae_lambda=float(train_cfg.get("gae_lambda", 0.95)),
            ent_coef=float(train_cfg.get("ent_coef", 0.0)),
            vf_coef=float(train_cfg.get("vf_coef", 0.5)),
        )
        if algorithm == "PPO":
            common.update(
                batch_size=int(train_cfg.get("batch_size", 256)),
                clip_range=float(train_cfg.get("clip_range", 0.2)),
                n_epochs=int(train_cfg.get("n_epochs", 10)),
            )

    elif algorithm == "SAC":
        common.update(
            buffer_size=int(train_cfg.get("buffer_size", 100000)),
            batch_size=int(train_cfg.get("batch_size", 256)),
            tau=float(train_cfg.get("tau", 0.005)),
            ent_coef=train_cfg.get("ent_coef", "auto"),
            train_freq=int(train_cfg.get("train_freq", 1)),
            gradient_steps=int(train_cfg.get("gradient_steps", 1)),
            learning_starts=int(train_cfg.get("learning_starts", 1000)),
        )

    return algo_cls(**common)


def _evaluate_policy(model: Any, env: Monitor, episodes: int) -> dict[str, float]:
    """Run a quick deterministic evaluation after training completes."""

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
    """Convert SB3 monitor CSV format to a compact plotting-friendly CSV."""

    if not monitor_path.exists():
        return

    rows: list[dict[str, float]] = []
    with monitor_path.open("r", encoding="utf-8") as f:
        # SB3 monitor files include JSON metadata comment lines starting with '#'.
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


def _resolve_run_context(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Resolve config-derived runtime context with CLI overrides."""

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
    algorithm = train_cfg.get("algorithm", "PPO").upper()
    if algorithm not in ALGO_MAP:
        raise ValueError(f"Unknown algorithm '{algorithm}'. Supported: {list(ALGO_MAP)}")

    run_id = f"{env_cfg.task}_{algorithm.lower()}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return {
        "env_cfg": env_cfg,
        "train_cfg": train_cfg,
        "paths_cfg": paths_cfg,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "algorithm": algorithm,
        "run_id": run_id,
    }


def main() -> None:
    """Train PPO model and export reproducible artifacts."""

    args = parse_args()
    cfg = load_yaml(args.config)
    context = _resolve_run_context(cfg, args)

    env_cfg = context["env_cfg"]
    train_cfg = context["train_cfg"]
    paths_cfg = context["paths_cfg"]
    seed = int(context["seed"])
    total_timesteps = int(context["total_timesteps"])
    algorithm = str(context["algorithm"])
    run_id = str(context["run_id"])

    output_root = ensure_dir(paths_cfg.get("output_root", "outputs"))
    checkpoint_root = ensure_dir(paths_cfg.get("checkpoint_root", "checkpoints"))
    tensorboard_root = ensure_dir(paths_cfg.get("tensorboard_root", "logs/tensorboard"))

    run_output_dir = ensure_dir(output_root / run_id)
    run_checkpoint_dir = ensure_dir(checkpoint_root / run_id)

    env = make_env_from_config(env_cfg, seed=seed)
    monitor_path = run_output_dir / "monitor.csv"
    env = Monitor(env, filename=str(monitor_path))

    # Model is instantiated based on the ``algorithm`` key in the YAML config.
    model = _build_model(algorithm, train_cfg, env, tensorboard_root, seed)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, int(train_cfg.get("save_freq_steps", 50000))),
        save_path=str(run_checkpoint_dir),
        name_prefix=algorithm.lower(),
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
        "algorithm": algorithm,
        "task": env_cfg.task,
        "seed": seed,
        "total_timesteps": total_timesteps,
        "final_model": f"{final_model_path}.zip",
        **eval_metrics,
    }

    with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Store resolved config for reproducibility and experiment tracing.
    with (run_output_dir / "resolved_train_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    env.close()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
