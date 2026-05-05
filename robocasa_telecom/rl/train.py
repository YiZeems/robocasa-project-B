"""Algorithm-agnostic training entrypoint for RoboCasa tasks.

Supports both PPO (on-policy baseline) and SAC (off-policy main method) via the
`train.algorithm` field in the YAML config. The pipeline also wires periodic
validation rollouts, best-checkpoint saving keyed on validation success rate,
and an early-stopping signal so the experiment plan in docs/EXPERIMENTS.md is
reproducible end-to-end.
"""

from __future__ import annotations

import argparse
import csv
import functools
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import yaml
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from ..utils.checkpoints import (
    CheckpointArtifact,
    find_latest_resume_candidate,
    load_checkpoint_metadata,
    resolve_checkpoint_artifact,
    save_checkpoint_metadata,
)
from ..envs.factory import EnvConfig, load_env_config, make_env_from_config
from ..utils.metrics import prefixed_metrics, summarize_rollout_episodes
from ..utils.io import ensure_dir, load_yaml
from .render_best_run import render_best_checkpoint_video


SUPPORTED_ALGOS = {"PPO", "SAC"}
BEST_RUN_VIDEO_ENV = "ROBOCASA_RENDER_BEST_RUN_VIDEO"
BEST_RUN_VIDEO_MAX_STEPS_ENV = "ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_STEPS"
BEST_RUN_VIDEO_FPS_ENV = "ROBOCASA_RENDER_BEST_RUN_VIDEO_FPS"
BEST_RUN_VIDEO_MIN_SECONDS_ENV = "ROBOCASA_RENDER_BEST_RUN_VIDEO_MIN_SECONDS"
BEST_RUN_VIDEO_MAX_EPISODES_ENV = "ROBOCASA_RENDER_BEST_RUN_VIDEO_MAX_EPISODES"


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
    parser.add_argument(
        "--algorithm",
        choices=sorted(SUPPORTED_ALGOS),
        default=None,
        help="Override algorithm declared in YAML (PPO or SAC)",
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help="Resume training from a checkpoint zip, checkpoint directory, or checkpoint metadata JSON.",
    )
    parser.add_argument(
        "--auto-resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Automatically resume from the latest incomplete run matching the same task/algo/seed.",
    )
    return parser.parse_args()


@dataclass
class RunContext:
    """Resolved runtime context combining YAML config and CLI overrides."""

    env_cfg: EnvConfig
    train_cfg: dict[str, Any]
    paths_cfg: dict[str, Any]
    eval_cfg: dict[str, Any]
    algorithm: str
    seed: int
    total_timesteps: int
    run_id: str
    resume_from: str | None
    resume_artifact: CheckpointArtifact | None
    auto_resume: bool


def _resolve_run_context(cfg: dict[str, Any], args: argparse.Namespace) -> RunContext:
    """Resolve config-derived runtime context with CLI overrides."""

    env_cfg_path = cfg.get("env", {}).get(
        "config_path", "configs/env/open_single_door.yaml"
    )
    env_cfg = load_env_config(env_cfg_path)

    train_cfg = cfg.get("train", {})
    paths_cfg = cfg.get("paths", {})
    eval_cfg = cfg.get("eval", {})

    algorithm = args.algorithm or train_cfg.get("algorithm", "PPO")
    algorithm = str(algorithm).upper()
    if algorithm not in SUPPORTED_ALGOS:
        raise ValueError(
            f"Unsupported algorithm '{algorithm}'. Choose one of: {sorted(SUPPORTED_ALGOS)}"
        )

    seed = int(args.seed if args.seed is not None else train_cfg.get("seed", 0))
    total_timesteps = int(
        args.total_timesteps
        if args.total_timesteps is not None
        else train_cfg.get("total_timesteps", 200_000)
    )

    resume_artifact: CheckpointArtifact | None = None
    resume_from = None
    if args.resume_from:
        resume_from = str(Path(args.resume_from).expanduser())
        resume_artifact = resolve_checkpoint_artifact(resume_from)
        run_id = resume_artifact.model_path.parent.name
    else:
        if bool(getattr(args, "auto_resume", True)):
            run_prefix = f"{env_cfg.task}_{algorithm}_seed{seed}_"
            candidate = find_latest_resume_candidate(
                paths_cfg.get("checkpoint_root", "checkpoints"),
                run_prefix,
            )
            if candidate is not None:
                resume_artifact = candidate.artifact
                resume_from = str(candidate.run_dir)
                run_id = candidate.run_dir.name
            else:
                run_id = f"{env_cfg.task}_{algorithm}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            run_id = f"{env_cfg.task}_{algorithm}_seed{seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    return RunContext(
        env_cfg=env_cfg,
        train_cfg=train_cfg,
        paths_cfg=paths_cfg,
        eval_cfg=eval_cfg,
        algorithm=algorithm,
        seed=seed,
        total_timesteps=total_timesteps,
        run_id=run_id,
        resume_from=resume_from,
        resume_artifact=resume_artifact,
        auto_resume=bool(getattr(args, "auto_resume", True)),
    )


def _resolve_policy_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build policy_kwargs dict from optional `pi`/`qf` size hints in YAML."""

    policy_kwargs = dict(train_cfg.get("policy_kwargs") or {})
    net_arch = train_cfg.get("net_arch")
    if net_arch is not None and "net_arch" not in policy_kwargs:
        policy_kwargs["net_arch"] = list(net_arch)
    return policy_kwargs


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse a boolean environment variable."""

    raw = os.getenv(name)
    if raw is None:
        return default

    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off", ""}:
        return False
    raise ValueError(
        f"Environment variable {name} must be 0/1/true/false/yes/no/on/off, got {raw!r}"
    )


def _env_int(name: str, default: int) -> int:
    """Parse an integer environment variable."""

    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return int(default)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Environment variable {name} must be an integer, got {raw!r}") from exc


class MLflowMetricsCallback(BaseCallback):
    """Forward SB3 rollout/train logger entries to the active MLflow run."""

    def _on_step(self) -> bool:
        if self.logger is None:
            return True
        for key, value in self.logger.name_to_value.items():
            if isinstance(value, (int, float)):
                try:
                    mlflow.log_metric(key, float(value), step=self.num_timesteps)
                except Exception:
                    pass
        return True


class RewardHackMonitorCallback(BaseCallback):
    """Accumulate per-episode reward component stats and log to MLflow.

    Logged metrics (reward_hack/* prefix):
      approach_frac    — fraction of reward from approach; >0.5 after 100k = hover hacking
      progress_frac    — fraction from progress; should grow over training
      success_frac     — fraction from success bonus
      oscillation_frac — fraction from oscillation penalty
      stagnation_rate  — fraction of episodes that fired the stagnation penalty
      theta_best_mean  — mean best door angle per episode (tracks door opening progress)
      stagnation_steps_mean  — mean steps stuck near handle per episode
      oscillation_steps_mean — mean steps oscillating per episode
      sign_changes_mean      — mean door angle sign changes per episode
      reward_without_success — mean episode return for failed episodes (reward hacking proxy)
    """

    def __init__(self, log_freq: int = 25_000):
        super().__init__()
        self._log_freq = log_freq
        self._last_log_t = 0
        self._reset_buffers()

    def _reset_buffers(self) -> None:
        self._ep_approach:    list[float] = []
        self._ep_progress:    list[float] = []
        self._ep_success_r:   list[float] = []
        self._ep_oscillation: list[float] = []
        self._ep_total:       list[float] = []
        self._ep_stagnation_fired: list[float] = []
        self._ep_theta_best:  list[float] = []
        self._ep_stagnation_steps:  list[float] = []
        self._ep_oscillation_steps: list[float] = []
        self._ep_sign_changes: list[float] = []
        self._ep_success_flag: list[float] = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))

        for info, done in zip(infos, dones):
            if not done:
                continue
            comp = info.get("episode_reward_components")
            if comp is None:
                continue

            total = comp.get("total", 0.0)
            self._ep_approach.append(comp.get("approach", 0.0))
            self._ep_progress.append(comp.get("progress", 0.0))
            self._ep_success_r.append(comp.get("success", 0.0))
            self._ep_oscillation.append(comp.get("oscillation", 0.0))
            self._ep_total.append(total)
            self._ep_stagnation_fired.append(
                1.0 if comp.get("stagnation_steps", 0.0) > 0 else 0.0
            )
            self._ep_theta_best.append(comp.get("theta_best", 0.0))
            self._ep_stagnation_steps.append(comp.get("stagnation_steps", 0.0))
            self._ep_oscillation_steps.append(comp.get("oscillation_steps", 0.0))
            # sign_changes not in episode_summary, use step-level if available
            rc = info.get("reward_components", {})
            self._ep_sign_changes.append(rc.get("sign_changes", 0.0))
            self._ep_success_flag.append(1.0 if comp.get("success", 0.0) > 0 else 0.0)

        if self.num_timesteps - self._last_log_t >= self._log_freq and self._ep_total:
            self._last_log_t = self.num_timesteps
            n = len(self._ep_total)
            total_mean = float(np.mean(self._ep_total))
            denom = max(abs(total_mean), 1e-8)

            failed_returns = [
                t for t, s in zip(self._ep_total, self._ep_success_flag) if s == 0.0
            ]

            try:
                mlflow.log_metrics({
                    # Reward fraction breakdown (hover-hack / oscillation detection)
                    "reward_hack/approach_frac":    float(np.mean(self._ep_approach))    / denom,
                    "reward_hack/progress_frac":    float(np.mean(self._ep_progress))    / denom,
                    "reward_hack/success_frac":     float(np.mean(self._ep_success_r))   / denom,
                    "reward_hack/oscillation_frac": float(np.mean(self._ep_oscillation)) / denom,
                    # Episode-level diagnostics
                    "reward_hack/stagnation_rate":       float(np.mean(self._ep_stagnation_fired)),
                    "reward_hack/theta_best_mean":       float(np.mean(self._ep_theta_best)),
                    "reward_hack/stagnation_steps_mean": float(np.mean(self._ep_stagnation_steps)),
                    "reward_hack/oscillation_steps_mean":float(np.mean(self._ep_oscillation_steps)),
                    "reward_hack/sign_changes_mean":     float(np.mean(self._ep_sign_changes)),
                    "reward_hack/reward_without_success":float(np.mean(failed_returns)) if failed_returns else float(np.mean(self._ep_total)),
                    "reward_hack/episodes":              float(n),
                }, step=self.num_timesteps)
            except Exception:
                pass

            self._reset_buffers()

        return True


def _build_ppo(
    env: Monitor, train_cfg: dict[str, Any], seed: int
) -> PPO:
    """Instantiate PPO with hyperparameters from YAML."""

    policy_kwargs = _resolve_policy_kwargs(train_cfg)
    return PPO(
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
        max_grad_norm=float(train_cfg.get("max_grad_norm", 0.5)),
        n_epochs=int(train_cfg.get("n_epochs", 10)),
        tensorboard_log=None,
        policy_kwargs=policy_kwargs or None,
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=0,
    )


def _build_sac(
    env: Monitor, train_cfg: dict[str, Any], seed: int
) -> SAC:
    """Instantiate SAC with hyperparameters from YAML.

    `ent_coef` accepts numeric values, the literal string "auto", or strings of
    form "auto_<initial>" so the YAML can stay strict-typed.
    """

    policy_kwargs = _resolve_policy_kwargs(train_cfg)

    raw_ent = train_cfg.get("ent_coef", "auto")
    if isinstance(raw_ent, str) and raw_ent.lower().startswith("auto"):
        ent_coef: Any = raw_ent
    else:
        ent_coef = float(raw_ent)

    return SAC(
        policy=train_cfg.get("policy", "MlpPolicy"),
        env=env,
        learning_rate=float(train_cfg.get("learning_rate", 3e-4)),
        buffer_size=int(train_cfg.get("buffer_size", 1_000_000)),
        learning_starts=int(train_cfg.get("learning_starts", 10_000)),
        batch_size=int(train_cfg.get("batch_size", 256)),
        tau=float(train_cfg.get("tau", 0.005)),
        gamma=float(train_cfg.get("gamma", 0.99)),
        train_freq=int(train_cfg.get("train_freq", 1)),
        gradient_steps=int(train_cfg.get("gradient_steps", 1)),
        ent_coef=ent_coef,
        target_update_interval=int(train_cfg.get("target_update_interval", 1)),
        target_entropy=train_cfg.get("target_entropy", "auto"),
        tensorboard_log=None,
        policy_kwargs=policy_kwargs or None,
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=0,
    )


def _build_model(
    algorithm: str,
    env: Monitor,
    train_cfg: dict[str, Any],
    seed: int,
) -> BaseAlgorithm:
    """Dispatch model construction by algorithm name."""

    if algorithm == "PPO":
        return _build_ppo(env, train_cfg, seed)
    if algorithm == "SAC":
        return _build_sac(env, train_cfg, seed)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _evaluate_policy(
    model: BaseAlgorithm,
    env: Any,
    episodes: int,
    seed: int = 0,
    deterministic: bool = True,
) -> dict[str, float]:
    """Run a deterministic evaluation and summarize rollout metrics."""

    return summarize_rollout_episodes(
        model=model,
        env=env,
        episodes=episodes,
        seed=seed,
        deterministic=deterministic,
    )


class ValidationCallback(BaseCallback):
    """Periodic validation, best-checkpoint saving, and early stopping.

    The callback runs `n_eval_episodes` rollouts on a separate validation env
    every `eval_freq` env steps. The best model (highest validation success
    rate, ties broken by mean return) is persisted to `best_model.zip`. A
    history CSV is written so the rapport can plot the validation curve.
    Training stops once `patience` consecutive evaluations fail to improve
    the best success rate.
    """

    def __init__(
        self,
        eval_env: Any,
        eval_freq: int,
        n_eval_episodes: int,
        log_path: Path,
        best_model_path: Path,
        validation_seed: int,
        patience: int = 0,
        deterministic: bool = True,
    ):
        super().__init__(verbose=1)
        self._eval_env = eval_env
        self._eval_freq = max(1, int(eval_freq))
        self._n_eval_episodes = max(1, int(n_eval_episodes))
        self._log_path = Path(log_path)
        self._best_model_path = Path(best_model_path)
        self._validation_seed = int(validation_seed)
        self._patience = int(patience)
        self._deterministic = bool(deterministic)

        self._best_success: float = -1.0
        self._best_return_mean: float = -np.inf
        self._best_episode_length_mean: float = -np.inf
        self._best_action_magnitude_mean: float = -np.inf
        self._best_door_angle_final_mean: float = -np.inf
        self._best_step: int = 0
        self._no_improve_evals: int = 0
        self._history: list[dict[str, float]] = []
        self._last_eval_timestep: int = -1

        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._best_model_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def best_success(self) -> float:
        return float(max(self._best_success, 0.0))

    @property
    def best_return_mean(self) -> float:
        return float(
            self._best_return_mean if np.isfinite(self._best_return_mean) else 0.0
        )

    @property
    def best_episode_length_mean(self) -> float:
        return float(
            self._best_episode_length_mean
            if np.isfinite(self._best_episode_length_mean)
            else 0.0
        )

    @property
    def best_action_magnitude_mean(self) -> float:
        return float(
            self._best_action_magnitude_mean
            if np.isfinite(self._best_action_magnitude_mean)
            else 0.0
        )

    @property
    def best_door_angle_final_mean(self) -> float:
        return float(
            self._best_door_angle_final_mean
            if np.isfinite(self._best_door_angle_final_mean)
            else np.nan
        )

    @property
    def best_step(self) -> int:
        return int(self._best_step)

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_eval_timestep < self._eval_freq:
            return True
        self._last_eval_timestep = self.num_timesteps

        metrics = _evaluate_policy(
            self.model,
            self._eval_env,
            episodes=self._n_eval_episodes,
            seed=self._validation_seed,
            deterministic=self._deterministic,
        )
        success = float(metrics["success_rate"])
        return_mean = float(metrics["return_mean"])
        episode_length_mean = float(metrics["episode_length_mean"])
        action_magnitude_mean = float(metrics["action_magnitude_mean"])
        door_angle_final_mean = float(metrics.get("door_angle_final_mean", np.nan))

        row: dict[str, float] = {"step": float(self.num_timesteps)}
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)):
                row[f"val_{k}"] = float(v)
        self._history.append(row)

        # Log all metrics to MLflow (core + anti-hacking)
        mlflow_metrics: dict[str, float] = {}
        for k, v in metrics.items():
            if v is not None and isinstance(v, (int, float)) and np.isfinite(float(v)):
                mlflow_metrics[f"validation_{k}"] = float(v)
        try:
            mlflow.log_metrics(mlflow_metrics, step=self.num_timesteps)
        except Exception:
            pass

        write_header = not self._log_path.exists()
        with self._log_path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        improved = success > self._best_success or (
            success == self._best_success and return_mean > self._best_return_mean
        )
        if improved:
            self._best_success = success
            self._best_return_mean = return_mean
            self._best_episode_length_mean = episode_length_mean
            self._best_action_magnitude_mean = action_magnitude_mean
            if "door_angle_final_mean" in metrics:
                self._best_door_angle_final_mean = door_angle_final_mean
            self._best_step = int(self.num_timesteps)
            self._no_improve_evals = 0
            self.model.save(str(self._best_model_path))
            if self.verbose:
                print(
                    f"[ValidationCallback] new best at step={self._best_step} "
                    f"success={success:.3f} return_mean={return_mean:.3f}"
                )
        else:
            self._no_improve_evals += 1

        if self._patience > 0 and self._no_improve_evals >= self._patience:
            if self.verbose:
                print(
                    f"[ValidationCallback] early stop: no improvement for "
                    f"{self._no_improve_evals} evals (patience={self._patience})"
                )
            return False

        return True


class PeriodicCheckpointCallback(BaseCallback):
    """Save model checkpoints plus optional replay-buffer sidecars."""

    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        name_prefix: str,
        algorithm: str,
        run_id: str,
        save_replay_buffer: bool = False,
    ):
        super().__init__(verbose=1)
        self._save_freq = max(1, int(save_freq))
        self._save_path = Path(save_path)
        self._name_prefix = str(name_prefix)
        self._algorithm = str(algorithm)
        self._run_id = str(run_id)
        self._save_replay_buffer = bool(save_replay_buffer)
        self._save_path.mkdir(parents=True, exist_ok=True)
        self._last_checkpoint_timestep: int = 0

    def _save_checkpoint(self, suffix: str) -> None:
        model_path = self._save_path / f"{self._name_prefix}_{suffix}_steps.zip"
        self.model.save(str(model_path))

        if self._save_replay_buffer and hasattr(self.model, "save_replay_buffer"):
            replay_buffer_path = model_path.with_name(
                f"{model_path.stem}_replay_buffer.pkl"
            )
            try:
                self.model.save_replay_buffer(str(replay_buffer_path))
            except Exception as exc:  # pragma: no cover - best-effort persistence.
                if self.verbose:
                    print(f"[PeriodicCheckpointCallback] replay buffer save failed: {exc}")

        save_checkpoint_metadata(
            model_path,
            run_id=self._run_id,
            algorithm=self._algorithm,
            num_timesteps=int(self.num_timesteps),
            source="periodic",
            save_replay_buffer=self._save_replay_buffer,
        )

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_checkpoint_timestep < self._save_freq:
            return True
        self._last_checkpoint_timestep = self.num_timesteps

        self._save_checkpoint(str(self.num_timesteps))
        if self.verbose:
            print(
                f"[PeriodicCheckpointCallback] saved checkpoint at step={self.num_timesteps}"
            )
        return True


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


def _load_resume_summary(run_output_dir: Path) -> dict[str, Any]:
    """Load previous train summary if a resumed run already exists."""

    summary_path = run_output_dir / "train_summary.json"
    if not summary_path.exists():
        return {}

    try:
        with summary_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _load_resume_validation_snapshot(run_output_dir: Path) -> dict[str, Any]:
    """Recover the best validation metrics from a previous run directory."""

    snapshot = _load_resume_summary(run_output_dir)
    needed_keys = {
        "best_validation_success_rate",
        "best_validation_return_mean",
        "best_validation_episode_length_mean",
        "best_validation_action_magnitude_mean",
        "best_validation_step",
    }
    if needed_keys.issubset(snapshot):
        return snapshot

    curve_path = run_output_dir / "validation_curve.csv"
    if not curve_path.exists():
        return snapshot

    best_row: dict[str, float] | None = None
    with curve_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                success = float(row.get("val_success_rate", 0.0))
                return_mean = float(row.get("val_return_mean", 0.0))
            except (TypeError, ValueError):
                continue

            if best_row is None:
                best_row = row
                continue

            best_success = float(best_row.get("val_success_rate", 0.0))
            best_return = float(best_row.get("val_return_mean", 0.0))
            if success > best_success or (
                success == best_success and return_mean > best_return
            ):
                best_row = row

    if best_row is None:
        return snapshot

    snapshot.setdefault(
        "best_validation_success_rate", float(best_row.get("val_success_rate", 0.0))
    )
    snapshot.setdefault(
        "best_validation_return_mean", float(best_row.get("val_return_mean", 0.0))
    )
    snapshot.setdefault(
        "best_validation_step", int(float(best_row.get("step", 0.0)))
    )
    snapshot.setdefault(
        "best_validation_episode_length_mean",
        float(best_row.get("val_episode_length_mean", 0.0)),
    )
    snapshot.setdefault(
        "best_validation_action_magnitude_mean",
        float(best_row.get("val_action_magnitude_mean", 0.0)),
    )
    if "val_door_angle_final_mean" in best_row:
        try:
            snapshot.setdefault(
                "best_validation_door_angle_final_mean",
                float(best_row.get("val_door_angle_final_mean", 0.0)),
            )
        except (TypeError, ValueError):
            pass
    return snapshot


def _seed_validation_callback_from_snapshot(
    callback: ValidationCallback, snapshot: dict[str, Any]
) -> None:
    """Seed validation callback best-state from a prior run snapshot."""

    if not snapshot:
        return

    try:
        callback._best_success = float(snapshot.get("best_validation_success_rate", -1.0))
        callback._best_return_mean = float(
            snapshot.get("best_validation_return_mean", -np.inf)
        )
        callback._best_step = int(snapshot.get("best_validation_step", 0))
        callback._best_episode_length_mean = float(
            snapshot.get("best_validation_episode_length_mean", -np.inf)
        )
        callback._best_action_magnitude_mean = float(
            snapshot.get("best_validation_action_magnitude_mean", -np.inf)
        )
        if "best_validation_door_angle_final_mean" in snapshot:
            callback._best_door_angle_final_mean = float(
                snapshot["best_validation_door_angle_final_mean"]
            )
    except (TypeError, ValueError):
        pass


def _set_model_num_timesteps(model: BaseAlgorithm, num_timesteps: int) -> None:
    """Best-effort setter for SB3's internal step counter."""

    for attr in ("num_timesteps", "_num_timesteps"):
        try:
            setattr(model, attr, int(num_timesteps))
            return
        except Exception:
            continue


def _build_model_for_resume(
    ctx: RunContext,
    train_env: Monitor,
    resume_artifact: CheckpointArtifact | None,
) -> BaseAlgorithm:
    """Create or resume a model, restoring replay buffer when available."""

    device = ctx.train_cfg.get("device", "auto")
    if resume_artifact is None:
        return _build_model(ctx.algorithm, train_env, ctx.train_cfg, ctx.seed)

    load_kwargs = {
        "env": train_env,
        "device": device,
        "tensorboard_log": None,
        "print_system_info": False,
    }
    try:
        if ctx.algorithm == "PPO":
            model = PPO.load(str(resume_artifact.model_path), **load_kwargs)
        elif ctx.algorithm == "SAC":
            model = SAC.load(str(resume_artifact.model_path), **load_kwargs)
        else:  # pragma: no cover - guarded by config validation.
            raise ValueError(f"Unsupported algorithm: {ctx.algorithm}")
    except ValueError as exc:
        print(
            f"[train] Checkpoint incompatible ({exc}); starting fresh model instead."
        )
        return _build_model(ctx.algorithm, train_env, ctx.train_cfg, ctx.seed)

    if ctx.algorithm == "SAC" and resume_artifact.replay_buffer_path.exists():
        try:
            fresh_rb = model.replay_buffer  # already sized for current n_envs
            model.load_replay_buffer(str(resume_artifact.replay_buffer_path))
            if model.replay_buffer.n_envs != model.n_envs:
                print(
                    f"[train] Replay buffer n_envs={model.replay_buffer.n_envs} != "
                    f"env n_envs={model.n_envs}; discarding loaded buffer"
                )
                model.replay_buffer = fresh_rb
        except Exception as exc:  # pragma: no cover - best-effort resume.
            print(
                f"[train] Warning: failed to restore replay buffer from "
                f"{resume_artifact.replay_buffer_path}: {exc}"
            )

    return model


def _worker_env_init(
    env_cfg: EnvConfig,
    seed: int,
    reference_obs_space: Any,
) -> Any:
    """Module-level factory for SubprocVecEnv workers (must be picklable for spawn).

    Defined at module level so multiprocessing.spawn can pickle it by reference.
    Each worker imports robosuite/robocasa independently in its fresh subprocess.
    """
    return make_env_from_config(env_cfg, seed=seed, reference_obs_space=reference_obs_space)


def _make_vec_env(
    env_cfg: EnvConfig,
    n_envs: int,
    base_seed: int,
    monitor_dir: Path,
    reference_obs_space: Any = None,
) -> Any:
    """Build a parallel VecEnv with n_envs workers.

    Uses SubprocVecEnv (spawn) for n_envs > 1 so each worker runs in its own
    clean process — safe on macOS/Apple Silicon with MPS.
    Each worker gets a unique seed so scenes and object layouts differ.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv
    monitor_dir.mkdir(parents=True, exist_ok=True)

    factories = [
        functools.partial(
            _worker_env_init,
            env_cfg,
            base_seed + i,
            reference_obs_space,
        )
        for i in range(n_envs)
    ]

    if n_envs == 1:
        return DummyVecEnv(factories)

    return SubprocVecEnv(factories, start_method="spawn")


def main() -> None:
    """Train an RL model and export reproducible artifacts."""

    args = parse_args()
    cfg = load_yaml(args.config)
    ctx = _resolve_run_context(cfg, args)
    best_run_video_enabled = _env_flag(BEST_RUN_VIDEO_ENV, False)
    best_run_video_fps = _env_int(BEST_RUN_VIDEO_FPS_ENV, 20)
    best_run_video_max_steps = _env_int(BEST_RUN_VIDEO_MAX_STEPS_ENV, 500)
    best_run_video_min_seconds = _env_int(BEST_RUN_VIDEO_MIN_SECONDS_ENV, 12)
    best_run_video_max_episodes = _env_int(BEST_RUN_VIDEO_MAX_EPISODES_ENV, 5)

    output_root = ensure_dir(ctx.paths_cfg.get("output_root", "outputs"))
    checkpoint_root = ensure_dir(ctx.paths_cfg.get("checkpoint_root", "checkpoints"))

    run_output_dir = ensure_dir(output_root / ctx.run_id)
    run_checkpoint_dir = ensure_dir(checkpoint_root / ctx.run_id)

    n_envs = int(ctx.train_cfg.get("n_envs", 1))

    # Build a single env first to fix the obs space, then use it as reference
    # for all parallel workers so every worker has the same obs shape.
    _ref_env = make_env_from_config(ctx.env_cfg, seed=ctx.seed)
    reference_obs_space = _ref_env.observation_space
    _ref_env.close()

    train_env = _make_vec_env(
        ctx.env_cfg,
        n_envs=n_envs,
        base_seed=ctx.seed,
        monitor_dir=run_output_dir / "monitors",
        reference_obs_space=reference_obs_space,
    )
    monitor_path = run_output_dir / "monitor.csv"
    train_env = VecMonitor(train_env, filename=str(monitor_path))
    model: BaseAlgorithm | None = None
    val_callback: ValidationCallback | None = None
    summary: dict[str, Any] = {}
    final_model_path: Path | None = None
    final_step: int = 0
    resume_snapshot = _load_resume_validation_snapshot(run_output_dir)
    try:
        # Initialize MLflow experiment
        mlflow.set_experiment(f"RoboCasa-{ctx.env_cfg.task}")
        mlflow.start_run(run_name=ctx.run_id)

        # Log training parameters to MLflow
        mlflow.log_param("algorithm", ctx.algorithm)
        mlflow.log_param("seed", ctx.seed)
        mlflow.log_param("total_timesteps", ctx.total_timesteps)
        mlflow.log_param("task", ctx.env_cfg.task)
        mlflow.log_param("auto_resume", ctx.auto_resume)
        mlflow.log_param("resume_mode", ctx.resume_artifact is not None)
        mlflow.log_param("resume_from", ctx.resume_from or "")
        mlflow.log_param("eval_freq", int(ctx.eval_cfg.get("eval_freq", 0)))
        mlflow.log_param("n_eval_episodes", int(ctx.eval_cfg.get("n_eval_episodes", 0)))
        mlflow.log_param("best_run_video_enabled", best_run_video_enabled)
        mlflow.log_param("best_run_video_fps", best_run_video_fps)
        mlflow.log_param("best_run_video_max_steps", best_run_video_max_steps)
        mlflow.log_param("best_run_video_min_seconds", best_run_video_min_seconds)
        mlflow.log_param("best_run_video_max_episodes", best_run_video_max_episodes)
        mlflow.log_param(
            "early_stopping_patience",
            int(ctx.eval_cfg.get("early_stopping_patience", 0)),
        )

        # Log hyperparameters from train config
        for key, value in ctx.train_cfg.items():
            if key not in ["policy_kwargs", "net_arch"] and not isinstance(
                value, (dict, list)
            ):
                try:
                    mlflow.log_param(f"train_{key}", value)
                except Exception:
                    pass  # Skip non-serializable params

        model = _build_model_for_resume(ctx, train_env, ctx.resume_artifact)
        resume_start_timesteps = 0
        if ctx.resume_artifact is not None:
            resume_metadata = load_checkpoint_metadata(ctx.resume_artifact.model_path)
            resume_start_timesteps = int(
                resume_metadata.get(
                    "num_timesteps",
                    ctx.resume_artifact.step if ctx.resume_artifact.step is not None else 0,
                )
            )
            if resume_start_timesteps <= 0:
                resume_start_timesteps = int(getattr(model, "num_timesteps", 0))
            _set_model_num_timesteps(model, resume_start_timesteps)
        else:
            resume_metadata = {}

        remaining_timesteps = max(0, int(ctx.total_timesteps) - resume_start_timesteps)
        final_step = resume_start_timesteps

        eval_freq = int(ctx.eval_cfg.get("eval_freq", 25_000))
        callbacks: list[BaseCallback] = [
            MLflowMetricsCallback(),
            RewardHackMonitorCallback(log_freq=eval_freq),
        ]

        save_freq_steps = int(ctx.train_cfg.get("save_freq_steps", 50_000))
        if save_freq_steps > 0:
            callbacks.append(
                PeriodicCheckpointCallback(
                    save_freq=save_freq_steps,
                    save_path=run_checkpoint_dir,
                    name_prefix=ctx.algorithm.lower(),
                    algorithm=ctx.algorithm,
                    run_id=ctx.run_id,
                    save_replay_buffer=ctx.algorithm == "SAC",
                )
            )

        if eval_freq > 0:
            validation_seed = int(
                ctx.eval_cfg.get("validation_seed", ctx.seed + 10_000)
            )
            val_env = make_env_from_config(
                ctx.env_cfg,
                seed=validation_seed,
                reference_obs_space=train_env.observation_space,
            )
            val_callback = ValidationCallback(
                eval_env=val_env,
                eval_freq=eval_freq,
                n_eval_episodes=int(ctx.eval_cfg.get("n_eval_episodes", 20)),
                log_path=run_output_dir / "validation_curve.csv",
                best_model_path=run_checkpoint_dir / "best_model.zip",
                validation_seed=validation_seed,
                patience=int(ctx.eval_cfg.get("early_stopping_patience", 0)),
                deterministic=bool(ctx.eval_cfg.get("deterministic", True)),
            )
            if ctx.resume_artifact is not None:
                _seed_validation_callback_from_snapshot(val_callback, resume_snapshot)
            callbacks.append(val_callback)

        mlflow.log_param("resume_start_timesteps", resume_start_timesteps)
        mlflow.log_param("resume_remaining_timesteps", remaining_timesteps)
        if ctx.resume_artifact is not None:
            mlflow.log_param("resume_checkpoint_path", str(ctx.resume_artifact.model_path))

        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callbacks or None,
                reset_num_timesteps=ctx.resume_artifact is None,
            )
        elif ctx.resume_artifact is not None:
            print(
                f"[train] resume checkpoint already meets target timesteps "
                f"({resume_start_timesteps} >= {ctx.total_timesteps}); skipping learn()"
            )

        final_model_path = run_checkpoint_dir / "final_model.zip"
        model.save(str(final_model_path))
        if ctx.algorithm == "SAC" and hasattr(model, "save_replay_buffer"):
            try:
                model.save_replay_buffer(
                    str(final_model_path.with_name(f"{final_model_path.stem}_replay_buffer.pkl"))
                )
            except Exception as exc:  # pragma: no cover - best-effort persistence.
                print(f"[train] Warning: failed to save final replay buffer: {exc}")
        save_checkpoint_metadata(
            final_model_path,
            run_id=ctx.run_id,
            algorithm=ctx.algorithm,
            num_timesteps=int(getattr(model, "num_timesteps", resume_start_timesteps)),
            source="final",
            save_replay_buffer=ctx.algorithm == "SAC",
            extra={"target_total_timesteps": int(ctx.total_timesteps)},
        )

        _final_eval_env = make_env_from_config(
            ctx.env_cfg,
            seed=ctx.seed,
            reference_obs_space=train_env.observation_space,
        )
        final_metrics = _evaluate_policy(
            model,
            _final_eval_env,
            episodes=int(ctx.train_cfg.get("eval_episodes", 5)),
            seed=ctx.seed,
        )
        _final_eval_env.close()
        final_step = int(getattr(model, "num_timesteps", resume_start_timesteps))

        _export_training_curve(monitor_path, run_output_dir / "training_curve.csv")

        summary = {
            "run_id": ctx.run_id,
            "task": ctx.env_cfg.task,
            "algorithm": ctx.algorithm,
            "seed": ctx.seed,
            "total_timesteps": ctx.total_timesteps,
            "completed_timesteps": final_step,
            "auto_resume": ctx.auto_resume,
            "resume_mode": ctx.resume_artifact is not None,
            "resume_from": ctx.resume_from,
            "resume_start_timesteps": resume_start_timesteps,
            "resume_remaining_timesteps": remaining_timesteps,
            "final_model": str(final_model_path),
            "train_return_mean": final_metrics["return_mean"],
            "train_return_std": final_metrics["return_std"],
            "train_success_rate": final_metrics["success_rate"],
            "train_episode_length_mean": final_metrics["episode_length_mean"],
            "train_episode_length_std": final_metrics["episode_length_std"],
            "train_action_magnitude_mean": final_metrics["action_magnitude_mean"],
            "train_action_magnitude_std": final_metrics["action_magnitude_std"],
            "eval_return_mean": final_metrics["return_mean"],
            "eval_return_std": final_metrics["return_std"],
            "eval_success_rate": final_metrics["success_rate"],
            "eval_episode_length_mean": final_metrics["episode_length_mean"],
            "eval_episode_length_std": final_metrics["episode_length_std"],
            "eval_action_magnitude_mean": final_metrics["action_magnitude_mean"],
            "eval_action_magnitude_std": final_metrics["action_magnitude_std"],
        }
        if "door_angle_final_mean" in final_metrics:
            summary["train_door_angle_final_mean"] = final_metrics["door_angle_final_mean"]
            summary["train_door_angle_final_std"] = final_metrics[
                "door_angle_final_std"
            ]
            summary["eval_door_angle_final_mean"] = final_metrics[
                "door_angle_final_mean"
            ]
            summary["eval_door_angle_final_std"] = final_metrics[
                "door_angle_final_std"
            ]

        if val_callback is not None:
            summary.update(
                {
                    "best_model": str(run_checkpoint_dir / "best_model.zip"),
                    "best_validation_success_rate": val_callback.best_success,
                    "best_validation_return_mean": val_callback.best_return_mean,
                    "best_validation_episode_length_mean": val_callback.best_episode_length_mean,
                    "best_validation_action_magnitude_mean": val_callback.best_action_magnitude_mean,
                    "best_validation_step": val_callback.best_step,
                }
            )
            if np.isfinite(val_callback.best_door_angle_final_mean):
                summary["best_validation_door_angle_final_mean"] = (
                    val_callback.best_door_angle_final_mean
                )
            summary["train_validation_success_gap_best"] = (
                final_metrics["success_rate"] - val_callback.best_success
            )
        elif ctx.resume_artifact is not None and resume_snapshot:
            for key in (
                "best_validation_success_rate",
                "best_validation_return_mean",
                "best_validation_episode_length_mean",
                "best_validation_action_magnitude_mean",
                "best_validation_door_angle_final_mean",
                "best_validation_step",
            ):
                if key in resume_snapshot:
                    summary[key] = resume_snapshot[key]

        # Log final metrics to MLflow
        mlflow.log_metrics(
            prefixed_metrics(final_metrics, "train_"),
            step=final_step,
        )

        if val_callback is not None:
            mlflow.log_metrics(
                {
                    "best_validation_success_rate": val_callback.best_success,
                    "best_validation_return_mean": val_callback.best_return_mean,
                    "best_validation_episode_length_mean": val_callback.best_episode_length_mean,
                    "best_validation_action_magnitude_mean": val_callback.best_action_magnitude_mean,
                    "best_validation_step": val_callback.best_step,
                },
                step=val_callback.best_step,
            )
            if np.isfinite(val_callback.best_door_angle_final_mean):
                mlflow.log_metric(
                    "best_validation_door_angle_final_mean",
                    val_callback.best_door_angle_final_mean,
                    step=val_callback.best_step,
                )
            mlflow.log_metric(
                "train_validation_success_gap_best",
                final_metrics["success_rate"] - val_callback.best_success,
                step=final_step,
            )

        summary["best_run_video_enabled"] = best_run_video_enabled
        summary["best_run_video_fps"] = best_run_video_fps
        summary["best_run_video_max_steps"] = best_run_video_max_steps
        summary["best_run_video_min_seconds"] = best_run_video_min_seconds
        summary["best_run_video_max_episodes"] = best_run_video_max_episodes
        if best_run_video_enabled:
            best_video_output_dir = ensure_dir(run_output_dir / "videos")
            best_video_output_path = (
                best_video_output_dir / f"{ctx.run_id}_best_arm_views.mp4"
            )
            try:
                video_metadata = render_best_checkpoint_video(
                    ctx.env_cfg,
                    run_checkpoint_dir,
                    algorithm=ctx.algorithm,
                    device=ctx.train_cfg.get("device", "auto"),
                    seed=ctx.seed,
                    output=best_video_output_path,
                    video_fps=best_run_video_fps,
                    max_steps=best_run_video_max_steps,
                    min_seconds=best_run_video_min_seconds,
                    max_episodes=best_run_video_max_episodes,
                )
                summary["best_run_video"] = video_metadata["video_path"]
                summary["best_run_video_metadata"] = video_metadata["metadata_path"]
                summary["best_run_video_source"] = video_metadata["checkpoint"]
                summary["best_run_video_frames"] = video_metadata["num_frames"]
                try:
                    mlflow.log_artifact(video_metadata["video_path"])
                    mlflow.log_artifact(video_metadata["metadata_path"])
                except Exception as exc:
                    print(f"[train] Warning: failed to log best-run video artifact: {exc}")
            except Exception as exc:
                summary["best_run_video_error"] = str(exc)
                print(f"[train] Warning: failed to render best-run video: {exc}")

        with (run_output_dir / "train_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Store resolved config for reproducibility and experiment tracing.
        with (run_output_dir / "resolved_train_config.yaml").open(
            "w", encoding="utf-8"
        ) as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Log artifacts to MLflow
        mlflow.log_artifact(str(run_output_dir / "train_summary.json"))
        mlflow.log_artifact(str(run_output_dir / "resolved_train_config.yaml"))
        if (run_output_dir / "validation_curve.csv").exists():
            mlflow.log_artifact(str(run_output_dir / "validation_curve.csv"))
        if (run_output_dir / "training_curve.csv").exists():
            mlflow.log_artifact(str(run_output_dir / "training_curve.csv"))
        if (run_checkpoint_dir / "best_model.zip").exists():
            mlflow.log_artifact(str(run_checkpoint_dir / "best_model.zip"))
        if final_model_path is not None and final_model_path.exists():
            mlflow.log_artifact(str(final_model_path))

    finally:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        train_env.close()

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
