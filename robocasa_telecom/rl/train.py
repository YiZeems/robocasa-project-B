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
import json
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
from stable_baselines3.common.monitor import Monitor

from ..utils.checkpoints import (
    CheckpointArtifact,
    load_checkpoint_metadata,
    resolve_checkpoint_artifact,
    save_checkpoint_metadata,
)
from ..envs.factory import EnvConfig, load_env_config, make_env_from_config
from ..utils.metrics import prefixed_metrics, summarize_rollout_episodes
from ..utils.io import ensure_dir, load_yaml


SUPPORTED_ALGOS = {"PPO", "SAC"}


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
    )


def _resolve_policy_kwargs(train_cfg: dict[str, Any]) -> dict[str, Any]:
    """Build policy_kwargs dict from optional `pi`/`qf` size hints in YAML."""

    policy_kwargs = dict(train_cfg.get("policy_kwargs") or {})
    net_arch = train_cfg.get("net_arch")
    if net_arch is not None and "net_arch" not in policy_kwargs:
        policy_kwargs["net_arch"] = list(net_arch)
    return policy_kwargs


def _build_ppo(
    env: Monitor, train_cfg: dict[str, Any], tensorboard_root: Path, seed: int
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
        tensorboard_log=str(tensorboard_root),
        policy_kwargs=policy_kwargs or None,
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=1,
    )


def _build_sac(
    env: Monitor, train_cfg: dict[str, Any], tensorboard_root: Path, seed: int
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
        tensorboard_log=str(tensorboard_root),
        policy_kwargs=policy_kwargs or None,
        seed=seed,
        device=train_cfg.get("device", "auto"),
        verbose=1,
    )


def _build_model(
    algorithm: str,
    env: Monitor,
    train_cfg: dict[str, Any],
    tensorboard_root: Path,
    seed: int,
) -> BaseAlgorithm:
    """Dispatch model construction by algorithm name."""

    if algorithm == "PPO":
        return _build_ppo(env, train_cfg, tensorboard_root, seed)
    if algorithm == "SAC":
        return _build_sac(env, train_cfg, tensorboard_root, seed)
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
        if self.n_calls == 0 or self.n_calls % self._eval_freq != 0:
            return True

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

        row = {
            "step": float(self.num_timesteps),
            "val_success_rate": success,
            "val_return_mean": return_mean,
            "val_return_std": float(metrics["return_std"]),
            "val_episode_length_mean": episode_length_mean,
            "val_episode_length_std": float(metrics["episode_length_std"]),
            "val_action_magnitude_mean": action_magnitude_mean,
            "val_action_magnitude_std": float(metrics["action_magnitude_std"]),
        }
        if "door_angle_final_mean" in metrics:
            row["val_door_angle_final_mean"] = door_angle_final_mean
            row["val_door_angle_final_std"] = float(metrics["door_angle_final_std"])
        self._history.append(row)

        # Log metrics to MLflow
        mlflow.log_metrics(
            prefixed_metrics(
                {
                    "success_rate": success,
                    "return_mean": return_mean,
                    "return_std": float(metrics["return_std"]),
                    "episode_length_mean": episode_length_mean,
                    "episode_length_std": float(metrics["episode_length_std"]),
                    "action_magnitude_mean": action_magnitude_mean,
                    "action_magnitude_std": float(metrics["action_magnitude_std"]),
                    "door_angle_final_mean": metrics.get("door_angle_final_mean"),
                    "door_angle_final_std": metrics.get("door_angle_final_std"),
                },
                "validation_",
            ),
            step=self.num_timesteps,
        )

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
        if self.n_calls == 0 or self.n_calls % self._save_freq != 0:
            return True

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
    tensorboard_root: Path,
    resume_artifact: CheckpointArtifact | None,
) -> BaseAlgorithm:
    """Create or resume a model, restoring replay buffer when available."""

    device = ctx.train_cfg.get("device", "auto")
    if resume_artifact is None:
        return _build_model(
            ctx.algorithm, train_env, ctx.train_cfg, tensorboard_root, ctx.seed
        )

    load_kwargs = {
        "env": train_env,
        "device": device,
        "tensorboard_log": str(tensorboard_root),
        "print_system_info": False,
    }
    if ctx.algorithm == "PPO":
        model = PPO.load(str(resume_artifact.model_path), **load_kwargs)
    elif ctx.algorithm == "SAC":
        model = SAC.load(str(resume_artifact.model_path), **load_kwargs)
    else:  # pragma: no cover - guarded by config validation.
        raise ValueError(f"Unsupported algorithm: {ctx.algorithm}")

    if ctx.algorithm == "SAC" and resume_artifact.replay_buffer_path.exists():
        try:
            model.load_replay_buffer(str(resume_artifact.replay_buffer_path))
        except Exception as exc:  # pragma: no cover - best-effort resume.
            print(
                f"[train] Warning: failed to restore replay buffer from "
                f"{resume_artifact.replay_buffer_path}: {exc}"
            )

    return model


def main() -> None:
    """Train an RL model and export reproducible artifacts."""

    args = parse_args()
    cfg = load_yaml(args.config)
    ctx = _resolve_run_context(cfg, args)

    output_root = ensure_dir(ctx.paths_cfg.get("output_root", "outputs"))
    checkpoint_root = ensure_dir(ctx.paths_cfg.get("checkpoint_root", "checkpoints"))
    tensorboard_root = ensure_dir(
        ctx.paths_cfg.get("tensorboard_root", "logs/tensorboard")
    )

    run_output_dir = ensure_dir(output_root / ctx.run_id)
    run_checkpoint_dir = ensure_dir(checkpoint_root / ctx.run_id)

    train_env = make_env_from_config(ctx.env_cfg, seed=ctx.seed)
    monitor_path = run_output_dir / "monitor.csv"
    monitor_override_existing = not (
        ctx.resume_artifact is not None
        and monitor_path.exists()
        and monitor_path.stat().st_size > 0
    )
    train_env = Monitor(
        train_env,
        filename=str(monitor_path),
        override_existing=monitor_override_existing,
    )
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
        mlflow.log_param("resume_mode", ctx.resume_artifact is not None)
        mlflow.log_param("resume_from", ctx.resume_from or "")
        mlflow.log_param("eval_freq", int(ctx.eval_cfg.get("eval_freq", 0)))
        mlflow.log_param("n_eval_episodes", int(ctx.eval_cfg.get("n_eval_episodes", 0)))
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

        model = _build_model_for_resume(
            ctx, train_env, tensorboard_root, ctx.resume_artifact
        )
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

        callbacks: list[BaseCallback] = []

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

        eval_freq = int(ctx.eval_cfg.get("eval_freq", 0))
        if eval_freq > 0:
            validation_seed = int(
                ctx.eval_cfg.get("validation_seed", ctx.seed + 10_000)
            )
            val_env = make_env_from_config(ctx.env_cfg, seed=validation_seed)
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

        final_metrics = _evaluate_policy(
            model,
            train_env,
            episodes=int(ctx.train_cfg.get("eval_episodes", 5)),
            seed=ctx.seed,
        )
        final_step = int(getattr(model, "num_timesteps", resume_start_timesteps))

        _export_training_curve(monitor_path, run_output_dir / "training_curve.csv")

        summary = {
            "run_id": ctx.run_id,
            "task": ctx.env_cfg.task,
            "algorithm": ctx.algorithm,
            "seed": ctx.seed,
            "total_timesteps": ctx.total_timesteps,
            "completed_timesteps": final_step,
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
