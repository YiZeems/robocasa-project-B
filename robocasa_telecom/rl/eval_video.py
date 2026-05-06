"""Best-episode video evaluation for RoboCasa checkpoints.

Two-pass approach:
  Pass 1 — Run N episodes WITHOUT render (fast, full metrics collected).
            Score each episode with a reward-hacking-resistant criterion.
  Pass 2 — Re-run best and worst episodes WITH render (reproducible via seed).
            Save MP4 files, metadata JSON, and debug CSV.

Why two passes?
  SubprocVecEnv workers run in separate processes and cannot render.
  Even for single-env eval, buffering N * 500 frames in RAM is ~2 GB for N=20.
  Re-running with the same seed is cheaper and fully reproducible.

Scoring criterion (reward-hacking resistant):
  score = 1000 * success
        + 100  * door_angle_final   (normalised [0,1])
        + 10   * door_angle_max     (high-watermark, anti-oscillation)
        + 1    * episode_return     (tiebreaker only, intentionally small)
        - 0.1  * stagnation_steps   (penalise hover hacking)
        - 0.01 * episode_length     (prefer faster successes)

Usage:
  make eval-video CONFIG=... CHECKPOINT=... EPISODES=20

  uv run python -m robocasa_telecom.eval_video \\
    --config configs/train/open_single_door_sac_debug.yaml \\
    --checkpoint checkpoints/OpenCabinet_SAC_seed0_.../best_model.zip \\
    --episodes 20 --seed 0 --out outputs/eval/videos/

Outputs:
  best_episode.mp4
  worst_episode.mp4                (optional, for failure analysis)
  best_episode_metadata.json
  video_selection_debug.csv
  (all logged to MLflow as artifacts)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm

from ..envs.factory import EnvConfig, load_env_config, make_env_from_config
from ..utils.checkpoints import resolve_run_checkpoint_path
from ..utils.device import resolve_device
from ..utils.io import ensure_dir, load_yaml
from ..utils.metrics import DOOR_ANGLE_KEYS, extract_scalar_metric
from ..utils.success import infer_success
from ..utils.video import ensure_uint8_frame, grid_2x2, resolve_arm_video_cameras, save_mp4


SUPPORTED_ALGOS = ("PPO", "SAC")


# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------

@dataclass
class EpisodeData:
    episode_id: int
    seed_used: int
    success: bool = False
    episode_return: float = 0.0
    episode_length: int = 0
    door_angle_initial: float = 0.0
    door_angle_final: float = 0.0
    door_angle_max: float = 0.0
    door_angle_delta: float = 0.0
    stagnation_steps: int = 0
    oscillation_steps: int = 0
    sign_changes: int = 0
    action_norm_mean: float = 0.0
    time_to_success: int = -1       # step index of first success, -1 if not
    score: float = 0.0

    def compute_score(self) -> float:
        s = (
            1000.0 * float(self.success)
            + 100.0 * self.door_angle_final
            + 10.0  * self.door_angle_max
            + 1.0   * self.episode_return
            - 0.1   * self.stagnation_steps
            - 0.01  * self.episode_length
        )
        self.score = s
        return s

    def as_csv_row(self, selected_as_best: bool = False) -> dict[str, Any]:
        row = {
            "episode_id":               self.episode_id,
            "seed_used":                self.seed_used,
            "success":                  int(self.success),
            "episode_return":           round(self.episode_return, 4),
            "episode_length":           self.episode_length,
            "door_angle_initial":       round(self.door_angle_initial, 4),
            "door_angle_final":         round(self.door_angle_final, 4),
            "door_angle_max":           round(self.door_angle_max, 4),
            "door_angle_delta":         round(self.door_angle_delta, 4),
            "stagnation_steps":         self.stagnation_steps,
            "oscillation_steps":        self.oscillation_steps,
            "sign_changes":             self.sign_changes,
            "action_norm_mean":         round(self.action_norm_mean, 4),
            "time_to_success":          self.time_to_success,
            "score":                    round(self.score, 4),
            "selected_as_best":         int(selected_as_best),
        }
        return row


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_model(algorithm: str, checkpoint: str, device: str) -> BaseAlgorithm:
    if algorithm == "PPO":
        return PPO.load(checkpoint, device=device)
    if algorithm == "SAC":
        return SAC.load(checkpoint, device=device)
    raise ValueError(f"Unsupported algorithm: {algorithm}")


def _resolve_algorithm(cfg: dict[str, Any], override: str | None) -> str:
    raw = override or cfg.get("train", {}).get("algorithm") or "PPO"
    algo = str(raw).upper()
    if algo not in SUPPORTED_ALGOS:
        raise ValueError(f"Unsupported algorithm '{algo}'. Choose from: {SUPPORTED_ALGOS}")
    return algo


def _render_frame(env: Any, cameras: tuple[str, ...]) -> np.ndarray:
    """Render a 4-view grid frame from the environment."""
    sim = getattr(getattr(env, "raw_env", None), "sim", None)
    frames: list[np.ndarray] = []
    for cam in cameras:
        frame = None
        if sim is not None:
            try:
                frame = sim.render(height=256, width=256, camera_name=cam)
            except Exception:
                pass
        if frame is None:
            try:
                frame = env.raw_env.render(camera_name=cam)
            except Exception:
                pass
        if frame is None:
            frame = np.zeros((256, 256, 3), dtype=np.uint8)
        frames.append(ensure_uint8_frame(frame))
    return grid_2x2(list(frames))


# ---------------------------------------------------------------------------
# Pass 1: score all episodes (no render, fast)
# ---------------------------------------------------------------------------

def _run_scoring_pass(
    model: BaseAlgorithm,
    env_cfg: EnvConfig,
    n_episodes: int,
    base_seed: int,
    deterministic: bool,
) -> list[EpisodeData]:
    """Run N episodes without render, collect all anti-hacking metrics."""
    from dataclasses import replace
    score_env_cfg = replace(env_cfg, has_renderer=False, has_offscreen_renderer=False)
    env = make_env_from_config(
        score_env_cfg,
        seed=base_seed,
        reference_obs_space=getattr(model, "observation_space", None),
    )

    episodes: list[EpisodeData] = []

    for ep_id in range(n_episodes):
        ep_seed = base_seed + ep_id
        obs, _ = env.reset(seed=ep_seed)
        ep = EpisodeData(episode_id=ep_id, seed_used=ep_seed)

        # Door angle at reset (initial state)
        ep.door_angle_initial = float(
            extract_scalar_metric(None, env, DOOR_ANGLE_KEYS) or 0.0
        )
        ep.door_angle_max = ep.door_angle_initial
        ep_theta_best = ep.door_angle_initial

        done = False
        action_norms: list[float] = []
        step_idx = 0
        success_step = -1

        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            action_norms.append(float(np.linalg.norm(np.asarray(action, dtype=np.float32))))

            obs, reward, terminated, truncated, info = env.step(action)
            ep.episode_return += float(reward)
            ep.episode_length += 1
            step_idx += 1

            if not ep.success and infer_success(info, env):
                ep.success = True
                success_step = step_idx

            # Extract state from reward_components (populated by AntiHackingReward)
            comp = info.get("reward_components", {}) if isinstance(info, dict) else {}
            theta = comp.get("theta") or extract_scalar_metric(info, env, DOOR_ANGLE_KEYS) or 0.0
            d_ee = comp.get("d_ee_handle", 1.0)

            ep.door_angle_final = float(theta)
            ep.door_angle_max = max(ep.door_angle_max, float(theta))

            # Sign changes (oscillation)
            if float(theta) > ep_theta_best + 1e-4:
                ep_theta_best = float(theta)
            elif d_ee < 0.15:
                ep.stagnation_steps += 1

            sc = comp.get("sign_changes", 0.0)
            ep.sign_changes += int(sc)

            # Episode summary (emitted at done=True)
            if bool(terminated or truncated):
                ep_comp = info.get("episode_reward_components", {})
                if ep_comp:
                    ep.stagnation_steps = int(ep_comp.get("stagnation_steps", ep.stagnation_steps))
                    ep.oscillation_steps = int(ep_comp.get("oscillation_steps", 0))
                    ep.door_angle_max = max(ep.door_angle_max, float(ep_comp.get("theta_best", 0.0)))

            done = bool(terminated or truncated)

        ep.door_angle_delta = ep.door_angle_final - ep.door_angle_initial
        ep.time_to_success = success_step
        ep.action_norm_mean = float(np.mean(action_norms)) if action_norms else 0.0
        ep.compute_score()
        episodes.append(ep)

        print(
            f"  ep{ep_id:02d} seed={ep_seed} "
            f"success={ep.success} "
            f"θ_final={ep.door_angle_final:.3f} "
            f"θ_max={ep.door_angle_max:.3f} "
            f"ret={ep.episode_return:.2f} "
            f"score={ep.score:.1f}"
        )

    env.close()
    return episodes


# ---------------------------------------------------------------------------
# Pass 2: render a specific episode (deterministic, same seed)
# ---------------------------------------------------------------------------

def _render_episode(
    model: BaseAlgorithm,
    env_cfg: EnvConfig,
    ep: EpisodeData,
    cameras: tuple[str, ...],
    video_fps: int,
    out_path: Path,
    deterministic: bool,
    max_steps: int = 600,
) -> list[np.ndarray]:
    """Re-run a single episode with render enabled and save MP4."""
    from dataclasses import replace
    render_env_cfg = replace(env_cfg, has_offscreen_renderer=True, has_renderer=False)
    env = make_env_from_config(
        render_env_cfg,
        seed=ep.seed_used,
        reference_obs_space=getattr(model, "observation_space", None),
    )

    obs, _ = env.reset(seed=ep.seed_used)
    frames: list[np.ndarray] = []
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        frames.append(_render_frame(env, cameras))
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        step_count += 1

    # Capture final frame
    if done:
        frames.append(_render_frame(env, cameras))

    env.close()

    if frames:
        save_mp4(frames, out_path, fps=video_fps)
        print(f"  saved: {out_path} ({len(frames)} frames)")
    return frames


# ---------------------------------------------------------------------------
# Main eval_video function
# ---------------------------------------------------------------------------

def eval_video(
    cfg: dict[str, Any],
    env_cfg: EnvConfig,
    checkpoint_path: Path,
    *,
    algorithm: str,
    n_episodes: int = 20,
    base_seed: int = 0,
    deterministic: bool = True,
    out_dir: Path,
    video_fps: int = 20,
    device: str = "auto",
    mlflow_run_id: str | None = None,
    save_worst: bool = True,
) -> dict[str, Any]:
    """Run scoring pass then render best (and worst) episode."""

    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model = _load_model(algorithm, str(checkpoint_path), device)
    cameras = resolve_arm_video_cameras(env_cfg)

    print(f"\n[eval_video] Pass 1 — scoring {n_episodes} episodes (no render)")
    episodes = _run_scoring_pass(model, env_cfg, n_episodes, base_seed, deterministic)

    # Sort by score
    episodes.sort(key=lambda e: e.score, reverse=True)
    best = episodes[0]
    worst = episodes[-1]

    print(f"\n[eval_video] Best  ep{best.episode_id:02d}: score={best.score:.1f} success={best.success} θ={best.door_angle_final:.3f}")
    print(f"[eval_video] Worst ep{worst.episode_id:02d}: score={worst.score:.1f} success={worst.success} θ={worst.door_angle_final:.3f}")

    # Pass 2 — render
    run_name = checkpoint_path.parent.name
    best_mp4 = out_dir / f"{run_name}_best_episode_{timestamp}.mp4"
    worst_mp4 = out_dir / f"{run_name}_worst_episode_{timestamp}.mp4"

    print(f"\n[eval_video] Pass 2 — rendering best episode (seed={best.seed_used})")
    _render_episode(model, env_cfg, best, cameras, video_fps, best_mp4, deterministic)

    if save_worst and worst.episode_id != best.episode_id:
        print(f"[eval_video] Pass 2 — rendering worst episode (seed={worst.seed_used})")
        _render_episode(model, env_cfg, worst, cameras, video_fps, worst_mp4, deterministic)

    # Metadata JSON
    metadata: dict[str, Any] = {
        "run_id":             run_name,
        "checkpoint_path":    str(checkpoint_path),
        "algorithm":          algorithm,
        "n_episodes_scored":  n_episodes,
        "base_seed":          base_seed,
        "deterministic":      deterministic,
        "timestamp":          timestamp,
        "cameras":            list(cameras),
        # Best episode
        "best": {
            "episode_id":                 best.episode_id,
            "seed_used":                  best.seed_used,
            "success":                    best.success,
            "episode_return":             best.episode_return,
            "episode_length":             best.episode_length,
            "door_angle_initial":         best.door_angle_initial,
            "door_angle_final":           best.door_angle_final,
            "door_angle_max":             best.door_angle_max,
            "door_angle_delta":           best.door_angle_delta,
            "stagnation_steps":           best.stagnation_steps,
            "oscillation_steps":          best.oscillation_steps,
            "sign_changes":               best.sign_changes,
            "action_norm_mean":           best.action_norm_mean,
            "time_to_success":            best.time_to_success,
            "score":                      best.score,
            "reason_selected":            "max(1000*success + 100*θ_final + 10*θ_max + 1*return - 0.1*stagnation - 0.01*length)",
            "video_path":                 str(best_mp4),
        },
        # Aggregate stats
        "aggregate": {
            "success_rate":           float(np.mean([e.success for e in episodes])),
            "return_mean":            float(np.mean([e.episode_return for e in episodes])),
            "door_angle_final_mean":  float(np.mean([e.door_angle_final for e in episodes])),
            "door_angle_max_mean":    float(np.mean([e.door_angle_max for e in episodes])),
            "stagnation_steps_mean":  float(np.mean([e.stagnation_steps for e in episodes])),
            "oscillation_steps_mean": float(np.mean([e.oscillation_steps for e in episodes])),
        },
    }
    if save_worst and worst.episode_id != best.episode_id:
        metadata["worst"] = {
            "episode_id":     worst.episode_id,
            "seed_used":      worst.seed_used,
            "success":        worst.success,
            "score":          worst.score,
            "door_angle_final": worst.door_angle_final,
            "video_path":     str(worst_mp4),
        }

    meta_path = out_dir / f"{run_name}_best_episode_metadata_{timestamp}.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[eval_video] Metadata: {meta_path}")

    # Debug CSV — one row per episode
    csv_path = out_dir / f"{run_name}_video_selection_debug_{timestamp}.csv"
    best_ids = {best.episode_id}
    fieldnames = list(EpisodeData(0, 0).as_csv_row().keys())
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for ep in sorted(episodes, key=lambda e: e.episode_id):
            writer.writerow(ep.as_csv_row(selected_as_best=(ep.episode_id in best_ids)))
    print(f"[eval_video] Debug CSV: {csv_path}")

    # MLflow logging
    if mlflow_run_id is not None:
        try:
            with mlflow.start_run(run_id=mlflow_run_id):
                _mlflow_log(metadata, best_mp4, worst_mp4 if save_worst else None, meta_path, csv_path)
        except Exception as exc:
            print(f"[eval_video] MLflow logging failed: {exc}")
    else:
        # Active run if exists
        active = mlflow.active_run()
        if active is not None:
            try:
                _mlflow_log(metadata, best_mp4, worst_mp4 if save_worst else None, meta_path, csv_path)
            except Exception as exc:
                print(f"[eval_video] MLflow logging failed: {exc}")

    return metadata


def _mlflow_log(
    metadata: dict[str, Any],
    best_mp4: Path,
    worst_mp4: Path | None,
    meta_path: Path,
    csv_path: Path,
) -> None:
    """Log artifacts and metrics to the active MLflow run."""
    agg = metadata.get("aggregate", {})
    mlflow.log_metrics({
        "eval_video/success_rate":          agg.get("success_rate", 0.0),
        "eval_video/return_mean":           agg.get("return_mean", 0.0),
        "eval_video/door_angle_final_mean": agg.get("door_angle_final_mean", 0.0),
        "eval_video/door_angle_max_mean":   agg.get("door_angle_max_mean", 0.0),
        "eval_video/stagnation_steps_mean": agg.get("stagnation_steps_mean", 0.0),
        "eval_video/best_success":          float(metadata["best"]["success"]),
        "eval_video/best_door_angle_final": metadata["best"]["door_angle_final"],
        "eval_video/best_score":            metadata["best"]["score"],
    })
    if best_mp4.exists():
        mlflow.log_artifact(str(best_mp4), artifact_path="videos")
    if worst_mp4 is not None and worst_mp4.exists():
        mlflow.log_artifact(str(worst_mp4), artifact_path="videos")
    mlflow.log_artifact(str(meta_path), artifact_path="videos")
    mlflow.log_artifact(str(csv_path), artifact_path="videos")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Best-episode video evaluation for RoboCasa")
    p.add_argument("--config",      required=True, help="Path to train YAML config")
    p.add_argument("--checkpoint",  required=True, help="Checkpoint zip or run directory")
    p.add_argument("--algorithm",   default=None,  choices=list(SUPPORTED_ALGOS))
    p.add_argument("--episodes",    type=int, default=20, help="Episodes to score (pass 1)")
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--deterministic", action="store_true", default=True)
    p.add_argument("--out",         default=None, help="Output directory for videos")
    p.add_argument("--fps",         type=int, default=20)
    p.add_argument("--no-worst",    action="store_true", help="Skip worst episode video")
    p.add_argument("--mlflow-run-id", default=None, help="MLflow run ID to attach artifacts to")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    env_cfg_path = cfg.get("env", {}).get("config_path", "configs/env/open_single_door.yaml")
    env_cfg = load_env_config(env_cfg_path)
    algorithm = _resolve_algorithm(cfg, args.algorithm)

    # Anchor MLflow at an absolute file:// URI (works on Windows, any cwd).
    mlruns_dir = Path(cfg.get("paths", {}).get("mlruns_dir", "mlruns")).resolve()
    mlruns_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(mlruns_dir.as_uri())

    checkpoint_path = resolve_run_checkpoint_path(args.checkpoint, preference="best")
    out_dir = Path(args.out) if args.out else ensure_dir(
        Path(cfg.get("paths", {}).get("output_root", "outputs")) / "eval" / "videos"
    )

    metadata = eval_video(
        cfg=cfg,
        env_cfg=env_cfg,
        checkpoint_path=checkpoint_path,
        algorithm=algorithm,
        n_episodes=args.episodes,
        base_seed=args.seed,
        deterministic=args.deterministic,
        out_dir=out_dir,
        video_fps=args.fps,
        device=resolve_device(cfg.get("train", {}).get("device", "auto")),
        mlflow_run_id=args.mlflow_run_id,
        save_worst=not args.no_worst,
    )

    print("\n" + json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
