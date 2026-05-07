"""Generate PNG curve plots for all runs > 200k steps.

Creates docs/courbes/<run_label>/ for each qualifying run with:
- Validation metrics (from validation_curve.csv)
- Train/reward_hack metrics (from MLflow)
- Monitor episode returns (smoothed)
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RUNS = {
    "run_sac_v1": {
        "run_id": "1fcab0b7b43c420ab696d4fc112754c9",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260506_201751",
        "label": "SAC v1 — ent_coef=auto (500k steps)",
        "color": "#e05c5c",
    },
    "run_sac_v2": {
        "run_id": "6048c37a1bd44ca8a31ed2d5bc406a8a",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_073628",
        "label": "SAC v2 — ent_coef=auto_0.1, target_entropy=-4 (900k steps)",
        "color": "#e0964a",
    },
    "run_sac_v3": {
        "run_id": "4956899f0f484aeda59ca882ee254754",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_112859",
        "label": "SAC v3 — ent_coef=0.1 fixed, SDE (400k steps)",
        "color": "#4a90d9",
    },
    "run_sac_v3_curriculum": {
        "run_id": "9ac4e34fe9d54775bdb75e5ae35846e4",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_133418",
        "label": "SAC v3 Curriculum — theta_success=0.40, spawn=0.05 (500k steps)",
        "color": "#5bac6e",
    },
}

MIN_STEPS = 200_000

VALIDATION_COLS = [
    ("val_return_mean", "Validation Return (mean)", "Return"),
    ("val_success_rate", "Validation Success Rate", "Rate"),
    ("val_door_angle_final_mean", "Validation Door Angle Final (rad)", "Angle (rad)"),
    ("val_door_angle_max_mean", "Validation Door Angle Max (rad)", "Angle (rad)"),
    ("val_episode_length_mean", "Validation Episode Length", "Steps"),
    ("val_approach_frac_mean", "Validation Approach Fraction", "Fraction"),
    ("val_action_magnitude_mean", "Validation Action Magnitude", "L2 norm"),
    ("val_action_smoothness_mean", "Validation Action Smoothness (jerk)", "||Δa||"),
    ("val_stagnation_steps_mean", "Validation Stagnation Steps", "Steps"),
    ("val_sign_changes_mean", "Validation Oscillation (sign changes)", "Count"),
    ("val_reward_without_success", "Validation Reward (failed episodes)", "Return"),
]

MLFLOW_METRICS = [
    ("train/actor_loss", "Train Actor Loss", "Loss"),
    ("train/critic_loss", "Train Critic Loss", "Loss"),
    ("train/ent_coef", "Train Entropy Coefficient (α)", "α"),
    ("train/learning_rate", "Learning Rate", "LR"),
    ("train/n_updates", "N Updates", "Updates"),
    ("reward_hack/theta_best_mean", "Training θ Best (door angle)", "rad"),
    ("reward_hack/train_success_rate", "Training Success Rate", "Rate"),
    ("reward_hack/success_frac", "Success Reward Fraction", "Fraction"),
    ("reward_hack/approach_frac", "Approach Reward Fraction", "Fraction"),
    ("reward_hack/progress_frac", "Progress Reward Fraction", "Fraction"),
    ("reward_hack/oscillation_frac", "Oscillation Penalty Fraction", "Fraction"),
    ("reward_hack/stagnation_rate", "Stagnation Rate", "Rate"),
    ("reward_hack/reward_without_success", "Training Reward (failed episodes)", "Return"),
    ("reward_hack/episodes", "Episodes per log window", "Count"),
]


def _smooth(y: np.ndarray, window: int = 5) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def fetch_mlflow_metric(client, run_id: str, key: str):
    try:
        history = client.get_metric_history(run_id, key)
        if not history:
            return None, None
        steps = np.array([m.step for m in history])
        values = np.array([m.value for m in history])
        order = np.argsort(steps)
        return steps[order], values[order]
    except Exception:
        return None, None


def plot_metric(ax, steps, values, label, ylabel, color, smooth=3):
    ax.plot(steps / 1000, values, alpha=0.3, color=color, linewidth=0.8)
    if len(values) >= smooth:
        ax.plot(steps / 1000, _smooth(values, smooth), color=color, linewidth=1.8, label=label)
    else:
        ax.plot(steps / 1000, values, color=color, linewidth=1.8, label=label)
    ax.set_xlabel("Steps (k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)


def generate_plots_for_run(run_name: str, cfg: dict, out_dir: Path, client):
    out_dir.mkdir(parents=True, exist_ok=True)
    color = cfg["color"]
    label = cfg["label"]
    run_id = cfg["run_id"]
    output_dir = Path(cfg["output_dir"])

    # ---- Validation curves from CSV ----------------------------------------
    val_csv = output_dir / "validation_curve.csv"
    if val_csv.exists():
        df = pd.read_csv(val_csv)
        # Check max step
        if df["step"].max() < MIN_STEPS:
            print(f"  [skip] {run_name}: max_step={df['step'].max():.0f} < {MIN_STEPS}")
            return
        steps = df["step"].values

        for col, title, ylabel in VALIDATION_COLS:
            if col not in df.columns:
                continue
            values = df[col].values
            fig, ax = plt.subplots(figsize=(8, 4))
            plot_metric(ax, steps, values, label, ylabel, color)
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=7, loc="best")
            plt.tight_layout()
            safe_col = col.replace("/", "_").replace(" ", "_")
            fig.savefig(out_dir / f"{safe_col}.png", dpi=120)
            plt.close(fig)
            print(f"  [val] {col}")

    # ---- Monitor (episode returns) -----------------------------------------
    mon_csv = output_dir / "monitor.csv"
    if mon_csv.exists():
        try:
            mon = pd.read_csv(mon_csv, comment="#", header=0)
            if "r" in mon.columns and len(mon) > 10:
                rewards = mon["r"].values
                ep_idx = np.arange(len(rewards))
                fig, ax = plt.subplots(figsize=(8, 4))
                ax.plot(ep_idx, rewards, alpha=0.2, color=color, linewidth=0.5)
                ax.plot(ep_idx, _smooth(rewards, 50), color=color, linewidth=1.8, label=label)
                ax.set_xlabel("Episode")
                ax.set_ylabel("Return")
                ax.set_title("Training Episode Return (smoothed)", fontsize=11)
                ax.legend(fontsize=7, loc="best")
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                fig.savefig(out_dir / "monitor_episode_return.png", dpi=120)
                plt.close(fig)
                print(f"  [monitor] episode_return")
        except Exception as e:
            print(f"  [monitor] error: {e}")

    # ---- MLflow metrics ----------------------------------------------------
    for metric_key, title, ylabel in MLFLOW_METRICS:
        steps, values = fetch_mlflow_metric(client, run_id, metric_key)
        if steps is None or len(steps) < 2:
            continue
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_metric(ax, steps, values, label, ylabel, color)
        ax.set_title(title, fontsize=11)
        ax.legend(fontsize=7, loc="best")
        plt.tight_layout()
        safe_key = metric_key.replace("/", "_").replace(" ", "_")
        fig.savefig(out_dir / f"mlflow_{safe_key}.png", dpi=120)
        plt.close(fig)
        print(f"  [mlflow] {metric_key}")


def main():
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    docs_dir = Path("docs/courbes")

    for run_name, cfg in RUNS.items():
        out_dir = docs_dir / run_name
        print(f"\n=== {run_name} → {out_dir} ===")
        generate_plots_for_run(run_name, cfg, out_dir, client)

    print("\nDone.")


if __name__ == "__main__":
    main()
