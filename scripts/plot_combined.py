"""Generate combined comparison plots — one graph per metric, all runs overlaid.

5 key metrics chosen to tell the full diagnostic story:
  1. val_door_angle_max_mean  — main result: how much the door opened
  2. train/ent_coef           — v1/v2 crash vs v3+ stability
  3. train/critic_loss        — Q-function divergence vs convergence
  4. val_return_mean          — overall cumulative reward
  5. val_approach_frac_mean   — hover-hacking detector

All curves capped at MAX_STEPS (400k). Output: docs/courbes/combined/
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Runs to include  (order = legend order)
# ---------------------------------------------------------------------------

RUNS = [
    {
        "key": "run_sac_v1",
        "run_id": "3518295f61e84b49",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260506_201751",
        "label": "SAC v1 — ent_coef=auto",
        "color": "#e05c5c",
        "linestyle": "-",
    },
    {
        "key": "run_sac_v2",
        "run_id": "6048c37a1bd44ca8a31ed2d5bc406a8a",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_073628",
        "label": "SAC v2 — ent_coef=auto_0.1",
        "color": "#e0964a",
        "linestyle": "-",
    },
    {
        "key": "run_sac_v3",
        "run_id": "4956899f0f484aeda59ca882ee254754",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_112859",
        "label": "SAC v3 — ent_coef=0.1 fixe",
        "color": "#4a90d9",
        "linestyle": "-",
    },
    {
        "key": "run_sac_v3_curriculum",
        "run_id": "9ac4e34fe9d54775bdb75e5ae35846e4",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_133418",
        "label": "SAC v3 Curriculum",
        "color": "#5bac6e",
        "linestyle": "-",
    },
    {
        "key": "run_sac_her_v2",
        "run_id": "94ad73ca2b1f48ee",
        "output_dir": "outputs/OpenCabinet_SAC_seed0_20260507_174253",
        "label": "SAC HER v2 — hybride",
        "color": "#1abc9c",
        "linestyle": "-",
    },
]

MAX_STEPS = 400_000

# ---------------------------------------------------------------------------
# Metrics: (csv_col_or_mlflow_key, from_mlflow, title, ylabel, filename)
# ---------------------------------------------------------------------------

METRICS = [
    (
        "val_door_angle_max_mean",
        False,
        "Angle max porte en validation (rad)\n— métrique principale : ouverture réelle de la porte",
        "Angle (rad)",
        "combined_val_door_angle_max.png",
    ),
    (
        "train/ent_coef",
        True,
        "Coefficient d'entropie α (train)\n— crash vers 0 dans v1/v2, stable à 0.1 dans v3+",
        "α",
        "combined_train_ent_coef.png",
    ),
    (
        "train/critic_loss",
        True,
        "Critic loss (train)\n— divergence dans v1/v2, convergence dans v3+",
        "MSE",
        "combined_train_critic_loss.png",
    ),
    (
        "val_return_mean",
        False,
        "Return validation moyen\n— progression globale de la politique",
        "Return",
        "combined_val_return_mean.png",
    ),
    (
        "val_approach_frac_mean",
        False,
        "Fraction reward approche en validation\n— détecteur de hover-hacking",
        "Fraction",
        "combined_val_approach_frac.png",
    ),
    (
        "train/actor_loss",
        True,
        "Actor loss (train)\n— décroissant = Q-values croissantes ; remontée positive = divergence",
        "Loss",
        "combined_train_actor_loss.png",
    ),
]


def _smooth(y: np.ndarray, window: int = 3) -> np.ndarray:
    if len(y) < window:
        return y
    kernel = np.ones(window) / window
    return np.convolve(y, kernel, mode="same")


def load_val_col(output_dir: str, col: str, max_steps: int):
    p = Path(output_dir) / "validation_curve.csv"
    if not p.exists():
        return None, None
    df = pd.read_csv(p)
    if col not in df.columns or "step" not in df.columns:
        return None, None
    mask = df["step"] <= max_steps
    steps = df.loc[mask, "step"].values
    values = df.loc[mask, col].values
    if len(steps) == 0:
        return None, None
    return steps, values


def load_mlflow_col(client, run_id: str, key: str, max_steps: int):
    try:
        history = client.get_metric_history(run_id, key)
        if not history:
            return None, None
        steps = np.array([m.step for m in history])
        values = np.array([m.value for m in history])
        order = np.argsort(steps)
        steps, values = steps[order], values[order]
        mask = steps <= max_steps
        steps, values = steps[mask], values[mask]
        if len(steps) == 0:
            return None, None
        return steps, values
    except Exception:
        return None, None


def plot_combined(metric_key, from_mlflow, title, ylabel, out_path, client):
    fig, ax = plt.subplots(figsize=(10, 5))
    any_plotted = False

    for run in RUNS:
        if from_mlflow:
            steps, values = load_mlflow_col(client, run["run_id"], metric_key, MAX_STEPS)
        else:
            steps, values = load_val_col(run["output_dir"], metric_key, MAX_STEPS)

        if steps is None or len(steps) < 2:
            continue

        color = run["color"]
        label = run["label"]
        ls = run.get("linestyle", "-")

        # raw (faint)
        ax.plot(steps / 1000, values, alpha=0.2, color=color, linewidth=0.8, linestyle=ls)
        # smoothed
        smoothed = _smooth(values, window=3)
        ax.plot(steps / 1000, smoothed, color=color, linewidth=2.0,
                label=label, linestyle=ls)
        any_plotted = True

    if not any_plotted:
        plt.close(fig)
        return False

    ax.set_xlabel("Steps (k)", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, pad=10)
    ax.set_xlim(left=0, right=MAX_STEPS / 1000)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def main():
    import mlflow
    mlflow.set_tracking_uri("mlruns")
    client = mlflow.tracking.MlflowClient()

    out_dir = Path("docs/courbes/combined")
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric_key, from_mlflow, title, ylabel, fname in METRICS:
        out_path = out_dir / fname
        ok = plot_combined(metric_key, from_mlflow, title, ylabel, out_path, client)
        status = "OK" if ok else "skip (no data)"
        print(f"  [{status}] {fname}")

    print(f"\nDone — {out_dir}")


if __name__ == "__main__":
    main()
