#!/usr/bin/env python3
"""Generate training diagnostic plots from validation_curve.csv files.

Usage:
  # Single run
  uv run python scripts/plot_training.py --run outputs/OpenCabinet_SAC_seed0_*/

  # Compare multiple runs (labels auto-generated from directory names)
  uv run python scripts/plot_training.py \\
      --run outputs/OpenCabinet_SAC_seed0_*/ outputs/OpenCabinet_SAC_seed1_*/ \\
      --label "seed0" "seed1"

  # Custom output dir
  uv run python scripts/plot_training.py --run outputs/OpenCabinet_SAC_seed0_*/ \\
      --out outputs/plots/

Plots generated:
  - success_rate.png          : val_success_rate vs timesteps (with 95% CI shading if multi-run)
  - return_mean.png           : val_return_mean vs timesteps
  - door_angle_final.png      : val_door_angle_final_mean vs timesteps
  - door_angle_max.png        : val_door_angle_max_mean vs timesteps
  - reward_components.png     : approach/progress/success/oscillation fractions
  - anti_hacking.png          : stagnation rate, sign changes, approach fraction
  - action_smoothness.png     : action magnitude + smoothness vs timesteps
  - summary.png               : 2x3 grid of key metrics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    print("matplotlib not available — install it with: uv add matplotlib")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("pandas not available — install it with: uv add pandas")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Colour palette (colour-blind safe)
# ---------------------------------------------------------------------------
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]


def _load_validation_curve(run_dir: Path) -> pd.DataFrame | None:
    for name in ("validation_curve.csv", "val_curve.csv"):
        p = run_dir / name
        if p.exists():
            try:
                return pd.read_csv(p)
            except Exception:
                return None
    return None


def _smooth(series: pd.Series, window: int = 3) -> pd.Series:
    return series.rolling(window=window, min_periods=1, center=True).mean()


def _setup_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}k"))
    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _plot_single(
    ax: plt.Axes,
    dfs: list[pd.DataFrame],
    labels: list[str],
    col: str,
    title: str,
    ylabel: str,
    smooth: int = 1,
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    _setup_ax(ax, title, "Timesteps", ylabel)
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if col not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        x = df["step"].values
        y = df[col].values
        if smooth > 1:
            y = _smooth(pd.Series(y), smooth).values
        ax.plot(x, y, label=label, color=color, linewidth=1.8)

    if len(dfs) > 1:
        ax.legend(fontsize=7, framealpha=0.7)
    if ymin is not None or ymax is not None:
        ax.set_ylim(ymin, ymax)


def _plot_multi_series(
    ax: plt.Axes,
    df: pd.DataFrame,
    cols: list[str],
    labels: list[str],
    title: str,
    ylabel: str,
    smooth: int = 1,
) -> None:
    _setup_ax(ax, title, "Timesteps", ylabel)
    for col, label, color in zip(cols, labels, PALETTE):
        if col not in df.columns:
            continue
        x = df["step"].values
        y = df[col].values
        if smooth > 1:
            y = _smooth(pd.Series(y), smooth).values
        ax.plot(x, y, label=label, color=color, linewidth=1.8)
    ax.legend(fontsize=7, framealpha=0.7)


def plot_success_rate(dfs, labels, out_dir, smooth=3):
    fig, ax = plt.subplots(figsize=(7, 4))
    _setup_ax(ax, "Validation Success Rate", "Timesteps", "Success Rate")
    for i, (df, label) in enumerate(zip(dfs, labels)):
        if "val_success_rate" not in df.columns:
            continue
        color = PALETTE[i % len(PALETTE)]
        x = df["step"].values
        y = _smooth(df["val_success_rate"], smooth).values
        ax.plot(x, y, label=label, color=color, linewidth=2)

        # 95% CI band if available
        lo_col = "val_success_ci_lo"
        hi_col = "val_success_ci_hi"
        if lo_col in df.columns and hi_col in df.columns:
            lo = _smooth(df[lo_col], smooth).values
            hi = _smooth(df[hi_col], smooth).values
            ax.fill_between(x, lo, hi, alpha=0.15, color=color)

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    if len(dfs) > 1:
        ax.legend(fontsize=8, framealpha=0.8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.6)
    ax.axhline(0.9, color="green", linestyle=":", linewidth=1, alpha=0.6, label="90% target")
    fig.tight_layout()
    fig.savefig(out_dir / "success_rate.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'success_rate.png'}")


def plot_return(dfs, labels, out_dir, smooth=3):
    fig, ax = plt.subplots(figsize=(7, 4))
    _plot_single(ax, dfs, labels, "val_return_mean", "Validation Return (mean)", "Episode Return", smooth)
    fig.tight_layout()
    fig.savefig(out_dir / "return_mean.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'return_mean.png'}")


def plot_door_angle(dfs, labels, out_dir, smooth=3):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    _plot_single(ax1, dfs, labels, "val_door_angle_final_mean",
                 "Door Angle at Episode End", "Norm. Door Opening [0,1]", smooth, 0, 1)
    _plot_single(ax2, dfs, labels, "val_door_angle_max_mean",
                 "Best Door Angle per Episode (HWM)", "Norm. Door Opening [0,1]", smooth, 0, 1)
    for ax in (ax1, ax2):
        ax.axhline(0.9, color="green", linestyle=":", linewidth=1.2, alpha=0.7)
    fig.tight_layout()
    fig.savefig(out_dir / "door_angle.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'door_angle.png'}")


def plot_anti_hacking(dfs, labels, out_dir, smooth=3):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    _plot_single(axes[0], dfs, labels, "val_approach_frac_mean",
                 "Approach Reward Fraction\n(>0.5 → hover hacking)",
                 "Fraction", smooth, 0, 1)
    axes[0].axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.7)

    _plot_single(axes[1], dfs, labels, "val_stagnation_steps_mean",
                 "Stagnation Steps / Episode\n(near handle, no progress)",
                 "Steps", smooth)

    _plot_single(axes[2], dfs, labels, "val_sign_changes_mean",
                 "Door Oscillation\n(sign changes per episode)",
                 "Count", smooth)

    fig.tight_layout()
    fig.savefig(out_dir / "anti_hacking.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'anti_hacking.png'}")


def plot_reward_components(dfs, labels, out_dir, smooth=3):
    if len(dfs) == 0 or dfs[0] is None:
        return
    df = dfs[0]  # single-run plot
    comp_cols = [c for c in ["val_approach_frac_mean", "val_stagnation_steps_mean",
                              "val_action_smoothness_mean", "val_door_angle_max_mean"] if c in df.columns]
    if not comp_cols:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    _setup_ax(ax, "Key Anti-Hacking Metrics over Training", "Timesteps", "Value")
    comp_labels = {
        "val_approach_frac_mean": "Approach fraction",
        "val_stagnation_steps_mean": "Stagnation steps",
        "val_action_smoothness_mean": "Action smoothness",
        "val_door_angle_max_mean": "Door angle max",
    }
    for col, color in zip(comp_cols, PALETTE):
        y = _smooth(df[col], smooth).values
        ax.plot(df["step"].values, y, label=comp_labels.get(col, col), color=color, linewidth=1.8)
    ax.legend(fontsize=8, framealpha=0.8)
    fig.tight_layout()
    fig.savefig(out_dir / "reward_components.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'reward_components.png'}")


def plot_summary(dfs, labels, out_dir, smooth=3):
    """2×3 summary grid of the 6 most important metrics."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    panels = [
        ("val_success_rate",          "Success Rate",              "Rate [0,1]",  0, 1),
        ("val_return_mean",            "Episode Return (mean)",     "Return",      None, None),
        ("val_door_angle_final_mean",  "Door Angle (final)",        "Norm. [0,1]", 0, 1),
        ("val_door_angle_max_mean",    "Door Angle (max per ep.)",  "Norm. [0,1]", 0, 1),
        ("val_approach_frac_mean",     "Approach Fraction",         "Fraction",    0, 1),
        ("val_stagnation_steps_mean",  "Stagnation Steps / Ep.",    "Steps",       None, None),
    ]

    for ax, (col, title, ylabel, ymin, ymax) in zip(axes, panels):
        _plot_single(ax, dfs, labels, col, title, ylabel, smooth, ymin, ymax)
        if col == "val_success_rate":
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
            ax.axhline(0.9, color="green", linestyle=":", linewidth=1, alpha=0.6)
        if col == "val_approach_frac_mean":
            ax.axhline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.6)

    if len(dfs) > 1:
        handles, lbls = axes[0].get_legend_handles_labels()
        fig.legend(handles, lbls, loc="lower center", ncol=min(6, len(labels)),
                   fontsize=8, framealpha=0.8)

    fig.suptitle("Training Diagnostic Summary — OpenCabinet SAC", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    fig.savefig(out_dir / "summary.png", dpi=150)
    plt.close(fig)
    print(f"  saved: {out_dir / 'summary.png'}")


def main():
    parser = argparse.ArgumentParser(description="Plot training metrics from validation_curve.csv")
    parser.add_argument("--run", nargs="+", required=True,
                        help="One or more run output directories containing validation_curve.csv")
    parser.add_argument("--label", nargs="*", default=None,
                        help="Labels for each run (defaults to directory name)")
    parser.add_argument("--out", default="outputs/plots",
                        help="Output directory for PNG files")
    parser.add_argument("--smooth", type=int, default=3,
                        help="Rolling average window size (default: 3)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [Path(r) for r in args.run]
    labels = args.label if args.label else [d.name for d in run_dirs]

    if len(labels) < len(run_dirs):
        labels += [run_dirs[i].name for i in range(len(labels), len(run_dirs))]

    dfs = []
    valid_labels = []
    for rd, lbl in zip(run_dirs, labels):
        df = _load_validation_curve(rd)
        if df is None:
            print(f"  [warn] no validation_curve.csv found in {rd}")
            continue
        dfs.append(df)
        valid_labels.append(lbl)

    if not dfs:
        print("No valid runs found. Exiting.")
        sys.exit(1)

    print(f"Generating plots from {len(dfs)} run(s) → {out_dir}/")
    plot_success_rate(dfs, valid_labels, out_dir, args.smooth)
    plot_return(dfs, valid_labels, out_dir, args.smooth)
    plot_door_angle(dfs, valid_labels, out_dir, args.smooth)
    plot_anti_hacking(dfs, valid_labels, out_dir, args.smooth)
    plot_reward_components(dfs, valid_labels, out_dir, args.smooth)
    plot_summary(dfs, valid_labels, out_dir, args.smooth)
    print("Done.")


if __name__ == "__main__":
    main()
