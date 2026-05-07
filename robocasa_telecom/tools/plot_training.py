
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

                                                                             
ALGO_COLORS = {
    "PPO": "#2196F3",         
    "SAC": "#F44336",        
    "A2C": "#4CAF50",          
    "UNKNOWN": "#9E9E9E",
}
ALGO_MARKERS = {"PPO": "o", "SAC": "s", "A2C": "^", "UNKNOWN": "D"}


def _load_runs(runs_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        csv_path = run_dir / "training_curve.csv"
        summary_path = run_dir / "train_summary.json"
        if not csv_path.exists():
            continue

        episodes: list[dict] = []
        with csv_path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    episodes.append({
                        "episode": float(row["episode"]),
                        "reward":  float(row["reward"]),
                        "length":  float(row["length"]),
                        "time":    float(row["time"]),
                    })
                except (KeyError, ValueError):
                    continue

        summary: dict = {}
        if summary_path.exists():
            with summary_path.open(encoding="utf-8") as f:
                summary = json.load(f)

        algo = summary.get("algorithm", "UNKNOWN").upper()
        runs.append({
            "run_id":    run_dir.name,
            "algorithm": algo,
            "episodes":  episodes,
            "summary":   summary,
        })
    return runs


def _smooth(values: list[float], window: int) -> np.ndarray:
    arr = np.array(values, dtype=float)
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window) / window
    padded = np.pad(arr, (window // 2, window // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")[: len(arr)]


def _cumsteps(episodes: list[dict]) -> np.ndarray:
    lengths = np.array([e["length"] for e in episodes], dtype=float)
    return np.cumsum(lengths)


def _read_tb(tb_dir: Path, tag: str = "rollout/ep_rew_mean") -> list[dict]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return []

    algo_series: dict[str, tuple[list, list]] = {}
    for event_dir in sorted(tb_dir.iterdir()):
        if not event_dir.is_dir():
            continue
                                                           
        folder = event_dir.name.upper()
        algo = "UNKNOWN"
        for a in ("PPO", "SAC", "A2C"):
            if a in folder:
                algo = a
                break

        ea = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        try:
            ea.Reload()
        except Exception:
            continue
        if tag not in ea.Tags().get("scalars", []):
            continue

        events = ea.Scalars(tag)
        steps  = [e.step  for e in events]
        values = [e.value for e in events]

        if algo not in algo_series:
            algo_series[algo] = ([], [])
        algo_series[algo][0].extend(steps)
        algo_series[algo][1].extend(values)

    result = []
    for algo, (steps, values) in algo_series.items():
        order = np.argsort(steps)
        result.append({
            "algo":   algo,
            "steps":  np.array(steps)[order],
            "values": np.array(values)[order],
        })
    return result


def _apply_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)


def _millify(x: float, _pos=None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x/1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x/1_000:.0f}k"
    return str(int(x))


def plot_training_reward(runs: list[dict], smooth: int, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(9, 5))

    algo_runs: dict[str, list[dict]] = {}
    for r in runs:
        algo_runs.setdefault(r["algorithm"], []).append(r)

    for algo, group in sorted(algo_runs.items()):
        color = ALGO_COLORS.get(algo, ALGO_COLORS["UNKNOWN"])
        for i, run in enumerate(group):
            eps = run["episodes"]
            if not eps:
                continue
            xs = _cumsteps(eps)
            ys = [e["reward"] for e in eps]
            smoothed = _smooth(ys, smooth)
            label = algo if i == 0 else None
                         
            ax.plot(xs, ys, color=color, alpha=0.12, linewidth=0.8)
                             
            ax.plot(xs, smoothed, color=color, linewidth=2.2,
                    label=label, solid_capstyle="round")

    ax.xaxis.set_major_formatter(FuncFormatter(_millify))
    _apply_style(ax,
                 f"Episode Reward vs Timesteps  (smoothing window = {smooth})",
                 "Cumulative timesteps", "Episode reward")
    ax.legend(fontsize=10, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_episode_length(runs: list[dict], smooth: int, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(9, 4))

    seen_algos: set[str] = set()
    for run in runs:
        eps = run["episodes"]
        if not eps:
            continue
        algo  = run["algorithm"]
        color = ALGO_COLORS.get(algo, ALGO_COLORS["UNKNOWN"])
        xs = _cumsteps(eps)
        ys = [e["length"] for e in eps]
        label = algo if algo not in seen_algos else None
        seen_algos.add(algo)
        ax.plot(xs, _smooth(ys, smooth), color=color, linewidth=1.8,
                label=label, alpha=0.85)

    horizon = max(
        (e["length"] for r in runs for e in r["episodes"]),
        default=400,
    )
    ax.axhline(horizon, color="gray", linestyle="--", linewidth=1, label=f"Horizon ({int(horizon)})")
    ax.xaxis.set_major_formatter(FuncFormatter(_millify))
    _apply_style(ax, "Episode Length over Training", "Cumulative timesteps", "Steps per episode")
    ax.legend(fontsize=10, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_eval_metrics(runs: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt

                                                                    
    best: dict[str, dict] = {}
    for run in runs:
        s = run["summary"]
        if not s:
            continue
        algo = run["algorithm"]
        ret = s.get("eval_return_mean", 0.0)
        if algo not in best or ret > best[algo]["summary"].get("eval_return_mean", -1e9):
            best[algo] = run

    if not best:
        print("  No summary data — skipping eval_metrics.png")
        return

    algos = sorted(best.keys())
    metrics_keys = [
        ("eval_return_mean",    "Mean return"),
        ("eval_max_open_mean",  "Max door open (×10)"),
        ("eval_min_dist_mean",  "Min dist to handle"),
        ("eval_success_rate",   "Success rate"),
    ]

                                                      
    present = [
        (k, label)
        for k, label in metrics_keys
        if any(k in best[a]["summary"] for a in algos)
    ]

    n_metrics = len(present)
    n_algos   = len(algos)
    bar_w = 0.7 / n_algos
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5), sharey=False)
    if n_metrics == 1:
        axes = [axes]

    for col, (key, label) in enumerate(present):
        ax = axes[col]
        for i, algo in enumerate(algos):
            s = best[algo]["summary"]
            val = s.get(key, 0.0)
                                                                        
            if key == "eval_max_open_mean":
                val *= 10
            color = ALGO_COLORS.get(algo, ALGO_COLORS["UNKNOWN"])
            x = i * bar_w
            bar = ax.bar(x, val, width=bar_w * 0.85, color=color, label=algo, alpha=0.88)
            ax.bar_label(bar, labels=[f"{val:.2f}"], padding=2, fontsize=9)

        ax.set_xticks([i * bar_w for i in range(n_algos)])
        ax.set_xticklabels(algos, fontsize=10)
        _apply_style(ax, label, "Algorithm", "")
        ax.set_xlim(-bar_w, n_algos * bar_w)

    fig.suptitle("Final Evaluation Metrics by Algorithm  (best run per algo)",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_tensorboard(tb_dir: Path, smooth: int, out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    series = _read_tb(tb_dir)
    if not series:
        print("  No TensorBoard data found (or tensorboard package not installed) — skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    for s in series:
        algo  = s["algo"]
        color = ALGO_COLORS.get(algo, ALGO_COLORS["UNKNOWN"])
        raw   = s["values"]
        smoothed = _smooth(raw.tolist(), smooth)
        ax.plot(s["steps"], raw, color=color, alpha=0.15, linewidth=0.7)
        ax.plot(s["steps"], smoothed, color=color, linewidth=2.2, label=algo)

    ax.xaxis.set_major_formatter(FuncFormatter(_millify))
    _apply_style(ax,
                 f"TensorBoard — rollout/ep_rew_mean  (window={smooth})",
                 "Timesteps", "Mean episode reward (rollout)")
    ax.legend(fontsize=10, framealpha=0.7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate training plots for the report")
    p.add_argument("--runs-dir",    default="outputs",           help="Directory containing run subdirectories")
    p.add_argument("--tb-dir",      default="logs/tensorboard",  help="TensorBoard log root directory")
    p.add_argument("--output-dir",  default="figures",           help="Directory where PNGs are saved")
    p.add_argument("--smooth",      type=int, default=40,        help="Rolling-mean window (episodes)")
    p.add_argument("--algo-filter", nargs="*", default=None,
                   help="Restrict to these algorithms, e.g. --algo-filter PPO SAC")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    tb_dir   = Path(args.tb_dir)
    out_dir  = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = _load_runs(runs_dir)
    if not runs:
        print(f"[plot_training] No runs found under {runs_dir}")
        return

    if args.algo_filter:
        filt = {a.upper() for a in args.algo_filter}
        runs = [r for r in runs if r["algorithm"] in filt]

    total_eps = sum(len(r["episodes"]) for r in runs)
    print(f"[plot_training] Found {len(runs)} runs, {total_eps} total episodes")
    print(f"[plot_training] Algorithms: {sorted({r['algorithm'] for r in runs})}")
    print(f"[plot_training] Output dir: {out_dir.resolve()}")
    print()

    plot_training_reward(runs, args.smooth, out_dir / "training_reward.png")
    plot_episode_length(runs, args.smooth,  out_dir / "episode_length.png")
    plot_eval_metrics(runs,                 out_dir / "eval_metrics.png")

    if tb_dir.exists():
        plot_tensorboard(tb_dir, args.smooth, out_dir / "tensorboard.png")
    else:
        print(f"  TensorBoard dir not found at {tb_dir} — skipping")

    print(f"\n[plot_training] Done. Figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
