"""Compare training results across multiple runs / algorithms.

Usage
-----
# Compare all runs in the default outputs/ directory:
    python scripts/compare_runs.py

# Compare specific run directories:
    python scripts/compare_runs.py --runs outputs/OpenCabinet_ppo_seed0_* outputs/OpenCabinet_sac_seed0_*

# Export results to CSV:
    python scripts/compare_runs.py --csv results/comparison.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare training runs across algorithms")
    parser.add_argument(
        "--runs",
        nargs="*",
        default=None,
        help="Run directories to compare (default: all subdirs in outputs/)",
    )
    parser.add_argument(
        "--outputs-root",
        default="outputs",
        help="Root directory to scan for runs (default: outputs/)",
    )
    parser.add_argument(
        "--csv",
        default=None,
        help="Optional path to export comparison table as CSV",
    )
    return parser.parse_args()


def load_summary(run_dir: Path) -> dict | None:
    """Load train_summary.json from a run directory, return None if missing."""
    summary_path = run_dir / "train_summary.json"
    if not summary_path.exists():
        return None
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run_dir in run_dirs:
        summary = load_summary(run_dir)
        if summary is None:
            continue
        rows.append({
            "run_id": summary.get("run_id", run_dir.name),
            "algorithm": summary.get("algorithm", "?"),
            "task": summary.get("task", "?"),
            "seed": summary.get("seed", "?"),
            "total_timesteps": summary.get("total_timesteps", "?"),
            "success_rate": summary.get("eval_success_rate", summary.get("success_rate", "?")),
            "return_mean": round(float(summary.get("eval_return_mean", summary.get("return_mean", 0))), 3),
            "return_std": round(float(summary.get("eval_return_std", summary.get("return_std", 0))), 3),
        })
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No completed runs found.")
        return

    headers = ["algorithm", "seed", "success_rate", "return_mean", "return_std", "total_timesteps", "run_id"]
    col_widths = {h: max(len(h), max(len(str(r[h])) for r in rows)) for h in headers}

    header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
    sep_line = "  ".join("-" * col_widths[h] for h in headers)

    print("\n=== Training Run Comparison ===\n")
    print(header_line)
    print(sep_line)

    # Sort by algorithm then seed for readability.
    for row in sorted(rows, key=lambda r: (str(r["algorithm"]), str(r["seed"]))):
        print("  ".join(str(row[h]).ljust(col_widths[h]) for h in headers))

    print()
    _print_best(rows)


def _print_best(rows: list[dict]) -> None:
    valid = [r for r in rows if isinstance(r["success_rate"], (int, float))]
    if not valid:
        return
    best = max(valid, key=lambda r: float(r["success_rate"]))
    print(f"Best success rate: {best['algorithm']} seed={best['seed']} → {best['success_rate']:.1%}")


def export_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = ["algorithm", "seed", "success_rate", "return_mean", "return_std", "total_timesteps", "run_id"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison to: {path}")


def main() -> None:
    args = parse_args()

    if args.runs:
        run_dirs = [Path(p) for r in args.runs for p in Path(".").glob(r) if Path(p).is_dir()]
        if not run_dirs:
            # Fallback: treat each entry as a literal path
            run_dirs = [Path(r) for r in args.runs if Path(r).is_dir()]
    else:
        outputs_root = Path(args.outputs_root)
        if not outputs_root.exists():
            print(f"outputs directory not found: {outputs_root}", file=sys.stderr)
            sys.exit(1)
        run_dirs = [p for p in sorted(outputs_root.iterdir()) if p.is_dir() and p.name != "eval"]

    if not run_dirs:
        print("No run directories found. Train some models first.", file=sys.stderr)
        sys.exit(1)

    rows = collect_runs(run_dirs)
    print_table(rows)

    if args.csv:
        export_csv(rows, Path(args.csv))


if __name__ == "__main__":
    main()
