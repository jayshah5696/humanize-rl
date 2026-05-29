#!/usr/bin/env python3
"""Compare difficulty bucketing across reward modes (Slice 4 follow-up).

Reads ``rebucket_summary.json`` + the per-mode ``rebucket_<mode>.jsonl``
files written by ``rebucket_difficulty.py`` and produces a 4-panel
dashboard:

  1. Stacked bar of bucket counts per mode
  2. Reward-mean vs reward-std scatter for the strict reward (baseline)
  3. Reward-mean vs reward-std scatter for ``scalar_softened_permissive``
  4. Per-task reward shift: strict vs softened-permissive (scatter)

Run::

  rtk uv run python scripts/figures/plot_rebucket_comparison.py \\
    --input-dir outputs/rl_difficulty/mix_v1
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

BUCKET_COLORS = {
    "useful":       "#059669",
    "too_easy":     "#0ea5e9",
    "too_hard":     "#e11d48",
    "dead":         "#7c3aed",
    "clipped":      "#f59e0b",
    "same_pattern": "#6b7280",
}
BUCKET_ORDER = ["useful", "too_easy", "too_hard", "dead", "clipped", "same_pattern"]
MODE_LABELS = {
    "current_components":          "current\n(3-func)",
    "scalar_current":              "scalar\n(strict)",
    "scalar_softened":             "softened\n(default)",
    "scalar_softened_permissive":  "softened\n(permissive)",
}
COLOR_TEXT = "#1e293b"


def _panel_stacked_buckets(ax, summary: dict) -> None:
    modes = [m for m in MODE_LABELS if m in summary]
    bottoms = np.zeros(len(modes))
    for bucket in BUCKET_ORDER:
        values = np.array([summary[m]["bucket_counts"].get(bucket, 0) for m in modes])
        if not values.any():
            continue
        ax.bar(
            [MODE_LABELS[m] for m in modes],
            values, bottom=bottoms,
            color=BUCKET_COLORS[bucket], label=bucket, edgecolor="none",
        )
        bottoms = bottoms + values
    # Annotate kept count on top
    for i, m in enumerate(modes):
        kept = summary[m]["bucket_counts"].get("useful", 0)
        total = sum(summary[m]["bucket_counts"].values())
        ax.text(
            i, bottoms[i] + total * 0.015,
            f"useful={kept} ({kept/total*100:.1f}%)",
            ha="center", va="bottom", fontsize=9, fontweight="bold", color="#065f46",
        )
    ax.set_ylim(0, max(bottoms) * 1.15)
    ax.set_ylabel("tasks", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Bucket distribution per reward mode", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.5)
    ax.tick_params(axis="x", labelsize=8.5)


def _scatter(ax, rows: list[dict], title: str, xlim: tuple[float, float], y_max: float) -> None:
    for bucket in BUCKET_ORDER:
        sub = [r for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        ax.scatter(
            [r["reward_mean"] for r in sub],
            [r["reward_std"] for r in sub],
            c=BUCKET_COLORS[bucket],
            label=f"{bucket} ({len(sub)})",
            s=18, alpha=0.65, edgecolors="white", linewidths=0.3,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(-0.005, y_max)
    ax.set_xlabel("reward_mean", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("reward_std", fontsize=9, color=COLOR_TEXT)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.grid(alpha=0.5)


def _panel_strict_vs_soft(ax, strict_rows: list[dict], soft_rows: list[dict]) -> None:
    strict_map = {r["task_id"]: r for r in strict_rows}
    soft_map = {r["task_id"]: r for r in soft_rows}
    ids = sorted(set(strict_map) & set(soft_map))
    xs = [strict_map[i]["reward_mean"] for i in ids]
    ys = [soft_map[i]["reward_mean"] for i in ids]
    colors = [BUCKET_COLORS.get(soft_map[i]["bucket"], "#999") for i in ids]
    ax.scatter(xs, ys, c=colors, s=16, alpha=0.7, edgecolors="white", linewidths=0.3)
    lo = min(min(xs), min(ys)) - 0.05
    hi = max(max(xs), max(ys)) + 0.05
    ax.plot([lo, hi], [lo, hi], color="#94a3b8", linestyle="--", linewidth=0.8, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("reward_mean (current_components)", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("reward_mean (scalar_softened_permissive)", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Per-task reward shift: strict \u2192 softened", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.5)


@click.command()
@click.option(
    "--input-dir", required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--output-dir", default="outputs/figures/rebucket_comparison",
    show_default=True, type=click.Path(file_okay=False, path_type=Path),
)
def main(input_dir: Path, output_dir: Path) -> None:
    """Plot the rebucket comparison dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = json.loads((input_dir / "rebucket_summary.json").read_text())

    def _load(mode: str) -> list[dict]:
        path = input_dir / f"rebucket_{mode}.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

    strict_rows = _load("current_components")
    soft_perm_rows = _load("scalar_softened_permissive")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
    _panel_stacked_buckets(fig.add_subplot(gs[0, 0]), summary)
    _scatter(
        fig.add_subplot(gs[0, 1]),
        strict_rows,
        "Strict reward \u2014 reward mean vs std",
        xlim=(-1.05, 1.05),
        y_max=max(r["reward_std"] for r in strict_rows) + 0.02,
    )
    _scatter(
        fig.add_subplot(gs[1, 0]),
        soft_perm_rows,
        "Softened (permissive) \u2014 reward mean vs std",
        xlim=(0.0, 1.05),
        y_max=max(r["reward_std"] for r in soft_perm_rows) + 0.005,
    )
    _panel_strict_vs_soft(fig.add_subplot(gs[1, 1]), strict_rows, soft_perm_rows)

    fig.suptitle(
        "Difficulty rebucketing across reward modes \u2014 mix_v1 (506 tasks, K=8)",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, y=0.995,
    )

    png = output_dir / "rebucket_comparison.png"
    pdf = output_dir / "rebucket_comparison.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
