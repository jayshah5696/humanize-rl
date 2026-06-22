#!/usr/bin/env python3
"""Visualize Slice 4 difficulty scoring output.

Reads ``task_difficulty.jsonl`` from a difficulty-scoring run and produces
a 4-panel diagnostic figure:

  1. Bucket distribution (bar)
  2. reward_mean vs reward_std scatter, colored by bucket
  3. Per-family bucket stacked bar
  4. penalty_rate distribution per bucket (box-ish)

Run::

  uv run python scripts/figures/plot_task_difficulty.py \\
    --input outputs/rl_difficulty/mix_v1/task_difficulty.jsonl
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
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
    "useful": "#059669",  # emerald
    "too_easy": "#0ea5e9",  # sky
    "too_hard": "#e11d48",  # rose
    "dead": "#7c3aed",  # violet
    "clipped": "#f59e0b",  # amber
    "same_pattern": "#6b7280",  # slate
}
BUCKET_ORDER = ["useful", "too_easy", "too_hard", "dead", "clipped", "same_pattern"]
COLOR_TEXT = "#1e293b"


def _panel_bucket_bar(ax, rows: list[dict]) -> None:
    counts = Counter(r["bucket"] for r in rows)
    buckets = [b for b in BUCKET_ORDER if counts[b] > 0]
    values = [counts[b] for b in buckets]
    colors = [BUCKET_COLORS[b] for b in buckets]
    bars = ax.bar(buckets, values, color=colors, edgecolor="none")
    total = sum(values)
    for bar, v in zip(bars, values, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.01,
            f"{v}\n({v / total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="semibold",
            color=COLOR_TEXT,
        )
    ax.set_title(
        f"Bucket counts (n={total})", fontsize=11, fontweight="bold", color=COLOR_TEXT
    )
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_ylabel("tasks", fontsize=9, color=COLOR_TEXT)
    ax.tick_params(axis="x", labelsize=8.5, rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.5)


def _panel_reward_scatter(ax, rows: list[dict]) -> None:
    for bucket in BUCKET_ORDER:
        sub = [r for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        ax.scatter(
            [r["reward_mean"] for r in sub],
            [r["reward_std"] for r in sub],
            c=BUCKET_COLORS[bucket],
            label=f"{bucket} ({len(sub)})",
            s=22,
            alpha=0.7,
            edgecolors="white",
            linewidths=0.4,
        )
    # Useful-band guides.
    ax.axvspan(0.20, 0.80, alpha=0.05, color="#059669")
    ax.axhline(
        0.03,
        color="#94a3b8",
        linestyle="--",
        linewidth=0.7,
        label="min reward_std=0.03",
    )
    ax.axvline(0.20, color="#94a3b8", linestyle=":", linewidth=0.7)
    ax.axvline(0.80, color="#94a3b8", linestyle=":", linewidth=0.7)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.005, max(0.4, max(r["reward_std"] for r in rows) + 0.02))
    ax.set_xlabel("reward_mean", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("reward_std", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Reward mean vs std", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.grid(alpha=0.5)


def _panel_family_stacked(ax, rows: list[dict]) -> None:
    by_family: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_family[r["family"]][r["bucket"]] += 1
    families = sorted(by_family, key=lambda f: -sum(by_family[f].values()))
    bottoms = np.zeros(len(families))
    for bucket in BUCKET_ORDER:
        values = np.array([by_family[f][bucket] for f in families])
        if not values.any():
            continue
        ax.bar(
            families,
            values,
            bottom=bottoms,
            color=BUCKET_COLORS[bucket],
            label=bucket,
            edgecolor="none",
        )
        bottoms = bottoms + values
    ax.set_title("Bucket × family", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_ylabel("tasks", fontsize=9, color=COLOR_TEXT)
    ax.tick_params(axis="x", labelsize=7.5, rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    ax.legend(fontsize=7, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.5)


def _panel_penalty_rate(ax, rows: list[dict]) -> None:
    data = []
    labels = []
    colors = []
    for bucket in BUCKET_ORDER:
        sub = [r["penalty_rate"] for r in rows if r["bucket"] == bucket]
        if not sub:
            continue
        data.append(sub)
        labels.append(bucket)
        colors.append(BUCKET_COLORS[bucket])
    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, widths=0.55, showfliers=False
    )
    for patch, color in zip(bp["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
        patch.set_edgecolor(COLOR_TEXT)
    for median in bp["medians"]:
        median.set_color(COLOR_TEXT)
        median.set_linewidth(1.4)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("penalty_rate", fontsize=9, color=COLOR_TEXT)
    ax.set_title(
        "Penalty rate per bucket", fontsize=11, fontweight="bold", color=COLOR_TEXT
    )
    ax.tick_params(axis="x", labelsize=8.5, rotation=20)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.5)


@click.command()
@click.option(
    "--input",
    "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    default="outputs/figures/task_difficulty",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
def main(input_path: Path, output_dir: Path) -> None:
    """Plot Slice 4 difficulty scoring dashboard."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    click.echo(f"Loaded {len(rows)} task-difficulty rows from {input_path}")

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.28)
    _panel_bucket_bar(fig.add_subplot(gs[0, 0]), rows)
    _panel_reward_scatter(fig.add_subplot(gs[0, 1]), rows)
    _panel_family_stacked(fig.add_subplot(gs[1, 0]), rows)
    _panel_penalty_rate(fig.add_subplot(gs[1, 1]), rows)
    fig.suptitle(
        f"Task difficulty dashboard — {input_path.parent.name}",
        fontsize=14,
        fontweight="bold",
        color=COLOR_TEXT,
        y=0.995,
    )

    png = output_dir / "task_difficulty_dashboard.png"
    pdf = output_dir / "task_difficulty_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
