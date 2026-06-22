#!/usr/bin/env python3
"""Compare Slice 5 reward-mode ablations.

Reads ``outputs/ablations/<run>_summary.json`` and produces a 4-panel
dashboard:

  1. Reward (raw + EMA-20) over training steps, one line per ablation.
  2. Reward std over steps.
  3. diag/risk_compliance EMA over steps.
  4. diag/response_length_mean EMA over steps.

Run::

  uv run python scripts/figures/plot_ablations.py
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

COLOR_TEXT = "#1e293b"

ABLATION_GROUPS = {
    "reward": {
        "A0  current_components": (
            "outputs/ablations/gemma4-rl-ablation-a0-current-components_summary.json",
            "#6366f1",
        ),
        "A1  scalar_current": (
            "outputs/ablations/gemma4-rl-ablation-a1-scalar-current_summary.json",
            "#0ea5e9",
        ),
        "A2  scalar_softened": (
            "outputs/ablations/gemma4-rl-ablation-a2-scalar-softened_summary.json",
            "#f59e0b",
        ),
    },
    "stability": {
        "B0  G=8  acc=8  LR=2e-5": (
            "outputs/ablations/gemma4-rl-ablation-b0-g8-acc8-lr2e5_summary.json",
            "#6366f1",
        ),
        "B1  G=16 acc=16 LR=1e-5": (
            "outputs/ablations/gemma4-rl-ablation-b1-g16-acc16-lr1e5_summary.json",
            "#0ea5e9",
        ),
        "B2  G=16 acc=16 LR=2e-5": (
            "outputs/ablations/gemma4-rl-ablation-b2-g16-acc16-lr2e5_summary.json",
            "#f59e0b",
        ),
    },
    "scaling": {
        "C0  scale=group": (
            "outputs/ablations/gemma4-rl-ablation-c0-scale-group_summary.json",
            "#6366f1",
        ),
        "C1  scale=batch": (
            "outputs/ablations/gemma4-rl-ablation-c1-scale-batch_summary.json",
            "#0ea5e9",
        ),
        "C2  scale=none": (
            "outputs/ablations/gemma4-rl-ablation-c2-scale-none_summary.json",
            "#f59e0b",
        ),
    },
}
RUNS = ABLATION_GROUPS["reward"]  # default — overridden by --group below


def _series(hist: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for entry in hist:
        if key in entry and entry[key] is not None:
            xs.append(int(entry.get("step", len(xs) + 1)))
            ys.append(float(entry[key]))
    return xs, ys


def _panel(
    ax, runs: dict, key: str, title: str, ylabel: str, *, smooth_raw_too: bool = False
) -> None:
    for label, (path, color) in runs.items():
        data = json.loads(Path(path).read_text())
        hist = data.get("log_history", [])
        xs, ys = _series(hist, key)
        ema_xs, ema_ys = _series(hist, f"{key}/ema_20")
        if smooth_raw_too and xs:
            ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.9)
        if ema_xs:
            ax.plot(ema_xs, ema_ys, color=color, linewidth=1.8, label=label)
        elif xs:
            ax.plot(xs, ys, color=color, linewidth=1.6, label=label)
    ax.set_xlabel("step", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel(ylabel, fontsize=9, color=COLOR_TEXT)
    ax.set_title(title, fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False, loc="best")
    ax.grid(alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


@click.command()
@click.option(
    "--output-dir",
    default="outputs/figures/ablations",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--group",
    default="reward",
    show_default=True,
    type=click.Choice(list(ABLATION_GROUPS)),
    help="Which ablation group to plot.",
)
def main(output_dir: Path, group: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = ABLATION_GROUPS[group]
    titles = {
        "reward": "Slice 5 reward-mode ablations \u2014 50 steps, mid-band filtered mix",
        "stability": "Slice 6 group/batch/LR ablations \u2014 50 steps, softened reward",
        "scaling": "Slice 7 reward-scaling ablations \u2014 50 steps, B1 base config",
    }

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.45, wspace=0.30)
    _panel(
        fig.add_subplot(gs[0, 0]),
        runs,
        "reward",
        "Reward (EMA-20, raw shaded)",
        "reward",
        smooth_raw_too=True,
    )
    _panel(
        fig.add_subplot(gs[0, 1]),
        runs,
        "reward_std",
        "Reward std (EMA-20)",
        "reward_std",
    )
    _panel(
        fig.add_subplot(gs[1, 0]), runs, "grad_norm", "grad_norm (EMA-20)", "grad_norm"
    )
    _panel(
        fig.add_subplot(gs[1, 1]),
        runs,
        "completions/mean_length",
        "completions/mean_length (EMA-20)",
        "tokens",
    )

    fig.suptitle(
        titles[group], fontsize=14, fontweight="bold", color=COLOR_TEXT, y=0.995
    )

    png = output_dir / f"ablations_{group}_dashboard.png"
    pdf = output_dir / f"ablations_{group}_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {pdf}")


if __name__ == "__main__":
    main()
