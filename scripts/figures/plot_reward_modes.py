#!/usr/bin/env python3
"""Reward-mode dashboard for Slice 1 of the stable training plan.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md \u00a76 / Slice 1.

For each v01 task we score a "good" response (the source text itself, which
preserves required facts and stays terse) and a "bad" response (a
boilerplate AI-style reply that drops facts and triggers wrapper / refusal
penalties). We then compute all three reward modes plus the full ridge
8-dim rubric and deterministic component breakdown.

Outputs (PNG + PDF) in ``outputs/figures/reward_modes/``:

  * ``reward_modes_dashboard.{png,pdf}`` \u2014 six-panel master figure.
  * ``reward_modes_summary.json`` \u2014 numeric summary backing the figure.

Run::

  rtk uv run python scripts/figures/plot_reward_modes.py
"""

from __future__ import annotations

import json
import statistics
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

from humanize_rl.reward.grpo_rewards import (
    RewardModeConfig,
    _softened_scalar_from_result,
    _strict_scalar_from_result,
    risk_compliance,
)
from humanize_rl.reward.reward import (
    RIDGE_RUBRIC_DIMS,
    RewardResult,
    load_ridge_scorer,
    score_response,
)
from humanize_rl.reward.tasks import RLTask, load_tasks

# ---------------------------------------------------------------------------
# Plot styling \u2014 shared with paper figures.
# ---------------------------------------------------------------------------

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "Helvetica"]
plt.rcParams["axes.edgecolor"] = "#cbd5e1"
plt.rcParams["axes.linewidth"] = 1.0
plt.rcParams["grid.color"] = "#e2e8f0"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.6

COLOR_TEXT = "#1e293b"
COLOR_GOOD = "#059669"   # emerald
COLOR_BAD = "#e11d48"    # rose

MODE_COLORS = {
    "current_components": "#6366f1",  # indigo
    "scalar_current":     "#0ea5e9",  # sky
    "scalar_softened":    "#f59e0b",  # amber
}

DETERMINISTIC_DIMS = [
    "faithfulness",
    "task_following",
    "length",
    "format",
    "placeholder",
    "clarity",
    "no_corporate_filler",
]


# ---------------------------------------------------------------------------
# Data assembly
# ---------------------------------------------------------------------------


@dataclass
class ScoredRow:
    task_id: str
    family: str
    label: str  # "good" or "bad"
    result: RewardResult
    scalar_current: float
    scalar_softened: float


def _good_response(task: RLTask) -> str:
    """Faithful echo: preserves required facts and respects length."""
    return task.input_text


def _bad_response(task: RLTask) -> str:
    """Boilerplate AI reply: trips wrapper, drops facts, vague filler."""
    return (
        "Here's a polished version: I wanted to circle back and let you know "
        "that we will leverage our robust process to seamlessly unlock value "
        "across the organization. Let me know if you have any questions!"
    )


def _score(
    task: RLTask, response: str, label: str, scorer, soft_cfg: RewardModeConfig
) -> ScoredRow:
    result = score_response(task, response, scorer)
    return ScoredRow(
        task_id=task.id,
        family=task.family,
        label=label,
        result=result,
        scalar_current=_strict_scalar_from_result(result),
        scalar_softened=_softened_scalar_from_result(result, soft_cfg),
    )


def _collect(task_path: Path, limit: int | None) -> list[ScoredRow]:
    scorer = load_ridge_scorer()
    if scorer is None:
        raise SystemExit(
            "No ridge scorer pkl found. Expected models/track_a_10k/ridge.pkl "
            "or models/distilled/baseline_ridge.pkl."
        )
    soft_cfg = RewardModeConfig(mode="scalar_softened")
    tasks = list(load_tasks(task_path))
    if limit is not None:
        tasks = tasks[:limit]
    rows: list[ScoredRow] = []
    for task in tasks:
        rows.append(_score(task, _good_response(task), "good", scorer, soft_cfg))
        rows.append(_score(task, _bad_response(task), "bad", scorer, soft_cfg))
    return rows


# ---------------------------------------------------------------------------
# Panel helpers
# ---------------------------------------------------------------------------


def _mode_values(rows: list[ScoredRow], mode: str) -> list[float]:
    if mode == "current_components":
        return [
            r.result.weighted_components.get("ridge_rubric", 0.0)
            + r.result.weighted_components.get("deterministic", 0.0)
            + sum(r.result.penalties.values())
            for r in rows
        ]
    if mode == "scalar_current":
        return [r.scalar_current for r in rows]
    return [r.scalar_softened for r in rows]


def _panel_reward_distribution(ax, rows: list[ScoredRow]) -> None:
    bins = np.linspace(-1.0, 1.05, 40)
    for mode, color in MODE_COLORS.items():
        ax.hist(
            _mode_values(rows, mode),
            bins=bins,
            alpha=0.45,
            color=color,
            label=mode,
            edgecolor="none",
        )
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax.set_title("Reward distribution per mode", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_xlabel("reward", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("count", fontsize=9, color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.5)


def _panel_ridge_radar(ax, rows: list[ScoredRow]) -> None:
    dims = RIDGE_RUBRIC_DIMS
    angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
    angles += angles[:1]

    for label, color in (("good", COLOR_GOOD), ("bad", COLOR_BAD)):
        subset = [r for r in rows if r.label == label]
        means: list[float] = []
        for dim in dims:
            vals = [r.result.weighted_components.get(f"ridge_{dim}", 0.0) for r in subset]
            means.append(statistics.fmean(vals) if vals else 0.0)
        means += means[:1]
        ax.plot(angles, means, color=color, linewidth=1.6, label=label)
        ax.fill(angles, means, color=color, alpha=0.18)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([d.replace("_", "\n") for d in dims], fontsize=7, color=COLOR_TEXT)
    ax.set_ylim(0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=7)
    ax.set_title("Ridge rubric \u2014 8 dims (mean)", fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=18)
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.15, -0.05))


def _panel_deterministic_bar(ax, rows: list[ScoredRow]) -> None:
    width = 0.38
    x = np.arange(len(DETERMINISTIC_DIMS))
    for offset, label, color in (
        (-width / 2, "good", COLOR_GOOD),
        (+width / 2, "bad", COLOR_BAD),
    ):
        subset = [r for r in rows if r.label == label]
        means = [
            statistics.fmean([r.result.components.get(d, 0.0) for r in subset]) if subset else 0.0
            for d in DETERMINISTIC_DIMS
        ]
        ax.bar(x + offset, means, width=width, color=color, label=label, edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(DETERMINISTIC_DIMS, rotation=30, ha="right", fontsize=8, color=COLOR_TEXT)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("component score", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Deterministic components (mean)", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _panel_penalty_hist(ax, rows: list[ScoredRow]) -> None:
    good_pen = [sum(r.result.penalties.values()) for r in rows if r.label == "good"]
    bad_pen = [sum(r.result.penalties.values()) for r in rows if r.label == "bad"]
    bins = np.linspace(-2.0, 0.05, 30)
    ax.hist(good_pen, bins=bins, alpha=0.55, color=COLOR_GOOD, label="good", edgecolor="none")
    ax.hist(bad_pen, bins=bins, alpha=0.55, color=COLOR_BAD, label="bad", edgecolor="none")
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=0.8)
    ax.set_title("Total penalty per response", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.set_xlabel("sum(penalties)", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("count", fontsize=9, color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=0.5)


def _panel_ridge_vs_det_scatter(ax, rows: list[ScoredRow]) -> None:
    """2D rubric chart: ridge_rubric (x) vs deterministic (y), per response.

    Color = reward mode value (softened). Marker = good/bad.
    """
    xs, ys, cs, ms = [], [], [], []
    for r in rows:
        xs.append(r.result.weighted_components.get("ridge_rubric", 0.0))
        ys.append(r.result.weighted_components.get("deterministic", 0.0))
        cs.append(r.scalar_softened)
        ms.append("o" if r.label == "good" else "x")

    # Two separate scatter calls so legend markers render correctly.
    for marker, label, edge in (("o", "good", "#065f46"), ("x", "bad", "#9f1239")):
        idx = [i for i, m in enumerate(ms) if m == marker]
        if not idx:
            continue
        sc = ax.scatter(
            [xs[i] for i in idx],
            [ys[i] for i in idx],
            c=[cs[i] for i in idx],
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            marker=marker,
            s=42,
            alpha=0.85,
            edgecolors=edge,
            linewidths=0.6,
            label=label,
        )
        last_sc = sc
    cbar = plt.colorbar(last_sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("scalar_softened", fontsize=8, color=COLOR_TEXT)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlim(-0.05, max(0.55, max(xs) + 0.05))
    ax.set_ylim(-0.05, max(0.55, max(ys) + 0.05))
    ax.set_xlabel("ridge_rubric weighted", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("deterministic weighted", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Rubric 2D \u2014 ridge vs deterministic", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.grid(alpha=0.5)


def _panel_risk_compliance_curve(ax, rows: list[ScoredRow], penalty_cap: float) -> None:
    """Show how risk_compliance smooths penalties vs the raw sum-of-penalties signal."""
    xs = np.linspace(-1.5, 0.0, 200)
    raw = np.clip(xs, -1.0, 1.0)
    smoothed = np.array([risk_compliance(x, penalty_cap=penalty_cap) for x in xs])
    ax.plot(xs, raw, color="#94a3b8", linewidth=1.4, linestyle="--", label="raw clip(sum, -1, 1)")
    ax.plot(xs, smoothed, color="#f59e0b", linewidth=1.8, label=f"risk_compliance (cap={penalty_cap})")

    # Overlay actual penalty totals for context.
    pens = [sum(r.result.penalties.values()) for r in rows]
    comps = [risk_compliance(p, penalty_cap=penalty_cap) for p in pens]
    ax.scatter(pens, comps, c="#7c3aed", s=12, alpha=0.45, label="observed")
    ax.axhline(0, color="#cbd5e1", linewidth=0.6)
    ax.axvline(0, color="#cbd5e1", linewidth=0.6)
    ax.set_xlabel("sum(penalties)", fontsize=9, color=COLOR_TEXT)
    ax.set_ylabel("signal value", fontsize=9, color=COLOR_TEXT)
    ax.set_title("Penalty shaping: raw vs softened", fontsize=11, fontweight="bold", color=COLOR_TEXT)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(alpha=0.5)


# ---------------------------------------------------------------------------
# Summary JSON
# ---------------------------------------------------------------------------


def _summary(rows: list[ScoredRow]) -> dict:
    def stats(values: Iterable[float]) -> dict:
        vals = list(values)
        if not vals:
            return {"n": 0}
        return {
            "n": len(vals),
            "mean": statistics.fmean(vals),
            "std": statistics.pstdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals),
            "max": max(vals),
        }

    by_mode = {}
    for mode in MODE_COLORS:
        by_mode[mode] = {
            "all": stats(_mode_values(rows, mode)),
            "good": stats([v for r, v in zip(rows, _mode_values(rows, mode), strict=True) if r.label == "good"]),
            "bad": stats([v for r, v in zip(rows, _mode_values(rows, mode), strict=True) if r.label == "bad"]),
        }
    return {
        "n_tasks": len({r.task_id for r in rows}),
        "n_responses": len(rows),
        "by_mode": by_mode,
        "ridge_dims": RIDGE_RUBRIC_DIMS,
        "deterministic_dims": DETERMINISTIC_DIMS,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@click.command()
@click.option(
    "--task-path",
    default="data/rl/humanize_tasks_v01_smoke.jsonl",
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    default="outputs/figures/reward_modes",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option("--limit", type=int, default=None, help="Limit tasks (debug).")
@click.option("--penalty-cap", type=float, default=1.0, show_default=True)
def main(task_path: Path, output_dir: Path, limit: int | None, penalty_cap: float) -> None:
    """Generate the reward-mode dashboard + summary JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect(task_path, limit=limit)
    click.echo(f"Scored {len(rows)} responses across {len({r.task_id for r in rows})} tasks.")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        2, 3,
        width_ratios=[1, 1, 1.05],
        height_ratios=[1, 1],
        hspace=0.45, wspace=0.32,
    )
    _panel_reward_distribution(fig.add_subplot(gs[0, 0]), rows)
    _panel_ridge_radar(fig.add_subplot(gs[0, 1], polar=True), rows)
    _panel_deterministic_bar(fig.add_subplot(gs[0, 2]), rows)
    _panel_penalty_hist(fig.add_subplot(gs[1, 0]), rows)
    _panel_ridge_vs_det_scatter(fig.add_subplot(gs[1, 1]), rows)
    _panel_risk_compliance_curve(fig.add_subplot(gs[1, 2]), rows, penalty_cap)

    fig.suptitle(
        "Reward-mode dashboard \u2014 v01 tasks, good vs bad responses",
        fontsize=14, fontweight="bold", color=COLOR_TEXT, y=0.995,
    )

    png = output_dir / "reward_modes_dashboard.png"
    pdf = output_dir / "reward_modes_dashboard.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "reward_modes_summary.json"
    summary_path.write_text(json.dumps(_summary(rows), indent=2))

    click.echo(f"Wrote {png}")
    click.echo(f"Wrote {pdf}")
    click.echo(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
