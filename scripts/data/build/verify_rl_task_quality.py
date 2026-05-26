"""Dataset-level verification gate for humanize_tasks_v03 (plan §8).

Loads validated tasks + reference rollouts, computes 14 metrics across
Diversity / Hardness / Consistency, compares to hard + soft thresholds,
and emits a markdown + JSON report. Exits 1 if any HARD gate fails.

Usage:

    uv run scripts/data/build/verify_rl_task_quality.py \\
        --input data/rl/humanize_tasks_v03_filtered.jsonl \\
        --rollouts data/rl/v03_ref_rollouts.jsonl \\
        --report-out runs/v03/verification_report.md \\
        --json-out runs/v03/verification_report.json \\
        --thresholds configs/v03/verification_thresholds.yaml
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click
import yaml

from humanize_rl.reward.tasks import RLTask, load_tasks

# ----------------------------- IO helpers --------------------------------

def _load_rollouts(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return by_task
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        by_task[row["task_id"]].append(row)
    return by_task


def _word_count(text: str | None) -> int:
    return len((text or "").split())


def _populated_constraint_count(task: RLTask) -> int:
    """How many verifier-relevant constraint slots are filled?"""
    c = task.constraints
    count = 0
    optional_fields = [
        "target_words", "target_tolerance",
        "min_sentences", "max_sentences",
        "min_paragraphs", "max_paragraphs",
        "register_target", "max_passive_voice_pct",
        "min_contraction_count", "must_contain_number_kind",
        "max_words", "min_words", "exact_sentences",
    ]
    for field in optional_fields:
        if getattr(c, field) is not None:
            count += 1
    for field in [
        "required_section_headings",
        "must_include_phrases",
        "must_not_include_phrases",
        "forbidden_openers",
        "forbidden_phrases_global",
    ]:
        if getattr(c, field):
            count += 1
    for field in ["must_not_use_em_dash", "no_markdown"]:
        if getattr(c, field):
            count += 1
    return count


# ----------------------------- gate evaluators ---------------------------

def _gate(name: str, status: str, value: Any, hard: Any, soft: Any, note: str = "") -> dict:
    return {"gate": name, "status": status, "value": value,
            "hard": hard, "soft": soft, "note": note}


def _classify(value: float, hard_bound: float, soft_bound: float, direction: str) -> str:
    """direction: 'min' (value must be ≥ bound) or 'max' (value must be ≤ bound)."""
    if direction == "min":
        if value < hard_bound:
            return "FAIL"
        if value < soft_bound:
            return "WARN"
        return "PASS"
    if value > hard_bound:
        return "FAIL"
    if value > soft_bound:
        return "WARN"
    return "PASS"


def gate_A1(tasks: list[RLTask], t: dict) -> dict:
    lengths = sorted(_word_count(task.instruction) for task in tasks)
    n = len(lengths)
    if n == 0:
        return _gate("A1_instruction_length", "FAIL", {}, t["hard"], t["soft"], "no tasks")
    p10 = lengths[int(0.10 * (n - 1))]
    p50 = lengths[int(0.50 * (n - 1))]
    p90 = lengths[int(0.90 * (n - 1))]
    h = t["hard"]
    s = t["soft"]
    status = "PASS"
    if p90 < h["p90_min"] or p10 > h["p10_max"]:
        status = "FAIL"
    elif p50 < s["p50_min"] or p50 > s["p50_max"]:
        status = "WARN"
    return _gate("A1_instruction_length", status,
                 {"p10": p10, "p50": p50, "p90": p90}, h, s)


def gate_A2(tasks: list[RLTask], t: dict) -> dict:
    if not tasks:
        return _gate("A2_constraint_density", "FAIL", 0, t, t)
    mean_c = statistics.mean(_populated_constraint_count(task) for task in tasks)
    status = _classify(mean_c, t["hard_min"], t["soft_min"], "min")
    return _gate("A2_constraint_density", status, round(mean_c, 2),
                 t["hard_min"], t["soft_min"])


def gate_A3(tasks: list[RLTask], t: dict) -> dict:
    cells: Counter[tuple[str, str]] = Counter()
    for task in tasks:
        cells[(task.domain, task.register_)] += 1
    filled = sum(1 for v in cells.values() if v >= t["min_per_cell"])
    status = "PASS"
    if filled < t["hard_min_filled"]:
        status = "FAIL"
    elif filled < t["soft_min_filled"]:
        status = "WARN"
    return _gate(
        "A3_cell_coverage", status,
        {"filled": filled, "cells_total": t["cells_total"], "unique_cells": len(cells)},
        t["hard_min_filled"], t["soft_min_filled"],
    )


def gate_A4(tasks: list[RLTask], t: dict) -> dict:
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
    except ImportError:
        return _gate("A4_embedding_distance", "WARN", None,
                     t["hard_min"], t["soft_min"],
                     "sentence-transformers not installed; skipping")
    sample = tasks[: t["sample_size"]]
    if len(sample) < 5:
        return _gate("A4_embedding_distance", "WARN", None,
                     t["hard_min"], t["soft_min"], "too few tasks")
    model = SentenceTransformer(t["model"])
    embs = model.encode([task.instruction for task in sample], normalize_embeddings=True)
    sim = embs @ embs.T
    n = len(sample)
    iu = np.triu_indices(n, k=1)
    mean_dist = float(1.0 - sim[iu].mean())
    status = _classify(mean_dist, t["hard_min"], t["soft_min"], "min")
    return _gate("A4_embedding_distance", status, round(mean_dist, 3),
                 t["hard_min"], t["soft_min"])


def gate_A5(tasks: list[RLTask], t: dict) -> dict:
    counts: Counter[str] = Counter()
    for task in tasks:
        counts[task.task_author_model or "unknown"] += 1
    n = sum(counts.values()) or 1
    observed = {k: v / n for k, v in counts.items()}
    max_pp = 0.0
    for author, target in t["target_shares"].items():
        obs = observed.get(author, 0.0)
        pp = abs(obs - target) * 100
        max_pp = max(max_pp, pp)
    status = _classify(max_pp, t["hard_max_pp"], t["soft_max_pp"], "max")
    return _gate("A5_author_balance", status, round(max_pp, 2),
                 t["hard_max_pp"], t["soft_max_pp"],
                 note=f"observed={ {k: round(v,3) for k,v in observed.items()} }")


def gate_B1(rollouts: dict, t: dict) -> dict:
    strong = t["strong_model"]
    rewards = [r["reward"] for rs in rollouts.values() for r in rs
               if r.get("model") == strong and "reward" in r]
    if not rewards:
        return _gate("B1_strong_reward_band", "WARN", None,
                     t["hard"], t["soft"], "no strong-model rollouts")
    median = statistics.median(rewards)
    h, s = t["hard"], t["soft"]
    status = "PASS"
    if median < h["median_min"] or median > h["median_max"]:
        status = "FAIL"
    elif median < s["median_min"] or median > s["median_max"]:
        status = "WARN"
    return _gate("B1_strong_reward_band", status, round(median, 3), h, s)


def gate_B2(rollouts: dict, t: dict, strong_model: str) -> dict:
    strong = t.get("strong_model", strong_model)
    weak = t["weak_model"]
    strong_rewards = [r["reward"] for rs in rollouts.values() for r in rs
                      if r.get("model") == strong and "reward" in r]
    weak_rewards = [r["reward"] for rs in rollouts.values() for r in rs
                    if r.get("model") == weak and "reward" in r]
    if not strong_rewards or not weak_rewards:
        return _gate("B2_strong_weak_gap", "WARN", None,
                     t["hard_min"], t["soft_min"],
                     f"missing rollouts (strong={len(strong_rewards)}, weak={len(weak_rewards)})")
    gap = statistics.mean(strong_rewards) - statistics.mean(weak_rewards)
    status = _classify(gap, t["hard_min"], t["soft_min"], "min")
    return _gate("B2_strong_weak_gap", status, round(gap, 3),
                 t["hard_min"], t["soft_min"],
                 note=f"strong={strong} mean={statistics.mean(strong_rewards):.3f}; "
                      f"weak={weak} mean={statistics.mean(weak_rewards):.3f}")


def gate_B3(tasks: list[RLTask], rollouts: dict, t: dict) -> dict:
    """Per-mode reward std over the *primary strong* model only."""
    strong = t.get("strong_model") or "openai/gpt-5.4-mini"
    by_mode: dict[str, list[float]] = defaultdict(list)
    task_modes = {task.id: task.mode for task in tasks}
    for tid, rs in rollouts.items():
        mode = task_modes.get(tid)
        if not mode:
            continue
        for r in rs:
            if r.get("model") != strong or "reward" not in r:
                continue
            by_mode[mode].append(r["reward"])
    if not by_mode:
        return _gate("B3_per_mode_variance", "WARN", None,
                     t["hard_min"], t["soft_min"], "no rollouts")
    stds = {mode: statistics.pstdev(vals) if len(vals) > 1 else 0.0
            for mode, vals in by_mode.items()}
    min_std = min(stds.values())
    status = _classify(min_std, t["hard_min"], t["soft_min"], "min")
    return _gate("B3_per_mode_variance", status, round(min_std, 3),
                 t["hard_min"], t["soft_min"],
                 note=f"per_mode={ {k: round(v,3) for k,v in stds.items()} }")


def gate_B4(rollouts: dict, t: dict) -> dict:
    """Min over deterministic checks of 'fired in ≥ 1 rollout' rate."""
    penalty_present: Counter[str] = Counter()
    total_rollouts = 0
    for rs in rollouts.values():
        for r in rs:
            if "penalties" not in r:
                continue
            total_rollouts += 1
            for pname, pval in (r.get("penalties") or {}).items():
                if pval != 0:
                    penalty_present[pname] += 1
    if not penalty_present or total_rollouts == 0:
        return _gate("B4_check_firing_rate", "WARN", None,
                     t["hard_min"], t["soft_min"], "no penalties recorded")
    rates = {k: v / total_rollouts for k, v in penalty_present.items()}
    min_rate = min(rates.values())
    status = _classify(min_rate, t["hard_min"], t["soft_min"], "min")
    return _gate("B4_check_firing_rate", status, round(min_rate, 3),
                 t["hard_min"], t["soft_min"],
                 note=f"per_check={ {k: round(v,3) for k,v in rates.items()} }")


def gate_B5(rollouts: dict, t: dict) -> dict:
    totals: Counter[str] = Counter()
    for rs in rollouts.values():
        for r in rs:
            if "penalties" not in r:
                continue
            for pname, pval in (r.get("penalties") or {}).items():
                totals[pname] += abs(float(pval))
    total = sum(totals.values())
    if total <= 0:
        return _gate("B5_penalty_concentration", "WARN", None,
                     t["hard_max"], t["soft_max"], "no penalty mass")
    max_share = max(v / total for v in totals.values())
    status = _classify(max_share, t["hard_max"], t["soft_max"], "max")
    return _gate("B5_penalty_concentration", status, round(max_share, 3),
                 t["hard_max"], t["soft_max"])


def gate_C1(rollouts: dict, t: dict) -> dict:
    """Mean per-task rollout std over the primary strong model only."""
    strong = t.get("strong_model") or "openai/gpt-5.4-mini"
    stds = []
    for rs in rollouts.values():
        rewards = [r["reward"] for r in rs
                   if r.get("model") == strong and "reward" in r]
        if len(rewards) > 1:
            stds.append(statistics.pstdev(rewards))
    if not stds:
        return _gate("C1_rollout_std", "WARN", None,
                     t["hard_max"], t["soft_max"], "not enough rollouts per task")
    mean_std = statistics.mean(stds)
    status = _classify(mean_std, t["hard_max"], t["soft_max"], "max")
    return _gate("C1_rollout_std", status, round(mean_std, 3),
                 t["hard_max"], t["soft_max"])


def gate_C2(rollouts: dict, t: dict, primary_model: str) -> dict:
    """Spearman ρ between per-task mean reward for two strong models."""
    second = t["second_strong_model"]
    primary = t.get("primary_model", primary_model)
    paired: list[tuple[float, float]] = []
    for rs in rollouts.values():
        a = [r["reward"] for r in rs if r.get("model") == primary and "reward" in r]
        b = [r["reward"] for r in rs if r.get("model") == second and "reward" in r]
        if a and b:
            paired.append((statistics.mean(a), statistics.mean(b)))
    if len(paired) < 30:
        return _gate("C2_cross_model_rho", "WARN", None,
                     t["hard_min"], t["soft_min"],
                     f"only {len(paired)} paired tasks (<30); skipped")
    rho = _spearman_rho([p[0] for p in paired], [p[1] for p in paired])
    status = _classify(rho, t["hard_min"], t["soft_min"], "min")
    return _gate("C2_cross_model_rho", status, round(rho, 3),
                 t["hard_min"], t["soft_min"],
                 note=f"primary={primary}; second={second}; n_paired={len(paired)}")


def gate_C3(t: dict) -> dict:
    return _gate("C3_judge_kappa", "WARN", None, t["hard_min"], t["soft_min"],
                 "judge re-score requires a separate audit run; not computed here")


def gate_C4(t: dict) -> dict:
    return _gate("C4_check_determinism", "WARN", None, t["hard"], t["hard"],
                 "deterministic-check parity test requires 2 runs; see slice gate")


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    rx = _rank(xs)
    ry = _rank(ys)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - (6.0 * d2) / (n * (n * n - 1))


def _rank(vals: list[float]) -> list[float]:
    sorted_pairs = sorted(enumerate(vals), key=lambda p: p[1])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(sorted_pairs):
        j = i
        while j + 1 < len(sorted_pairs) and sorted_pairs[j + 1][1] == sorted_pairs[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[sorted_pairs[k][0]] = avg_rank
        i = j + 1
    return ranks


# ----------------------------- report ------------------------------------

def render_markdown(report: dict) -> str:
    lines = ["# humanize_tasks_v03 — verification report\n"]
    lines.append(f"- tasks: **{report['n_tasks']}**")
    lines.append(f"- rollouts: **{report['n_rollouts']}**")
    lines.append(f"- hard gates: {report['hard_pass']} PASS / {report['hard_warn']} WARN / {report['hard_fail']} FAIL\n")
    lines.append("| gate | status | value | hard | soft | note |")
    lines.append("|---|---|---|---|---|---|")
    for g in report["gates"]:
        lines.append(
            f"| {g['gate']} | **{g['status']}** | `{g['value']}` | "
            f"`{g['hard']}` | `{g['soft']}` | {g.get('note', '')} |"
        )
    return "\n".join(lines) + "\n"


@click.command(context_settings={"show_default": True})
@click.option("--input", "input_path", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--rollouts", type=click.Path(dir_okay=False, path_type=Path), default=None)
@click.option("--thresholds", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True)
@click.option("--report-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--json-out", type=click.Path(dir_okay=False, path_type=Path), required=True)
@click.option("--strict/--no-strict", default=True,
              help="Exit non-zero if any HARD gate FAILs.")
def main(
    input_path: Path, rollouts: Path | None, thresholds: Path,
    report_out: Path, json_out: Path, strict: bool,
) -> None:
    """Run all 14 verification gates and emit markdown + JSON reports."""
    tasks = load_tasks(input_path)
    rollouts_map = _load_rollouts(rollouts) if rollouts else {}
    n_rollouts = sum(len(v) for v in rollouts_map.values())
    th = yaml.safe_load(thresholds.read_text())

    strong_model = th["hardness"]["B1_strong_reward_band"]["strong_model"]
    gates: list[dict] = [
        gate_A1(tasks, th["diversity"]["A1_instruction_length"]),
        gate_A2(tasks, th["diversity"]["A2_constraint_density"]),
        gate_A3(tasks, th["diversity"]["A3_cell_coverage"]),
        gate_A4(tasks, th["diversity"]["A4_embedding_distance"]),
        gate_A5(tasks, th["diversity"]["A5_author_balance"]),
        gate_B1(rollouts_map, th["hardness"]["B1_strong_reward_band"]),
        gate_B2(rollouts_map, th["hardness"]["B2_strong_weak_gap"], strong_model),
        gate_B3(tasks, rollouts_map, th["hardness"]["B3_per_mode_variance"]),
        gate_B4(rollouts_map, th["hardness"]["B4_check_firing_rate"]),
        gate_B5(rollouts_map, th["hardness"]["B5_penalty_concentration"]),
        gate_C1(rollouts_map, th["consistency"]["C1_rollout_std"]),
        gate_C2(rollouts_map, th["consistency"]["C2_cross_model_rho"], strong_model),
        gate_C3(th["consistency"]["C3_judge_kappa"]),
        gate_C4(th["consistency"]["C4_check_determinism"]),
    ]

    hard_pass = sum(1 for g in gates if g["status"] == "PASS")
    hard_warn = sum(1 for g in gates if g["status"] == "WARN")
    hard_fail = sum(1 for g in gates if g["status"] == "FAIL")

    report = {
        "n_tasks": len(tasks),
        "n_rollouts": n_rollouts,
        "hard_pass": hard_pass,
        "hard_warn": hard_warn,
        "hard_fail": hard_fail,
        "gates": gates,
    }

    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(report, indent=2))
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(render_markdown(report))

    click.echo(render_markdown(report))
    click.echo(f"json → {json_out}")
    click.echo(f"md   → {report_out}")

    if strict and hard_fail > 0:
        click.secho(f"\n{hard_fail} HARD gate(s) failed — blocking publish.", fg="red")
        sys.exit(1)


if __name__ == "__main__":
    main()
