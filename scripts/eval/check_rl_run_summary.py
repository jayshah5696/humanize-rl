"""Check a GRPO Modal run summary against per-phase exit criteria.

Usage:
    uv run python scripts/check_rl_run_summary.py outputs/summary.json --phase probe
    uv run python scripts/check_rl_run_summary.py outputs/summary.json --phase pilot
    uv run python scripts/check_rl_run_summary.py outputs/summary.json --phase full

Exits 0 if every check passes, 1 if any fails. Prints offending metric.

See docs/plans/gemma4_rl_modal_20usd_budget_plan.md "Per-phase exit criteria".
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

PHASES = ("probe", "pilot", "full")


def _log_history(summary: dict) -> list[dict]:
    for key in ("log_history", "trainer_log_history"):
        log = summary.get(key)
        if isinstance(log, list):
            return log
    metrics = summary.get("metrics", {})
    log = metrics.get("log_history")
    return log if isinstance(log, list) else []


def _series(log: list[dict], key: str) -> list[float]:
    return [float(e[key]) for e in log if e.get(key) is not None]


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def _has_nan(log: list[dict]) -> tuple[bool, str]:
    for key in ("loss", "grad_norm", "kl", "reward"):
        for entry in log:
            v = entry.get(key)
            if v is None:
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if fv != fv:  # NaN
                return True, key
    return False, ""


def _check_common(summary: dict, log: list[dict]) -> list[str]:
    fails: list[str] = []
    nan, key = _has_nan(log)
    if nan:
        fails.append(f"NaN in metric `{key}`")
    clipped = _series(log, "completions/clipped_ratio")
    if clipped and max(clipped) >= 0.05:
        fails.append(f"clipped_ratio max={max(clipped):.3f} >= 0.05")
    peak_vram = summary.get("peak_vram_gb") or summary.get("metrics", {}).get(
        "peak_vram_gb"
    )
    if peak_vram is not None:
        if peak_vram < 12:
            fails.append(f"peak_vram={peak_vram:.1f}GB < 12 (under-utilising A100)")
        if peak_vram > 38:
            fails.append(f"peak_vram={peak_vram:.1f}GB > 38 (will OOM in next phase)")
    return fails


def check_probe(summary: dict) -> list[str]:
    log = _log_history(summary)
    fails = _check_common(summary, log)
    kls = _series(log, "kl")
    if len(kls) >= 2:
        first, last = kls[0] or 1e-6, kls[-1]
        if last > max(5.0 * first, 5.0):
            fails.append(f"kl exploded: first={first:.3f} last={last:.3f}")
    train_runtime = summary.get("train_runtime_seconds") or summary.get(
        "metrics", {}
    ).get("train_runtime")
    steps = summary.get("steps_completed") or len(_series(log, "loss")) or 4
    if train_runtime and steps:
        sec_per_step = train_runtime / steps
        if sec_per_step > 180:
            fails.append(
                f"sec/step={sec_per_step:.1f} > 180 (full run will not fit budget)"
            )
        if sec_per_step > 300:
            fails.append(
                f"sec/step={sec_per_step:.1f} > 300 (HARD STOP — do not start pilot)"
            )
    return fails


def check_pilot(summary: dict) -> list[str]:
    log = _log_history(summary)
    fails = _check_common(summary, log)
    rewards = _series(log, "reward")
    if len(rewards) < 20:
        fails.append(f"only {len(rewards)} reward points logged (need >= 20)")
    else:
        first = _mean(rewards[:10])
        last = _mean(rewards[-10:])
        if last <= first + 0.01:
            fails.append(
                f"reward trend flat/down: first10_mean={first:.3f} "
                f"last10_mean={last:.3f} (delta={last - first:+.3f})"
            )
    kls = _series(log, "kl")
    if len(kls) >= 5:
        kl_at_5 = kls[4] or 1e-6
        if kls[-1] > 10 * kl_at_5:
            fails.append(f"kl drifted: step5={kl_at_5:.3f} last={kls[-1]:.3f}")
    sps = summary.get("metrics", {}).get("train_samples_per_second")
    if sps is None:
        fails.append("train_samples_per_second not recorded (needed for Phase 5 projection)")
    return fails


def check_full(summary: dict) -> list[str]:
    log = _log_history(summary)
    fails = _check_common(summary, log)
    cost = summary.get("actual_cost_usd")
    if cost is None:
        fails.append("actual_cost_usd not recorded")
    elif cost > 14.00:
        fails.append(f"actual_cost_usd=${cost:.2f} > $14 cap")
    lora_path = summary.get("lora_path")
    if not lora_path:
        fails.append("lora_path not recorded (final LoRA not saved?)")
    return fails


CHECKERS = {"probe": check_probe, "pilot": check_pilot, "full": check_full}


@click.command()
@click.argument("summary_path", type=click.Path(exists=True, path_type=Path))
@click.option("--phase", type=click.Choice(PHASES), required=True)
def main(summary_path: Path, phase: str) -> None:
    summary = json.loads(summary_path.read_text())
    fails = CHECKERS[phase](summary)
    if fails:
        click.echo(f"FAIL ({phase}): {summary_path}")
        for f in fails:
            click.echo(f"  - {f}")
        sys.exit(1)
    click.echo(f"PASS ({phase}): {summary_path}")


if __name__ == "__main__":
    main()
