"""Project full-run cost on A100-40GB from a pilot summary + full-run config.

Usage:
    uv run python scripts/project_full_run_cost.py outputs/pilot_summary.json \
        configs/rl/gemma4_e2b_rl_a100_full.yaml

Exits 0 if projected_cost_usd <= 14, else 1 (and prints suggested fixes).

See docs/plans/gemma4_rl_modal_20usd_budget_plan.md "Phase 5".
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import click
import yaml

A100_USD_PER_SEC = 0.000583  # Modal A100-40GB, May 2026
COST_CAP_USD = 14.0


def _train_samples_per_second(summary: dict) -> float | None:
    for path in (
        ("metrics", "train_samples_per_second"),
        ("train_samples_per_second",),
    ):
        node = summary
        ok = True
        for key in path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and isinstance(node, (int, float)):
            return float(node)
    return None


def _train_rows_count(task_path: Path) -> int:
    if not task_path.exists():
        return 0
    count = 0
    for line in task_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("split", "train") == "train":
            count += 1
    return count


def _project(
    samples_per_sec: float,
    per_device_bs: int,
    grad_accum: int,
    num_generations: int,
    train_rows: int,
    epochs: float,
) -> tuple[float, float, int]:
    samples_per_step = per_device_bs * grad_accum * num_generations
    seconds_per_step = samples_per_step / samples_per_sec
    eff_batch = per_device_bs * grad_accum
    steps_per_epoch = math.ceil(train_rows / max(eff_batch, 1))
    total_steps = int(steps_per_epoch * epochs)
    wall_sec = total_steps * seconds_per_step
    cost = wall_sec * A100_USD_PER_SEC
    return wall_sec, cost, total_steps


@click.command()
@click.argument("pilot_summary", type=click.Path(exists=True, path_type=Path))
@click.argument("full_config", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--task-path",
    type=click.Path(path_type=Path),
    default=Path("data/rl/humanize_tasks_v01_smoke.jsonl"),
    show_default=True,
)
def main(pilot_summary: Path, full_config: Path, task_path: Path) -> None:
    summary = json.loads(pilot_summary.read_text())
    cfg = yaml.safe_load(full_config.read_text())

    sps = _train_samples_per_second(summary)
    if sps is None or sps <= 0:
        click.echo("FAIL: pilot summary has no train_samples_per_second")
        sys.exit(1)

    train_rows = _train_rows_count(task_path)
    if train_rows == 0:
        click.echo(f"FAIL: no train rows in {task_path}")
        sys.exit(1)

    epochs = float(cfg.get("num_train_epochs") or 0)
    if epochs <= 0:
        max_steps = cfg.get("max_steps")
        if max_steps and max_steps > 0:
            eff_batch = cfg["per_device_train_batch_size"] * cfg[
                "gradient_accumulation_steps"
            ]
            epochs = (max_steps * eff_batch) / max(train_rows, 1)
        else:
            epochs = 1.0

    wall_sec, cost, steps = _project(
        samples_per_sec=sps,
        per_device_bs=int(cfg["per_device_train_batch_size"]),
        grad_accum=int(cfg["gradient_accumulation_steps"]),
        num_generations=int(cfg["num_generations"]),
        train_rows=train_rows,
        epochs=epochs,
    )

    click.echo(f"config           : {full_config}")
    click.echo(f"pilot samples/s  : {sps:.3f}")
    click.echo(f"train rows       : {train_rows}")
    click.echo(f"epochs           : {epochs:.2f}")
    click.echo(f"projected steps  : {steps}")
    click.echo(f"projected wall   : {wall_sec / 60:.1f} min ({wall_sec / 3600:.2f} hr)")
    click.echo(f"projected cost   : ${cost:.2f}  (cap ${COST_CAP_USD:.2f})")

    if cost <= COST_CAP_USD:
        click.echo("PASS")
        sys.exit(0)

    click.echo("FAIL: projected cost exceeds cap. Try in order:")
    mcl = cfg.get("max_completion_length", 2048)
    if mcl > 1024:
        click.echo(f"  1. max_completion_length: {mcl} -> 1024 (~1.6x cheaper)")
    if int(cfg["num_generations"]) > 4:
        click.echo(f"  2. num_generations: {cfg['num_generations']} -> 4 (~1.7x cheaper)")
    if epochs > 2:
        click.echo(f"  3. num_train_epochs: {epochs:.1f} -> 2.0")
    sys.exit(1)


if __name__ == "__main__":
    main()
