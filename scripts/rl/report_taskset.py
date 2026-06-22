#!/usr/bin/env python3
"""Write a compact RL taskset report for Prime training gates."""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from humanize_rl.reward.env import render_prompt
from humanize_rl.reward.tasks import RLTask

DEFAULT_INPUT = Path("data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl")
DEFAULT_OUTPUT = Path("runs/reports/prime_mix_v2_p5050_taskset_report.json")


def _counter(values: list[str | None]) -> dict[str, int]:
    return dict(sorted(Counter(value or "missing" for value in values).items()))


def _percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = max(0, math.ceil((pct / 100.0) * len(ordered)) - 1)
    return ordered[index]


def _max_or_none(values: list[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _difficulty_reward_modes(rows: list[dict[str, Any]]) -> dict[str, int]:
    modes: list[str | None] = []
    for row in rows:
        difficulty_payload = row.get("difficulty")
        if isinstance(difficulty_payload, dict):
            mode = difficulty_payload.get("reward_mode")
            modes.append(str(mode) if mode else None)
        else:
            modes.append(None)
    return _counter(modes)


def _raw_extra_counts(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    values = [str(row.get(key)) if row.get(key) is not None else None for row in rows]
    return _counter(values)


def _top_counts(values: list[str], limit: int = 20) -> dict[str, int]:
    return dict(Counter(values).most_common(limit))


def build_taskset_report(
    input_path: Path,
    dataset_name: str = "mix_v2_p5050",
    reward_mode: str = "p50_50_no_penalty",
    generated_at: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable taskset report."""
    rows = [
        json.loads(line) for line in input_path.read_text().splitlines() if line.strip()
    ]
    tasks = [RLTask.model_validate(row) for row in rows]
    ids = [task.id for task in tasks]
    duplicate_ids = sorted(
        task_id for task_id, count in Counter(ids).items() if count > 1
    )
    prompt_word_counts = [len(render_prompt(task).split()) for task in tasks]

    max_words = [task.constraints.max_words for task in tasks]
    target_words = [task.constraints.target_words for task in tasks]
    source_groups = [task.source_group for task in tasks]

    return {
        "artifact": "prime_mix_v2_p5050_taskset_report",
        "generated_at": generated_at
        or datetime.now(UTC).replace(microsecond=0).isoformat(),
        "chronology": [
            "v01/v02/v03 RL tasks were generated and judged.",
            "mix_v2 combined the available RL tasks.",
            "saved Modal rollouts were filtered with p50_50_no_penalty.",
            "mix_v2_p5050 is the current Prime default candidate taskset.",
        ],
        "dataset": {
            "name": dataset_name,
            "path": str(input_path),
            "row_count": len(tasks),
            "duplicate_id_count": sum(
                Counter(ids)[task_id] - 1 for task_id in duplicate_ids
            ),
            "duplicate_unique_id_count": len(duplicate_ids),
            "duplicate_ids": duplicate_ids,
        },
        "diversity": {
            "split_counts": _counter([task.split for task in tasks]),
            "family_counts": _counter([task.family for task in tasks]),
            "mode_counts": _counter([task.mode for task in tasks]),
            "register_counts": _counter([task.register_ for task in tasks]),
            "domain_counts": _counter([task.domain for task in tasks]),
            "reward_profile_counts": _counter([task.reward_profile for task in tasks]),
            "source_group_unique_count": len(set(source_groups)),
            "source_group_top_counts": _top_counts(source_groups),
            "task_author_model_counts": _counter(
                [task.task_author_model for task in tasks]
            ),
            "difficulty_bucket_counts": _raw_extra_counts(rows, "difficulty_bucket"),
        },
        "prompt_word_length": {
            "p50": _percentile(prompt_word_counts, 50),
            "p90": _percentile(prompt_word_counts, 90),
            "max": max(prompt_word_counts) if prompt_word_counts else 0,
        },
        "constraints": {
            "max_max_words": _max_or_none(max_words),
            "max_target_words": _max_or_none(target_words),
            "target_words_task_count": sum(value is not None for value in target_words),
        },
        "reward_mode": {
            "train_reward_mode": reward_mode,
            "formula": "0.50 * ridge_rubric_mean + 0.50 * deterministic_mean",
            "penalties_in_optimizer_reward": False,
            "penalties_remain_diagnostics": True,
            "difficulty_reward_mode_counts": _difficulty_reward_modes(rows),
        },
    }


@click.command()
@click.option(
    "--input",
    "input_path",
    default=DEFAULT_INPUT,
    show_default=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    default=DEFAULT_OUTPUT,
    show_default=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--dataset-name", default="mix_v2_p5050", show_default=True)
@click.option("--reward-mode", default="p50_50_no_penalty", show_default=True)
@click.option("--generated-at", default=None, help="Optional fixed timestamp.")
def main(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    reward_mode: str,
    generated_at: str | None,
) -> None:
    """Write a JSON summary for an RL taskset."""
    report = build_taskset_report(
        input_path=input_path,
        dataset_name=dataset_name,
        reward_mode=reward_mode,
        generated_at=generated_at,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    click.echo(f"wrote taskset report to {output_path}")


if __name__ == "__main__":
    main()
