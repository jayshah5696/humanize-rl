#!/usr/bin/env python3
"""Build the p50_50_no_penalty filtered RL task mix from saved rollouts."""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import click


@dataclass(frozen=True)
class FilterThresholds:
    min_reward: float = 0.20
    max_reward: float = 1.00
    min_reward_std: float = 0.005
    max_clipped_rate: float = 0.10


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def p50_reward(rollout: dict) -> float:
    """Stored rollout fields are legacy-weighted 0.5 ridge + 0.5 deterministic."""
    return float(rollout["ridge_rubric"]) + float(rollout["deterministic"])


def _pstd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _bucket(
    rewards: list[float],
    clipped_rate: float,
    thresholds: FilterThresholds,
) -> tuple[str, list[str]]:
    reward_mean = statistics.fmean(rewards)
    reward_std = _pstd(rewards)
    if clipped_rate > thresholds.max_clipped_rate:
        return "clipped", [
            f"clipped_rate={clipped_rate:.2f}>max={thresholds.max_clipped_rate}"
        ]
    if reward_std < thresholds.min_reward_std:
        return "dead", [f"reward_std={reward_std:.3f}<min={thresholds.min_reward_std}"]
    if reward_mean <= thresholds.min_reward:
        return "too_hard", [
            f"reward_mean={reward_mean:.3f}<=min={thresholds.min_reward}"
        ]
    if reward_mean >= thresholds.max_reward:
        return "too_easy", [
            f"reward_mean={reward_mean:.3f}>=max={thresholds.max_reward}"
        ]
    return "useful", []


def summarize_task(
    task_id: str,
    rollouts: list[dict],
    thresholds: FilterThresholds,
) -> dict:
    rewards = [p50_reward(row) for row in rollouts]
    clipped_rate = sum(1 for row in rollouts if row.get("clipped")) / len(rollouts)
    bucket, reasons = _bucket(rewards, clipped_rate, thresholds)
    return {
        "task_id": task_id,
        "n_completions": len(rollouts),
        "reward_mean": statistics.fmean(rewards),
        "reward_std": _pstd(rewards),
        "ridge_mean": statistics.fmean(float(row["ridge_rubric"]) for row in rollouts),
        "det_mean": statistics.fmean(float(row["deterministic"]) for row in rollouts),
        "penalty_rate": sum(1 for row in rollouts if float(row["penalty_sum"]) < 0.0)
        / len(rollouts),
        "clipped_rate": clipped_rate,
        "bucket": bucket,
        "kept": bucket == "useful",
        "reasons": reasons,
    }


def build_filtered_mix(
    mix_rows: list[dict],
    rollout_rows: list[dict],
    thresholds: FilterThresholds,
) -> tuple[list[dict], list[dict], dict]:
    ids = [row["id"] for row in mix_rows]
    duplicates = [task_id for task_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise ValueError(f"duplicate task ids in mix: {duplicates[:5]}")

    by_task: dict[str, list[dict]] = defaultdict(list)
    for row in rollout_rows:
        by_task[str(row["task_id"])].append(row)

    difficulties: dict[str, dict] = {
        task_id: summarize_task(task_id, rows, thresholds)
        for task_id, rows in by_task.items()
    }

    filtered: list[dict] = []
    drops: Counter[str] = Counter()
    per_task_rows: list[dict] = []
    for row in mix_rows:
        task_id = row["id"]
        diff = difficulties.get(task_id)
        if diff is None:
            drops["unscored"] += 1
            continue
        per_task_rows.append(diff)
        if diff["kept"]:
            row_copy = dict(row)
            row_copy["difficulty_bucket"] = diff["bucket"]
            row_copy["difficulty"] = {
                "reward_mode": "p50_50_no_penalty",
                "reward_mean": diff["reward_mean"],
                "reward_std": diff["reward_std"],
                "penalty_rate": diff["penalty_rate"],
                "clipped_rate": diff["clipped_rate"],
                "n_completions": diff["n_completions"],
            }
            filtered.append(row_copy)
        else:
            drops[diff["bucket"]] += 1

    summary = {
        "reward_mode": "p50_50_no_penalty",
        "input_rows": len(mix_rows),
        "rollout_rows": len(rollout_rows),
        "scored_tasks": len(difficulties),
        "kept": len(filtered),
        "dropped": dict(drops),
        "bucket_counts": dict(Counter(row["bucket"] for row in per_task_rows)),
        "split_counts": dict(Counter(row.get("split", "?") for row in filtered)),
        "family_counts": dict(Counter(row.get("family", "?") for row in filtered)),
        "mode_counts": dict(Counter(row.get("mode", "?") for row in filtered)),
        "dataset_version_counts": dict(
            Counter(row.get("dataset_version", "?") for row in filtered)
        ),
        "thresholds": thresholds.__dict__,
    }
    return filtered, per_task_rows, summary


@click.command()
@click.option(
    "--mix",
    "mix_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--rollouts",
    "rollouts_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--summary",
    "summary_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--per-task",
    "per_task_path",
    default=None,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option("--min-reward", default=0.20, show_default=True, type=float)
@click.option("--max-reward", default=1.00, show_default=True, type=float)
@click.option("--min-reward-std", default=0.005, show_default=True, type=float)
@click.option("--max-clipped-rate", default=0.10, show_default=True, type=float)
def main(
    mix_path: Path,
    rollouts_path: Path,
    output_path: Path,
    summary_path: Path,
    per_task_path: Path | None,
    min_reward: float,
    max_reward: float,
    min_reward_std: float,
    max_clipped_rate: float,
) -> None:
    """Rebuild the useful p50-filtered task set from saved Modal rollouts."""
    thresholds = FilterThresholds(
        min_reward=min_reward,
        max_reward=max_reward,
        min_reward_std=min_reward_std,
        max_clipped_rate=max_clipped_rate,
    )
    try:
        filtered, per_task_rows, summary = build_filtered_mix(
            _read_jsonl(mix_path),
            _read_jsonl(rollouts_path),
            thresholds,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    _write_jsonl(output_path, filtered)
    if per_task_path is not None:
        _write_jsonl(per_task_path, per_task_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    click.echo(
        f"wrote {len(filtered)} rows to {output_path} "
        f"from {summary['input_rows']} input tasks"
    )


if __name__ == "__main__":
    main()
