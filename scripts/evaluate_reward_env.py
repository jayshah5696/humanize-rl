from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import click

from humanize_rl.reward.reward import load_track_a_scorer, score_jsonl


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_scores(rows: list[dict[str, object]]) -> dict[str, object]:
    rewards = [float(row["reward"]) for row in rows]
    by_profile: dict[str, list[float]] = defaultdict(list)
    penalties: Counter[str] = Counter()
    for row in rows:
        by_profile[str(row["profile"])].append(float(row["reward"]))
        penalty_map = row.get("penalties", {})
        if isinstance(penalty_map, dict):
            penalties.update(penalty_map.keys())

    return {
        "rows": len(rows),
        "mean_reward": _mean(rewards),
        "min_reward": min(rewards) if rewards else 0.0,
        "max_reward": max(rewards) if rewards else 0.0,
        "by_profile": {
            profile: _mean(values) for profile, values in by_profile.items()
        },
        "penalty_counts": dict(penalties),
    }


@click.command()
@click.option(
    "--task-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/rl/humanize_tasks_v01_smoke.jsonl"),
    show_default=True,
)
@click.option(
    "--response-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
)
@click.option(
    "--output-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("outputs/reward_env/scored_responses.jsonl"),
    show_default=True,
)
@click.option(
    "--summary-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("outputs/reward_env/summary.json"),
    show_default=True,
)
@click.option(
    "--track-a-scorer-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
)
def main(
    task_path: Path,
    response_path: Path,
    output_path: Path,
    summary_path: Path,
    track_a_scorer_path: Path | None,
) -> None:
    """Score offline RL rollout responses and write reward diagnostics."""
    scorer = load_track_a_scorer(track_a_scorer_path)
    rows = score_jsonl(task_path, response_path, output_path, scorer)
    summary = summarize_scores(rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    click.echo(f"rows={summary['rows']}")
    click.echo(f"mean_reward={summary['mean_reward']:.4f}")
    click.echo(f"output_path={output_path}")
    click.echo(f"summary_path={summary_path}")


if __name__ == "__main__":
    main()
