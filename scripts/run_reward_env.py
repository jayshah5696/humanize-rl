from __future__ import annotations

import json
from pathlib import Path

import click

from humanize_rl.reward.env import load_env_from_jsonl


def load_response_map(path: Path) -> dict[str, str]:
    """Load responses from JSONL rows shaped as {task_id, response}."""
    responses: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        task_id = str(row.get("task_id", ""))
        response = row.get("response")
        if not task_id or not isinstance(response, str):
            raise ValueError(f"Invalid response row at {path}:{line_number}")
        responses[task_id] = response
    return responses


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
    default=Path("outputs/reward_env/native_env_rollouts.jsonl"),
    show_default=True,
)
@click.option(
    "--split", type=click.Choice(["train", "validation", "test", "all"]), default="all"
)
def main(task_path: Path, response_path: Path, output_path: Path, split: str) -> None:
    """Run the native single-turn reward environment over response JSONL."""
    env = load_env_from_jsonl(task_path, split=split)
    steps = env.run_response_map(load_response_map(response_path))
    env.write_log_jsonl(output_path)
    mean_reward = sum(step.reward for step in steps) / len(steps) if steps else 0.0

    click.echo(f"rows={len(steps)}")
    click.echo(f"mean_reward={mean_reward:.4f}")
    click.echo(f"output_path={output_path}")


if __name__ == "__main__":
    main()
