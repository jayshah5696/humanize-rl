from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from humanize_rl.reward.env import prime_dataset_row, render_prompt
from humanize_rl.reward.tasks import load_tasks
from humanize_rl.reward.verifiers_adapter import (
    build_verifiers_rubric,
    clarity_metric,
    faithfulness_metric,
    format_metric,
    humanize_reward,
    invented_detail_penalty_metric,
    length_metric,
    option_menu_penalty_metric,
    placeholder_metric,
    risk_penalty_metric,
    style_metric,
    task_following_metric,
    wrapper_phrase_penalty_metric,
)

DEFAULT_TASK_PATH = (
    Path(__file__).resolve().parents[2] / "data/rl/humanize_tasks_v01_smoke.jsonl"
)


def _load_rows(task_path: Path, split: str) -> list[dict[str, object]]:
    tasks = load_tasks(task_path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    return [prime_dataset_row(task).__dict__ for task in tasks]


def load_environment(
    split: str = "train",
    task_path: str | None = None,
) -> Any:
    """Load a Prime Verifiers SingleTurnEnv.

    Current Verifiers docs use:
    - `Dataset.from_list` with a `prompt` message list column;
    - async reward functions receiving `completion` and dataset columns;
    - `vf.Rubric(funcs=[reward_fn])`;
    - `vf.SingleTurnEnv(dataset=dataset, rubric=rubric)`.
    """
    try:
        import verifiers as vf
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "Install Prime Verifiers before running this environment: `uv add verifiers`."
        ) from exc

    resolved_task_path = Path(task_path) if task_path is not None else DEFAULT_TASK_PATH
    dataset = Dataset.from_list(_load_rows(resolved_task_path, split))
    rubric = build_verifiers_rubric(vf)
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)


# Useful for quick import smoke tests without Verifiers installed.
def preview_dataset_row(
    task_path: str | None = None, split: str = "train"
) -> dict[str, object]:
    resolved_task_path = Path(task_path) if task_path is not None else DEFAULT_TASK_PATH
    rows = _load_rows(resolved_task_path, split)
    return rows[0] if rows else {}


# Keep these names referenced so package checkers see the public contract.
__all__ = [
    "humanize_reward",
    "style_metric",
    "task_following_metric",
    "faithfulness_metric",
    "length_metric",
    "format_metric",
    "clarity_metric",
    "placeholder_metric",
    "risk_penalty_metric",
    "option_menu_penalty_metric",
    "wrapper_phrase_penalty_metric",
    "invented_detail_penalty_metric",
    "load_environment",
    "preview_dataset_row",
    "render_prompt",
    "json",
]
