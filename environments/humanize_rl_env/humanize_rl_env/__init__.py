"""Prime Intellect Verifiers environment — Humanize-RL.

Self-contained: all scoring logic is bundled in humanize_rl_env/ sub-package.
No dependency on a separately-installed humanize-rl package.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from humanize_rl_env.reward.env import (
    build_prime_single_turn_env,
    prime_dataset_row,
    render_prompt,
)
from humanize_rl_env.reward.tasks import load_tasks
from humanize_rl_env.reward.verifiers_adapter import (
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

PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_TASK_SET = "mix_v2_p5050_filtered"
TASK_SET_PATHS = {
    "v02_smoke": PACKAGE_DIR / "humanize_tasks_v02_smoke.jsonl",
    "v03": PACKAGE_DIR / "humanize_tasks_v03_filtered.jsonl",
    "mix_v2": PACKAGE_DIR / "humanize_tasks_rl_mix_v2.jsonl",
    "mix_v2_p5050": PACKAGE_DIR / "humanize_tasks_rl_mix_v2_p5050_filtered.jsonl",
    "mix_v2_p5050_filtered": PACKAGE_DIR
    / "humanize_tasks_rl_mix_v2_p5050_filtered.jsonl",
}
SUPPORTED_REWARD_MODES = {"strict", "scalar_softened", "p50_50_no_penalty"}


def _resolve_task_path(task_path: str | None, task_set: str) -> Path:
    if task_path is not None:
        return Path(task_path)
    if task_set not in TASK_SET_PATHS:
        supported = ", ".join(sorted(TASK_SET_PATHS))
        raise ValueError(f"unknown task_set: {task_set!r}; supported: {supported}")
    return TASK_SET_PATHS[task_set]


def _load_rows(task_path: Path, split: str) -> list[dict[str, object]]:
    tasks = load_tasks(task_path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    return [
        prime_dataset_row(task, example_id=i).__dict__ for i, task in enumerate(tasks)
    ]


def load_environment(
    split: str = "train",
    task_path: str | None = None,
    task_set: str = DEFAULT_TASK_SET,
    reward_mode: str = "p50_50_no_penalty",
) -> Any:
    """Load a Prime Verifiers SingleTurnEnv for the Humanize-RL task."""
    if reward_mode not in SUPPORTED_REWARD_MODES:
        supported = ", ".join(sorted(SUPPORTED_REWARD_MODES))
        raise ValueError(
            f"unknown reward_mode: {reward_mode!r}; supported: {supported}"
        )
    try:
        import verifiers as vf
        from datasets import Dataset
    except ImportError as exc:
        raise ImportError(
            "Install Prime Verifiers before running this environment: `uv add verifiers`."
        ) from exc

    resolved_task_path = _resolve_task_path(task_path, task_set)
    dataset = Dataset.from_list(_load_rows(resolved_task_path, split))
    rubric = build_verifiers_rubric(vf, reward_mode=reward_mode)
    return build_prime_single_turn_env(vf, dataset, rubric)


def preview_dataset_row(
    task_path: str | None = None,
    split: str = "train",
    task_set: str = DEFAULT_TASK_SET,
) -> dict[str, object]:
    """Return the first dataset row without requiring verifiers installed."""
    resolved_task_path = _resolve_task_path(task_path, task_set)
    rows = _load_rows(resolved_task_path, split)
    return rows[0] if rows else {}


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
    "DEFAULT_TASK_SET",
    "TASK_SET_PATHS",
    "SUPPORTED_REWARD_MODES",
]
