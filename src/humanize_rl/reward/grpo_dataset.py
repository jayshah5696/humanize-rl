"""Dataset adapter for Unsloth/TRL GRPO training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from humanize_rl.reward.env import render_prompt
from humanize_rl.reward.tasks import RLTask, load_tasks


def task_to_grpo_row(task: RLTask) -> dict[str, Any]:
    """Convert one RL task to the TRL GRPO prompt row shape."""
    return {
        "prompt": [{"role": "user", "content": render_prompt(task)}],
        "task": task.model_dump(by_alias=True, exclude_none=True),
        "task_id": task.id,
        "family": task.family,
        "reward_profile": task.reward_profile,
    }


def load_grpo_rows(path: Path, split: str = "train") -> list[dict[str, Any]]:
    """Load GRPO rows from the RL task JSONL."""
    tasks = load_tasks(path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    return [task_to_grpo_row(task) for task in tasks]


def load_grpo_dataset(path: Path, split: str = "train") -> Any:
    """Load a Hugging Face Dataset for GRPO.

    Imported lazily so local tests do not require datasets internals beyond import time.
    """
    from datasets import Dataset

    return Dataset.from_list(load_grpo_rows(path, split=split))
