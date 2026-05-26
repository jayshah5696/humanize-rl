"""Dataset adapter for Unsloth/TRL GRPO training."""

from __future__ import annotations

from collections import Counter, defaultdict, deque
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


def stratify_rows_by_key(
    rows: list[dict[str, Any]],
    *,
    batch_size: int,
    key: str = "reward_profile",
) -> list[dict[str, Any]]:
    """Order rows so each sequential optimizer-step block has a stable mix.

    Slice 2 showed that random task-family mix made ``risk_penalty`` swamp the
    learned reward trend. This weighted-fair ordering keeps each block of
    ``batch_size`` rows close to the global distribution for ``key`` while using
    every row exactly once.
    """
    if batch_size <= 1 or len(rows) <= batch_size:
        return rows

    buckets: dict[str, deque[dict[str, Any]]] = defaultdict(deque)
    for row in rows:
        buckets[str(row.get(key, ""))].append(row)

    totals = Counter(str(row.get(key, "")) for row in rows)
    used = Counter[str]()
    ordered: list[dict[str, Any]] = []
    total_rows = len(rows)
    keys = sorted(totals)

    while len(ordered) < total_rows:
        block_end = min(len(ordered) + batch_size, total_rows)
        block_size = block_end - len(ordered)
        block: list[dict[str, Any]] = []
        while len(block) < block_size:
            candidates = []
            for bucket_key in keys:
                if not buckets[bucket_key]:
                    continue
                desired = totals[bucket_key] * block_end / total_rows
                deficit = desired - used[bucket_key]
                candidates.append((deficit, len(buckets[bucket_key]), bucket_key))
            if not candidates:
                break
            _, _, chosen_key = max(candidates)
            block.append(buckets[chosen_key].popleft())
            used[chosen_key] += 1
        ordered.extend(block)

    return ordered


def load_grpo_rows(
    path: Path,
    split: str = "train",
    *,
    stratify_batch_size: int | None = None,
    stratify_by: str = "reward_profile",
) -> list[dict[str, Any]]:
    """Load GRPO rows from the RL task JSONL."""
    tasks = load_tasks(path)
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    rows = [task_to_grpo_row(task) for task in tasks]
    if stratify_batch_size is not None:
        return stratify_rows_by_key(
            rows, batch_size=stratify_batch_size, key=stratify_by
        )
    return rows


def load_grpo_dataset(
    path: Path,
    split: str = "train",
    *,
    stratify_batch_size: int | None = None,
    stratify_by: str = "reward_profile",
) -> Any:
    """Load a Hugging Face Dataset for GRPO.

    Imported lazily so local tests do not require datasets internals beyond import time.
    """
    from datasets import Dataset

    return Dataset.from_list(
        load_grpo_rows(
            path,
            split=split,
            stratify_batch_size=stratify_batch_size,
            stratify_by=stratify_by,
        )
    )
