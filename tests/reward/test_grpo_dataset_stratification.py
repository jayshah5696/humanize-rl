from __future__ import annotations

from collections import Counter
from pathlib import Path

from humanize_rl.reward.grpo_dataset import load_grpo_rows, stratify_rows_by_key

TASK_PATH = Path("data/rl/humanize_tasks_v01_smoke.jsonl")


def test_stratify_rows_preserves_all_rows() -> None:
    rows = load_grpo_rows(TASK_PATH, split="train")
    ordered = stratify_rows_by_key(rows, batch_size=8, key="reward_profile")

    assert len(ordered) == len(rows)
    assert Counter(row["task_id"] for row in ordered) == Counter(
        row["task_id"] for row in rows
    )


def test_stratified_blocks_have_stable_reward_profile_mix() -> None:
    rows = load_grpo_rows(
        TASK_PATH,
        split="train",
        stratify_batch_size=8,
        stratify_by="reward_profile",
    )

    sensitive_counts = [
        sum(row["reward_profile"] == "sensitive_comms" for row in rows[i : i + 8])
        for i in range(0, len(rows), 8)
    ]

    assert max(sensitive_counts) - min(sensitive_counts) <= 1


def test_unstratified_loading_keeps_original_order() -> None:
    plain = load_grpo_rows(TASK_PATH, split="train")
    stratified = load_grpo_rows(
        TASK_PATH,
        split="train",
        stratify_batch_size=8,
        stratify_by="reward_profile",
    )

    assert [row["task_id"] for row in plain] != [row["task_id"] for row in stratified]
