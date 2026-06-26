from __future__ import annotations

import json

from scripts.rl.report_taskset import build_taskset_report


def _row(task_id: str, split: str, *, target_words: int | None) -> dict:
    constraints = {
        "max_words": 40,
        "preserve_numbers": True,
        "preserve_entities": True,
    }
    if target_words is not None:
        constraints["target_words"] = target_words
    return {
        "id": task_id,
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Rewrite this update plainly.",
        "input_text": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        "constraints": constraints,
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": split,
        "required_facts": ["Staging recovered"],
        "difficulty_bucket": "useful",
        "difficulty": {"reward_mode": "p50_50_no_penalty"},
    }


def test_build_taskset_report_summarizes_counts_and_lengths(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    rows = [
        _row("rl_v01_000001", "train", target_words=120),
        _row("rl_v01_000002", "validation", target_words=None),
        _row("rl_v01_000002", "test", target_words=200),
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))

    report = build_taskset_report(
        path,
        generated_at="2026-06-18T00:00:00+00:00",
    )

    assert report["dataset"]["row_count"] == 3
    assert report["dataset"]["duplicate_id_count"] == 1
    assert report["diversity"]["split_counts"] == {
        "test": 1,
        "train": 1,
        "validation": 1,
    }
    assert report["constraints"]["max_target_words"] == 200
    assert report["prompt_word_length"]["max"] > 0
    assert report["reward_mode"]["penalties_in_optimizer_reward"] is False
