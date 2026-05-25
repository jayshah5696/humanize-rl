from __future__ import annotations

import json

from humanize_rl.reward.tasks import (
    RLTask,
    TaskConstraints,
    load_tasks,
    summarize_tasks,
)


def _task_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "rl_v01_000001",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up this Slack update.",
        "input_text": "Staging recovered at 3 pm.",
        "constraints": {"max_words": 30, "preserve_numbers": True},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
    }
    base.update(overrides)
    return base


def test_task_schema_strips_text_and_validates_constraints() -> None:
    task = RLTask.model_validate(
        _task_kwargs(
            instruction="  Clean this up.  ", input_text="  Staging is back.  "
        )
    )

    assert task.instruction == "Clean this up."
    assert task.input_text == "Staging is back."
    assert task.constraints.max_words == 30


def test_load_tasks_round_trip(tmp_path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text(json.dumps(_task_kwargs()) + "\n")

    tasks = load_tasks(path)

    assert len(tasks) == 1
    assert tasks[0].id == "rl_v01_000001"


def test_summarize_tasks_counts_traps() -> None:
    tasks = [
        RLTask.model_validate(_task_kwargs()),
        RLTask.model_validate(
            _task_kwargs(
                id="rl_v01_000002",
                family="slack_chat",
                mode="direct_generation",
                input_text="",
                reward_profile="direct_workplace_message",
                trap_tags=["wrapper_phrase", "verbosity"],
            )
        ),
    ]

    summary = summarize_tasks(tasks)

    assert summary["family"] == {"rewrite_repair": 1, "slack_chat": 1}
    assert summary["trap"]["wrapper_phrase"] == 2


def test_constraints_reject_invalid_word_window() -> None:
    try:
        TaskConstraints(min_words=20, max_words=10)
    except ValueError as exc:
        assert "min_words" in str(exc)
    else:
        raise AssertionError("Expected invalid word window to fail")
