from __future__ import annotations

from pathlib import Path

from humanize_rl.reward.grpo_dataset import load_grpo_rows, task_to_grpo_row
from humanize_rl.reward.grpo_rewards import WEIGHTED_REWARD_FUNCS, scalar_reward
from humanize_rl.reward.tasks import RLTask


def _task() -> RLTask:
    return RLTask.model_validate(
        {
            "id": "rl_v01_000001",
            "family": "rewrite_repair",
            "domain": "slack",
            "mode": "rewrite",
            "register": "casual",
            "instruction": "Clean up this Slack update.",
            "input_text": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
            "constraints": {
                "max_words": 20,
                "preserve_numbers": True,
                "preserve_entities": True,
            },
            "reward_profile": "rewrite_faithful_concise",
            "trap_tags": ["wrapper_phrase"],
            "split": "train",
            "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
        }
    )


def test_task_to_grpo_row_contains_prompt_and_task_payload() -> None:
    task = _task()
    row = task_to_grpo_row(task)

    assert row["prompt"][0]["role"] == "user"
    assert "Source:" in row["prompt"][0]["content"]
    assert row["task"]["id"] == task.id
    assert row["task_id"] == task.id


def test_load_grpo_rows_filters_split(tmp_path: Path) -> None:
    task = _task()
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.model_dump_json(by_alias=True) + "\n")

    rows = load_grpo_rows(path, split="train")

    assert len(rows) == 1
    assert rows[0]["task_id"] == task.id


def test_scalar_reward_matches_completion_count() -> None:
    task_payload = _task().model_dump(by_alias=True, exclude_none=True)
    completions = [
        [
            {
                "role": "assistant",
                "content": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
            }
        ],
        [{"role": "assistant", "content": "Here's a version: Sarah fixed it."}],
    ]

    scores = scalar_reward(completions, task=[task_payload, task_payload])

    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_weighted_reward_funcs_sum_to_scalar_without_double_counting() -> None:
    """WEIGHTED_REWARD_FUNCS (ridge + deterministic + penalty) must sum to scalar_reward."""
    task_payload = _task().model_dump(by_alias=True, exclude_none=True)
    completions = [
        [
            {
                "role": "assistant",
                "content": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
            }
        ]
    ]

    scalar = scalar_reward(completions, task=[task_payload])[0]
    component_sum = sum(
        func(completions, task=[task_payload])[0] for func in WEIGHTED_REWARD_FUNCS
    )

    # scalar_reward is clipped to [-1,1]; component_sum is raw.
    # Assert they agree within the clip margin.
    from humanize_rl.reward.reward import clip
    assert abs(clip(component_sum) - scalar) < 1e-6
