from __future__ import annotations

import json

from humanize_rl.reward.env import (
    HumanizeRLEnv,
    load_env_from_jsonl,
    prime_dataset_row,
    render_prompt,
    write_rollout_log,
)
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


def test_native_env_one_episode_per_task() -> None:
    task = _task()
    env = HumanizeRLEnv(tasks=[task])

    observation = env.reset()
    step = env.step("Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.")

    assert observation.task_id == task.id
    assert step.done is True
    assert step.reward > 0
    assert step.info["components"]
    assert len(env.rollout_log) == 1


def test_env_writes_jsonl_diagnostics(tmp_path) -> None:
    task = _task()
    env = HumanizeRLEnv(tasks=[task])
    env.reset(task)
    step = env.step("Here's a version: Option 1: Sarah fixed staging.")
    path = tmp_path / "rollouts.jsonl"

    write_rollout_log(path, [step])

    row = json.loads(path.read_text().strip())
    assert row["task_id"] == task.id
    assert row["done"] is True
    assert "diagnostics" in row["info"]
    assert "wrapper_phrase" in row["info"]["penalties"]


def test_prime_dataset_row_matches_verifiers_single_turn_shape() -> None:
    task = _task()
    row = prime_dataset_row(task)

    assert row.prompt == [{"role": "user", "content": render_prompt(task)}]
    assert row.task_id == task.id
    assert json.loads(row.task)["id"] == task.id  # task is now a JSON string
    assert json.loads(row.info)["task_id"] == task.id


def test_load_env_from_jsonl_filters_split(tmp_path) -> None:
    task = _task()
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.model_dump_json(by_alias=True) + "\n")

    env = load_env_from_jsonl(path, split="train")

    assert len(env.tasks) == 1
