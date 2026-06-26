from __future__ import annotations

import json
from types import SimpleNamespace

from humanize_rl.reward.env import (
    HumanizeRLEnv,
    build_prime_single_turn_env,
    ensure_example_id_in_state,
    ensure_prompt_in_state,
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
    assert set(row.__dict__) == {"prompt", "info", "answer", "example_id"}
    info = json.loads(row.info)
    assert info["task_id"] == task.id
    assert info["task"]["id"] == task.id
    assert row.answer == ""
    assert row.example_id == 0


def test_ensure_prompt_in_state_restores_prompt_from_task() -> None:
    task = _task()
    row = prime_dataset_row(task)
    state = {"input": {"info": row.info}}

    ensure_prompt_in_state(state)

    assert state["prompt"] == [{"role": "user", "content": render_prompt(task)}]
    assert state["input"]["prompt"] == state["prompt"]


def test_ensure_prompt_in_state_restores_prompt_from_question() -> None:
    state = {"input": {"question": "Rewrite this plainly."}}

    ensure_prompt_in_state(state)

    assert state["prompt"] == [{"role": "user", "content": "Rewrite this plainly."}]


def test_ensure_example_id_in_state_restores_id_from_task_id() -> None:
    state = {"input": {"task_id": "rl_v03_000123"}}

    ensure_example_id_in_state(state)

    assert state["example_id"] == 123
    assert state["input"]["example_id"] == 123


def test_ensure_example_id_in_state_restores_id_from_task_payload() -> None:
    task = _task()
    row = prime_dataset_row(task)
    state = {"input": {"info": row.info}}

    ensure_example_id_in_state(state)

    assert state["example_id"] == 1


def test_prime_setup_state_keeps_parent_mutation_when_parent_returns_none() -> None:
    task = _task()
    row = prime_dataset_row(task)

    class ParentReturnsNoneSingleTurnEnv:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        async def setup_state(self, state):
            state["input"] = {"info": row.info}
            return None

    vf = SimpleNamespace(SingleTurnEnv=ParentReturnsNoneSingleTurnEnv)
    env = build_prime_single_turn_env(vf, dataset=[], rubric=object())

    import asyncio

    state = asyncio.run(env.setup_state({}))

    assert state["example_id"] == 1
    assert state["prompt"] == [{"role": "user", "content": render_prompt(task)}]
    assert state["input"]["example_id"] == 1
    assert state["input"]["prompt"] == state["prompt"]


def test_load_env_from_jsonl_filters_split(tmp_path) -> None:
    task = _task()
    path = tmp_path / "tasks.jsonl"
    path.write_text(task.model_dump_json(by_alias=True) + "\n")

    env = load_env_from_jsonl(path, split="train")

    assert len(env.tasks) == 1
