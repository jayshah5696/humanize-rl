from __future__ import annotations

import asyncio

from humanize_rl.reward.tasks import RLTask
from humanize_rl.reward.verifiers_adapter import (
    build_verifiers_rubric,
    humanize_reward,
    option_menu_penalty_metric,
    style_metric,
    task_following_metric,
)


def _task_payload() -> dict[str, object]:
    task = RLTask.model_validate(
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
            "trap_tags": ["option_menu", "wrapper_phrase"],
            "split": "train",
            "required_facts": ["Staging recovered", "3 pm", "STRIPE_WEBHOOK_SECRET"],
        }
    )
    return task.model_dump(by_alias=True, exclude_none=True)


def test_verifiers_reward_caches_result_and_metrics_read_state() -> None:
    task = _task_payload()
    completion = [
        {
            "role": "assistant",
            "content": "Here's a version:\n\nOption 1: Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        }
    ]
    state: dict[str, object] = {}

    reward = asyncio.run(humanize_reward(completion, task, state))
    style = asyncio.run(style_metric(completion, task, state))
    following = asyncio.run(task_following_metric(completion, task, state))
    option_penalty = asyncio.run(option_menu_penalty_metric(completion, task, state))

    assert reward < 1.0
    assert 0.0 <= style <= 1.0
    assert following < 1.0
    assert option_penalty == -0.40
    assert "humanize_reward_result" in state
    assert "humanize_components" in state
    assert "humanize_penalties" in state


class FakeRubric:
    def __init__(self, funcs):
        self.funcs = funcs
        self.metrics = []

    def add_metric(self, func):
        self.metrics.append(func)


class FakeVerifiers:
    Rubric = FakeRubric


def test_build_verifiers_rubric_has_reward_and_metrics() -> None:
    rubric = build_verifiers_rubric(FakeVerifiers)

    assert rubric.funcs == [humanize_reward]
    assert len(rubric.metrics) >= 10
