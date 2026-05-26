"""Prime Verifiers rubric adapter for the Humanize-RL reward scorer."""

from __future__ import annotations

import json
from typing import Any

from humanize_rl.reward.reward import RewardResult, load_ridge_scorer, score_response
from humanize_rl.reward.tasks import RLTask

# Loaded once at import time; None if pkl not found (graceful degradation).
_RIDGE_SCORER = load_ridge_scorer()

Completion = list[dict[str, str]]
State = dict[str, Any]

REWARD_STATE_KEY = "humanize_reward_result"


def response_from_completion(completion: Completion | None) -> str:
    """Extract assistant text from a Verifiers completion message list."""
    if not completion:
        return ""
    return completion[-1].get("content", "")


def _result_from_state(state: State) -> RewardResult | None:
    result = state.get(REWARD_STATE_KEY)
    if isinstance(result, RewardResult):
        return result
    return None


def _parse_task(task: dict[str, object] | str) -> dict[str, object]:
    """Accept task as dict or JSON string (string since v0.1.9)."""
    if isinstance(task, str):
        return json.loads(task)
    return task


def score_for_verifiers(
    completion: Completion | None,
    task: dict[str, object] | str,
    state: State | None = None,
) -> RewardResult:
    """Score a Verifiers rollout and optionally cache diagnostics in state."""
    if state is not None:
        cached = _result_from_state(state)
        if cached is not None:
            return cached

    result = score_response(
        RLTask.model_validate(_parse_task(task)),
        response_from_completion(completion),
        _RIDGE_SCORER,
    )
    if state is not None:
        state[REWARD_STATE_KEY] = result
        state["humanize_reward"] = result.reward
        state["humanize_components"] = result.components
        state["humanize_penalties"] = result.penalties
        state["humanize_diagnostics"] = result.diagnostics
    return result


async def humanize_reward(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    """Primary scalar reward for Prime Verifiers training/eval."""
    return score_for_verifiers(completion, task, state).reward


async def style_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get("style", 0.0)


async def task_following_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get(
        "task_following", 0.0
    )


async def faithfulness_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    result = score_for_verifiers(completion, task, state)
    return result.components.get(
        "faithfulness", result.components.get("fact_preservation", 0.0)
    )


async def length_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get("length", 0.0)


async def format_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get("format", 0.0)


async def clarity_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get("clarity", 0.0)


async def placeholder_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).components.get(
        "placeholder", 0.0
    )


async def risk_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return sum(score_for_verifiers(completion, task, state).penalties.values())


async def option_menu_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).penalties.get(
        "option_menu", 0.0
    )


async def wrapper_phrase_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).penalties.get(
        "wrapper_phrase", 0.0
    )


async def invented_detail_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
) -> float:
    return score_for_verifiers(completion, task, state).penalties.get(
        "invented_detail", 0.0
    )


VERIFIER_REWARD_FUNCS = [humanize_reward]
VERIFIER_METRIC_FUNCS = [
    style_metric,
    task_following_metric,
    faithfulness_metric,
    length_metric,
    format_metric,
    clarity_metric,
    placeholder_metric,
    risk_penalty_metric,
    option_menu_penalty_metric,
    wrapper_phrase_penalty_metric,
    invented_detail_penalty_metric,
]


def build_verifiers_rubric(vf: Any) -> Any:
    """Build a Prime Verifiers rubric with reward plus zero-weight diagnostics."""
    rubric = vf.Rubric(funcs=VERIFIER_REWARD_FUNCS)
    for metric_func in VERIFIER_METRIC_FUNCS:
        rubric.add_metric(metric_func)
    return rubric
