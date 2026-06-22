"""Prime Verifiers rubric adapter for the Humanize-RL reward scorer."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from humanize_rl.reward.reward import (
    RewardResult,
    load_ridge_scorer,
    score_response,
)
from humanize_rl.reward.tasks import RLTask

# Loaded once at import time; None if pkl not found (graceful degradation).
_RIDGE_SCORER = load_ridge_scorer()

Completion = list[Any]
State = dict[str, Any]

REWARD_STATE_KEY = "humanize_reward_result"
SUPPORTED_REWARD_MODES = {"strict", "scalar_softened", "p50_50_no_penalty"}


def _content_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for part in value:
            if isinstance(part, Mapping):
                text = part.get("text", part.get("content", ""))
            else:
                text = getattr(part, "text", getattr(part, "content", ""))
            if text is not None:
                parts.append(str(text))
        return "".join(parts)
    return str(value)


def response_from_completion(completion: Completion | str | None) -> str:
    """Extract assistant text from a Verifiers completion message list."""
    if not completion:
        return ""
    if isinstance(completion, str):
        return completion
    last = completion[-1]
    if last is None:
        return ""
    if isinstance(last, Mapping):
        return _content_to_text(last.get("content"))
    return _content_to_text(getattr(last, "content", None))


def _state_key(reward_mode: str) -> str:
    return (
        REWARD_STATE_KEY
        if reward_mode == "strict"
        else f"{REWARD_STATE_KEY}:{reward_mode}"
    )


def _result_from_state(state: State, reward_mode: str) -> RewardResult | None:
    result = state.get(_state_key(reward_mode))
    if isinstance(result, RewardResult):
        return result
    return None


def _parse_json_mapping(value: Any) -> dict[str, object] | None:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if isinstance(value, Mapping):
        return dict(value)
    return None


def _task_from_info(info: Any) -> dict[str, object] | None:
    parsed = _parse_json_mapping(info)
    if parsed is None:
        return None
    return _parse_json_mapping(parsed.get("task"))


def _maybe_parse_task(task: Any) -> dict[str, object] | None:
    if isinstance(task, str):
        return _parse_json_mapping(task)
    if isinstance(task, Mapping):
        if "id" in task and "instruction" in task:
            return dict(task)
        nested = _parse_json_mapping(task.get("task"))
        if nested is not None:
            return nested
        info_task = _task_from_info(task.get("info"))
        if info_task is not None:
            return info_task
        input_obj = task.get("input")
        if isinstance(input_obj, Mapping):
            return _maybe_parse_task(input_obj)
    return None


def _parse_task(
    task: dict[str, object] | str | None, state: State | None = None
) -> dict[str, object]:
    """Accept raw task payloads and Verifiers 0.1.14 state-shaped inputs."""
    parsed = _maybe_parse_task(task)
    if parsed is not None:
        return parsed
    if state is not None:
        state_task = _maybe_parse_task(state.get("task"))
        if state_task is not None:
            return state_task
        state_info_task = _task_from_info(state.get("info"))
        if state_info_task is not None:
            return state_info_task
        input_obj = state.get("input")
        if isinstance(input_obj, Mapping):
            input_task = _maybe_parse_task(input_obj)
            if input_task is not None:
                return input_task
    raise ValueError("Verifiers task payload is missing RL task fields")


def score_for_verifiers(
    completion: Completion | None,
    task: dict[str, object] | str | None,
    state: State | None = None,
    reward_mode: str = "strict",
) -> RewardResult:
    """Score a Verifiers rollout and optionally cache diagnostics in state."""
    if reward_mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"unknown reward_mode: {reward_mode!r}")

    if state is not None:
        cached = _result_from_state(state, reward_mode)
        if cached is not None:
            return cached

    result = score_response(
        RLTask.model_validate(_parse_task(task, state)),
        response_from_completion(completion),
        _RIDGE_SCORER,
        reward_mode=reward_mode,  # type: ignore[arg-type]
    )
    if state is not None:
        state[_state_key(reward_mode)] = result
        state[REWARD_STATE_KEY] = result
        state["humanize_reward"] = result.reward
        state["humanize_reward_mode"] = reward_mode
        state["humanize_components"] = result.components
        state["humanize_penalties"] = result.penalties
        state["humanize_diagnostics"] = result.diagnostics
    return result


async def humanize_reward(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    """Primary scalar reward for Prime Verifiers training/eval."""
    return score_for_verifiers(completion, task, state, reward_mode).reward


async def style_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "style", 0.0
    )


async def task_following_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "task_following", 0.0
    )


async def faithfulness_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    result = score_for_verifiers(completion, task, state, reward_mode)
    return result.components.get(
        "faithfulness", result.components.get("fact_preservation", 0.0)
    )


async def length_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "length", 0.0
    )


async def format_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "format", 0.0
    )


async def clarity_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "clarity", 0.0
    )


async def placeholder_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).components.get(
        "placeholder", 0.0
    )


async def risk_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return sum(
        score_for_verifiers(completion, task, state, reward_mode).penalties.values()
    )


async def option_menu_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).penalties.get(
        "option_menu", 0.0
    )


async def wrapper_phrase_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).penalties.get(
        "wrapper_phrase", 0.0
    )


async def invented_detail_penalty_metric(
    completion: Completion | None,
    task: dict[str, object],
    state: State,
    reward_mode: str = "strict",
) -> float:
    return score_for_verifiers(completion, task, state, reward_mode).penalties.get(
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


def _bind_reward_mode(metric_func: Any, reward_mode: str) -> Any:
    if reward_mode == "strict":
        return metric_func

    async def _wrapped(
        completion: Completion | None, task: dict[str, object], state: State
    ) -> float:
        return await metric_func(completion, task, state, reward_mode=reward_mode)

    _wrapped.__name__ = f"{metric_func.__name__}_{reward_mode}"
    return _wrapped


def build_verifiers_rubric(vf: Any, reward_mode: str = "strict") -> Any:
    """Build a Prime Verifiers rubric with reward plus zero-weight diagnostics."""
    if reward_mode not in SUPPORTED_REWARD_MODES:
        raise ValueError(f"unknown reward_mode: {reward_mode!r}")
    reward_funcs = [
        _bind_reward_mode(func, reward_mode) for func in VERIFIER_REWARD_FUNCS
    ]
    rubric = vf.Rubric(funcs=reward_funcs)
    for metric_func in VERIFIER_METRIC_FUNCS:
        rubric.add_metric(_bind_reward_mode(metric_func, reward_mode))
    return rubric
