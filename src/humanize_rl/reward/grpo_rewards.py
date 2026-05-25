"""TRL/Unsloth GRPO reward functions backed by the Humanize-RL scorer.

The ``dither_if_unanimous`` decorator guards against the
``frac_reward_zero_std`` -> NaN-grad failure documented in
``docs/plans/gemma4_rl_modal_20usd_budget_plan.md``. TRL invokes each
reward function once per GRPO group (size ``num_generations``); if the
returned list has std < ``DITHER_STD_THRESHOLD`` the group's advantages
collapse to ~0 and the bf16 backward path produces NaN. We add a tiny,
deterministic per-call jitter only on those degenerate groups; non-zero
std groups pass through unchanged.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Callable
from functools import wraps
from typing import Any

from humanize_rl.reward.reward import RewardResult, score_response
from humanize_rl.reward.tasks import RLTask

Completion = list[dict[str, str]]

DITHER_STD_THRESHOLD = 1e-4
DITHER_MAGNITUDE = 0.005

RewardFn = Callable[..., list[float]]


def _seed_from_inputs(completions: list[Completion], kwargs: dict[str, Any]) -> int:
    """Stable per-call seed so tests are deterministic.

    Hashes the textual completions plus task ids (when present).
    """
    h = hashlib.sha256()
    for completion in completions:
        h.update(b"\x1f")
        for message in completion:
            h.update(str(message.get("content", "")).encode("utf-8", errors="replace"))
    tasks = kwargs.get("task") or []
    for task in tasks:
        if isinstance(task, dict):
            h.update(str(task.get("id", "")).encode("utf-8", errors="replace"))
    return int.from_bytes(h.digest()[:8], "big")


def dither_if_unanimous(fn: RewardFn) -> RewardFn:
    """Decorator: if reward group std is below threshold, add tiny jitter.

    Pass-through when std >= ``DITHER_STD_THRESHOLD``. Deterministic per
    invocation (seeded from a hash of completions + task ids).
    """

    @wraps(fn)
    def wrapped(completions: list[Completion], **kwargs: Any) -> list[float]:
        values = fn(completions, **kwargs)
        if len(values) < 2:
            return values
        std = statistics.pstdev(values)
        if std >= DITHER_STD_THRESHOLD:
            return values
        rng = random.Random(_seed_from_inputs(completions, kwargs))
        return [
            v + rng.uniform(-DITHER_MAGNITUDE, DITHER_MAGNITUDE) for v in values
        ]

    return wrapped


def _response(completion: Completion) -> str:
    if not completion:
        return ""
    return completion[0].get("content", "")


def _task_payloads(kwargs: dict[str, Any], count: int) -> list[dict[str, object]]:
    tasks = kwargs.get("task")
    if not isinstance(tasks, list) or len(tasks) != count:
        raise ValueError("GRPO reward functions require a `task` column per completion")
    return [dict(task) for task in tasks]


def score_completions(
    completions: list[Completion], **kwargs: Any
) -> list[RewardResult]:
    """Score GRPO completions with full diagnostics."""
    task_payloads = _task_payloads(kwargs, len(completions))
    results: list[RewardResult] = []
    for completion, task_payload in zip(completions, task_payloads, strict=True):
        task = RLTask.model_validate(task_payload)
        results.append(score_response(task, _response(completion)))
    return results


def scalar_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [result.reward for result in score_completions(completions, **kwargs)]


def weighted_style_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [
        result.weighted_components.get("style", 0.0)
        + result.weighted_components.get("naturalness", 0.0)
        + result.weighted_components.get("tone_appropriateness", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def weighted_task_following_reward(
    completions: list[Completion], **kwargs: Any
) -> list[float]:
    return [
        result.weighted_components.get("task_following", 0.0)
        + result.weighted_components.get("correctness_adherence", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def weighted_faithfulness_reward(
    completions: list[Completion], **kwargs: Any
) -> list[float]:
    return [
        result.weighted_components.get("faithfulness", 0.0)
        + result.weighted_components.get("fact_preservation", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def weighted_length_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [
        result.weighted_components.get("length", 0.0)
        + result.weighted_components.get("concision", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def weighted_format_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [
        result.weighted_components.get("format", 0.0)
        + result.weighted_components.get("structure_restraint", 0.0)
        + result.weighted_components.get("placeholder", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def weighted_clarity_reward(
    completions: list[Completion], **kwargs: Any
) -> list[float]:
    return [
        result.weighted_components.get("clarity", 0.0)
        + result.weighted_components.get("no_corporate_filler", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def risk_penalty_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [
        sum(result.penalties.values())
        for result in score_completions(completions, **kwargs)
    ]


_RAW_REWARD_FUNCS = [
    weighted_style_reward,
    weighted_task_following_reward,
    weighted_faithfulness_reward,
    weighted_length_reward,
    weighted_format_reward,
    weighted_clarity_reward,
    risk_penalty_reward,
]

WEIGHTED_REWARD_FUNCS = [dither_if_unanimous(fn) for fn in _RAW_REWARD_FUNCS]
