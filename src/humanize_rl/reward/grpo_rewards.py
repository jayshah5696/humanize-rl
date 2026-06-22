"""TRL/Unsloth GRPO reward functions backed by the Humanize-RL scorer.

Reward modes (per ``docs/plans/gemma4_rl_modal_stable_training_continuation.md`` §6):

* ``current_components`` — legacy: three reward funcs (ridge, deterministic,
  risk_penalty), each dither-wrapped. Sum equals the strict reward.
  Default; preserves Slice-3 behavior.
* ``scalar_current`` — one reward func returning the same strict scalar
  (``RIDGE_WEIGHT * ridge + DET_WEIGHT * det + sum(penalties)``), clipped.
  Dither applies only to the final scalar.
* ``scalar_softened`` — one reward func returning the smoother scalar
  ``ridge_w * ridge + det_w * det + risk_w * risk_compliance`` where
  ``risk_compliance = clip(1 + sum(penalties)/penalty_cap, 0, 1)``.
* ``p50_50_no_penalty`` — one reward func returning
  ``0.50 * ridge_rubric + 0.50 * deterministic``. Penalties remain in
  diagnostics but do not affect the optimizer reward.

The ``dither_if_unanimous`` decorator guards against the
``frac_reward_zero_std`` -> NaN-grad failure documented in
``docs/plans/gemma4_rl_modal_20usd_budget_plan.md``.
"""

from __future__ import annotations

import hashlib
import random
import statistics
from collections.abc import Callable
from dataclasses import dataclass
from functools import wraps
from typing import Any, Literal

from humanize_rl.reward import diagnostics as _diag
from humanize_rl.reward.reward import (
    DETERMINISTIC_WEIGHT,
    RIDGE_WEIGHT,
    RewardResult,
    clip,
    load_ridge_scorer,
    score_response,
)
from humanize_rl.reward.tasks import RLTask

# Loaded once per worker process; None if pkl not found.
_RIDGE_SCORER = load_ridge_scorer()

Completion = list[dict[str, str]]
RewardMode = Literal[
    "current_components",
    "scalar_current",
    "scalar_softened",
    "p50_50_no_penalty",
]

DITHER_STD_THRESHOLD = 1e-4
DITHER_MAGNITUDE = 0.005

# Softened-scalar defaults (plan §6 Option C).
DEFAULT_PENALTY_CAP = 1.0
DEFAULT_RIDGE_WEIGHT_SOFT = 0.45
DEFAULT_DET_WEIGHT_SOFT = 0.35
DEFAULT_RISK_WEIGHT_SOFT = 0.20

RewardFn = Callable[..., list[float]]


@dataclass(frozen=True)
class RewardModeConfig:
    """Knobs for building reward functions.

    ``ridge_weight``/``deterministic_weight``/``risk_weight`` are only
    used by ``scalar_softened``. The other modes preserve the legacy
    50/50 strict reward exactly.
    """

    mode: RewardMode = "current_components"
    penalty_cap: float = DEFAULT_PENALTY_CAP
    ridge_weight: float = DEFAULT_RIDGE_WEIGHT_SOFT
    deterministic_weight: float = DEFAULT_DET_WEIGHT_SOFT
    risk_weight: float = DEFAULT_RISK_WEIGHT_SOFT


def _seed_from_inputs(completions: list[Completion], kwargs: dict[str, Any]) -> int:
    """Stable per-call seed so tests are deterministic."""
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
    """Decorator: if reward group std is below threshold, add tiny jitter."""

    @wraps(fn)
    def wrapped(completions: list[Completion], **kwargs: Any) -> list[float]:
        values = fn(completions, **kwargs)
        if len(values) < 2:
            return values
        std = statistics.pstdev(values)
        if std >= DITHER_STD_THRESHOLD:
            return values
        _diag.record_dither()
        rng = random.Random(_seed_from_inputs(completions, kwargs))
        return [v + rng.uniform(-DITHER_MAGNITUDE, DITHER_MAGNITUDE) for v in values]

    return wrapped


def _response(completion: Completion) -> str:
    if not completion:
        return ""
    return completion[0].get("content", "")


def _task_payloads(kwargs: dict[str, Any], count: int) -> list[dict[str, object]]:
    tasks = kwargs.get("task")
    if not isinstance(tasks, list) or len(tasks) != count:
        raise ValueError("GRPO reward functions require a `task` column per completion")
    result = []
    for task in tasks:
        if isinstance(task, str):
            import json

            result.append(json.loads(task))
        else:
            result.append(dict(task))
    return result


def score_completions(
    completions: list[Completion],
    *,
    reward_mode: str = "strict",
    **kwargs: Any,
) -> list[RewardResult]:
    """Score GRPO completions with full diagnostics.

    Side effect: increments the per-step diagnostics counters in
    :mod:`humanize_rl.reward.diagnostics`. The training-side EMA callback
    flushes them on every ``on_log``.
    """
    task_payloads = _task_payloads(kwargs, len(completions))
    results: list[RewardResult] = []
    for completion, task_payload in zip(completions, task_payloads, strict=True):
        task = RLTask.model_validate(task_payload)
        response = _response(completion)
        result = score_response(
            task,
            response,
            _RIDGE_SCORER,
            reward_mode=reward_mode,  # type: ignore[arg-type]
        )
        results.append(result)
        _diag.record_call(
            response_length=len(response),
            penalty_total=sum(result.penalties.values()),
        )
    return results


# ----------------------------------------------------------------------
# Component reward functions (legacy 3-func mode)
# ----------------------------------------------------------------------


def scalar_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    """Strict scalar reward (clipped) for diagnostics & tests."""
    return [result.reward for result in score_completions(completions, **kwargs)]


def ridge_rubric_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    """50% share: mean of 8 rubric dims from the ridge scorer."""
    return [
        result.weighted_components.get("ridge_rubric", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def deterministic_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    """50% share: mean of faithfulness, task_following, length, format, placeholder, clarity."""
    return [
        result.weighted_components.get("deterministic", 0.0)
        for result in score_completions(completions, **kwargs)
    ]


def risk_penalty_reward(completions: list[Completion], **kwargs: Any) -> list[float]:
    return [
        sum(result.penalties.values())
        for result in score_completions(completions, **kwargs)
    ]


_RAW_REWARD_FUNCS = [
    ridge_rubric_reward,
    deterministic_reward,
    risk_penalty_reward,
]

WEIGHTED_REWARD_FUNCS = [dither_if_unanimous(fn) for fn in _RAW_REWARD_FUNCS]
"""Legacy 3-function reward pack. Kept for backward compatibility."""


# ----------------------------------------------------------------------
# Scalar reward modes (plan §6 Options B & C)
# ----------------------------------------------------------------------


def _strict_scalar_from_result(result: RewardResult) -> float:
    """Strict reward exactly as ``current_components`` sums to (clipped)."""
    ridge = result.weighted_components.get("ridge_rubric", 0.0)
    det = result.weighted_components.get("deterministic", 0.0)
    pen = sum(result.penalties.values())
    return clip(ridge + det + pen)


def risk_compliance(penalty_sum: float, penalty_cap: float) -> float:
    """Smoothed compliance score in [0, 1].

    ``penalty_sum`` is the (negative-or-zero) total penalty. A value of
    0 maps to 1.0 (perfect); ``-penalty_cap`` and below maps to 0.0.
    """
    if penalty_cap <= 0:
        raise ValueError("penalty_cap must be > 0")
    return max(0.0, min(1.0, 1.0 + penalty_sum / penalty_cap))


def _softened_scalar_from_result(result: RewardResult, cfg: RewardModeConfig) -> float:
    """Softened scalar (plan §6 Option C). Returns value in [0, 1]."""
    # Unwind the legacy weighted_components back to raw 0..1 component scores
    # so the user-controlled weights apply cleanly.
    ridge_w_component = result.weighted_components.get("ridge_rubric", 0.0)
    det_w_component = result.weighted_components.get("deterministic", 0.0)
    ridge_raw = ridge_w_component / RIDGE_WEIGHT if RIDGE_WEIGHT > 0 else 0.0
    # When ridge scorer is missing, deterministic uses weight 1.0; otherwise 0.5.
    has_ridge = ridge_w_component != 0.0 or ridge_raw != 0.0
    det_divisor = DETERMINISTIC_WEIGHT if has_ridge else 1.0
    det_raw = det_w_component / det_divisor if det_divisor > 0 else 0.0
    compliance = risk_compliance(sum(result.penalties.values()), cfg.penalty_cap)
    return (
        cfg.ridge_weight * ridge_raw
        + cfg.deterministic_weight * det_raw
        + cfg.risk_weight * compliance
    )


def build_reward_funcs(cfg: RewardModeConfig) -> list[RewardFn]:
    """Factory: return TRL-compatible reward functions for ``cfg.mode``.

    * ``current_components``: legacy 3-func pack (each dither-wrapped).
    * ``scalar_current``: single strict-scalar func (dither-wrapped).
    * ``scalar_softened``: single softened-scalar func (dither-wrapped).
    """
    _diag.set_penalty_cap(cfg.penalty_cap)
    if cfg.mode == "current_components":
        return list(WEIGHTED_REWARD_FUNCS)

    if cfg.mode == "scalar_current":

        def _scalar_current(
            completions: list[Completion], **kwargs: Any
        ) -> list[float]:
            return [
                _strict_scalar_from_result(r)
                for r in score_completions(completions, **kwargs)
            ]

        _scalar_current.__name__ = "scalar_current_reward"
        return [dither_if_unanimous(_scalar_current)]

    if cfg.mode == "scalar_softened":

        def _scalar_softened(
            completions: list[Completion], **kwargs: Any
        ) -> list[float]:
            return [
                _softened_scalar_from_result(r, cfg)
                for r in score_completions(completions, **kwargs)
            ]

        _scalar_softened.__name__ = "scalar_softened_reward"
        return [dither_if_unanimous(_scalar_softened)]

    if cfg.mode == "p50_50_no_penalty":

        def _p50_50_no_penalty(
            completions: list[Completion], **kwargs: Any
        ) -> list[float]:
            return [
                result.reward
                for result in score_completions(
                    completions,
                    reward_mode="p50_50_no_penalty",
                    **kwargs,
                )
            ]

        _p50_50_no_penalty.__name__ = "p50_50_no_penalty_reward"
        return [dither_if_unanimous(_p50_50_no_penalty)]

    raise ValueError(f"unknown reward_mode: {cfg.mode!r}")
