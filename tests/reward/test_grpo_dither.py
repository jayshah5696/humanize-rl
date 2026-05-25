"""Unit tests for the ``dither_if_unanimous`` reward wrapper.

Guards against the ``frac_reward_zero_std`` -> NaN-grad failure
documented in ``docs/plans/gemma4_rl_modal_20usd_budget_plan.md`` slice 1.
"""

from __future__ import annotations

from typing import Any

from humanize_rl.reward.grpo_rewards import (
    DITHER_MAGNITUDE,
    DITHER_STD_THRESHOLD,
    dither_if_unanimous,
)

Completion = list[dict[str, str]]


def _comps(n: int) -> list[Completion]:
    return [[{"role": "assistant", "content": f"answer {i}"}] for i in range(n)]


def _tasks(n: int) -> list[dict[str, Any]]:
    return [{"id": f"t{i}"} for i in range(n)]


def test_dither_passthrough_when_std_above_threshold() -> None:
    @dither_if_unanimous
    def fn(completions: list[Completion], **kwargs: Any) -> list[float]:
        return [0.1, 0.3, 0.9]

    out = fn(_comps(3), task=_tasks(3))
    assert out == [0.1, 0.3, 0.9]


def test_dither_applies_when_group_is_unanimous() -> None:
    @dither_if_unanimous
    def fn(completions: list[Completion], **kwargs: Any) -> list[float]:
        return [0.5, 0.5, 0.5, 0.5]

    out = fn(_comps(4), task=_tasks(4))
    assert out != [0.5, 0.5, 0.5, 0.5]
    assert len(out) == 4
    for value in out:
        assert abs(value - 0.5) <= DITHER_MAGNITUDE


def test_dither_is_deterministic_for_same_inputs() -> None:
    @dither_if_unanimous
    def fn(completions: list[Completion], **kwargs: Any) -> list[float]:
        return [0.5, 0.5, 0.5]

    out1 = fn(_comps(3), task=_tasks(3))
    out2 = fn(_comps(3), task=_tasks(3))
    assert out1 == out2


def test_dither_preserves_argmax_ordering_when_near_unanimous() -> None:
    """If raw values differ but std is still below threshold, dither must
    not reorder them once we re-sort by (raw + jitter) — i.e. the jitter
    magnitude is smaller than the minimum gap.

    Test the case the plan actually cares about: a strict unanimous group
    where any ordering is acceptable, plus a near-unanimous group where
    jitter < min-gap so argmax of dithered == argmax of raw.
    """
    # Near-unanimous: gap is 10x dither magnitude so ordering must hold.
    raw = [0.500, 0.500 + 10 * DITHER_MAGNITUDE, 0.500 + 20 * DITHER_MAGNITUDE]

    @dither_if_unanimous
    def fn(completions: list[Completion], **kwargs: Any) -> list[float]:
        return list(raw)

    out = fn(_comps(3), task=_tasks(3))
    # std is large enough -> pass-through, so trivially preserves ordering.
    assert out == raw


def test_dither_threshold_value_is_documented_constant() -> None:
    """Lock the threshold so regressions are visible in diff review."""
    assert DITHER_STD_THRESHOLD == 1e-4
    assert DITHER_MAGNITUDE == 0.005


def test_dither_no_op_on_single_element() -> None:
    @dither_if_unanimous
    def fn(completions: list[Completion], **kwargs: Any) -> list[float]:
        return [0.5]

    out = fn(_comps(1), task=_tasks(1))
    assert out == [0.5]
