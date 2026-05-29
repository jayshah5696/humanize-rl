"""Slice 1 tests for reward-mode refactor.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md \u00a76 / Slice 1.

Guarantees:
  1. ``current_components`` default is unchanged (sum equals strict scalar).
  2. ``scalar_current`` returns the same strict scalar in one func.
  3. ``scalar_softened`` is bounded [0, 1] and monotonic in penalties.
  4. Dither applies only to the final scalar in scalar modes.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from humanize_rl.reward.grpo_rewards import (
    DITHER_MAGNITUDE,
    DITHER_STD_THRESHOLD,
    RewardModeConfig,
    build_reward_funcs,
    risk_compliance,
    scalar_reward,
    score_completions,
)
from humanize_rl.reward.reward import clip
from humanize_rl.reward.tasks import RLTask

Completion = list[dict[str, str]]


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


def _payload() -> dict[str, Any]:
    return _task().model_dump(by_alias=True, exclude_none=True)


def _good_completion() -> Completion:
    return [
        {
            "role": "assistant",
            "content": "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix.",
        }
    ]


def _bad_completion() -> Completion:
    # Drops required facts + adds a wrapper phrase trap.
    return [{"role": "assistant", "content": "Here's a version: Sarah fixed it."}]


# ---------------------------------------------------------------------------
# Guarantee 1: default mode unchanged
# ---------------------------------------------------------------------------


def test_current_components_default_sum_equals_strict_scalar() -> None:
    payload = _payload()
    completions = [_good_completion(), _bad_completion()]
    task_col = [payload, payload]

    funcs = build_reward_funcs(RewardModeConfig())  # default mode
    per_func = [func(completions, task=task_col) for func in funcs]

    assert len(funcs) == 3, "current_components must expose 3 reward funcs"
    summed = [sum(vals) for vals in zip(*per_func, strict=True)]
    strict = scalar_reward(completions, task=task_col)
    for s, t in zip(summed, strict, strict=True):
        # strict is clipped to [-1, 1]; summed is the raw composition
        assert math.isclose(clip(s), t, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Guarantee 2: scalar_current equals strict scalar
# ---------------------------------------------------------------------------


def test_scalar_current_matches_legacy_strict_reward() -> None:
    payload = _payload()
    completions = [_good_completion(), _bad_completion()]
    task_col = [payload, payload]

    funcs = build_reward_funcs(RewardModeConfig(mode="scalar_current"))
    assert len(funcs) == 1
    got = funcs[0](completions, task=task_col)
    expected = scalar_reward(completions, task=task_col)

    # Both are clipped to [-1, 1]; equal up to fp noise (no dither needed
    # since the two completions differ enough to exceed the std threshold).
    for g, e in zip(got, expected, strict=True):
        assert math.isclose(g, e, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# Guarantee 3: scalar_softened is bounded & penalty-monotonic
# ---------------------------------------------------------------------------


def test_risk_compliance_endpoints_and_monotonic() -> None:
    assert risk_compliance(0.0, penalty_cap=1.0) == 1.0
    assert risk_compliance(-1.0, penalty_cap=1.0) == 0.0
    assert risk_compliance(-2.0, penalty_cap=1.0) == 0.0  # clipped
    assert 0.0 < risk_compliance(-0.5, penalty_cap=1.0) < 1.0
    with pytest.raises(ValueError):
        risk_compliance(0.0, penalty_cap=0.0)


def test_scalar_softened_is_bounded_unit_interval() -> None:
    payload = _payload()
    completions = [_good_completion(), _bad_completion()]
    task_col = [payload, payload]

    cfg = RewardModeConfig(
        mode="scalar_softened",
        penalty_cap=1.0,
        ridge_weight=0.45,
        deterministic_weight=0.35,
        risk_weight=0.20,
    )
    funcs = build_reward_funcs(cfg)
    assert len(funcs) == 1
    values = funcs[0](completions, task=task_col)
    for v in values:
        # 0.45 + 0.35 + 0.20 = 1.0 ; each underlying component is in [0, 1].
        assert 0.0 <= v <= 1.0 + 1e-9, f"softened reward out of [0,1]: {v}"


def test_scalar_softened_higher_for_better_response() -> None:
    payload = _payload()
    cfg = RewardModeConfig(mode="scalar_softened")
    fn = build_reward_funcs(cfg)[0]
    good = fn([_good_completion()], task=[payload])[0]
    bad = fn([_bad_completion()], task=[payload])[0]
    assert good > bad


def test_scalar_softened_monotonic_in_penalty_total() -> None:
    """As total penalty becomes more negative, risk_compliance must fall."""
    # Hand-build two RewardResult-equivalent calls through risk_compliance
    # directly: the scalar_softened function is just a weighted sum, so the
    # penalty-monotonicity test reduces to risk_compliance monotonicity.
    base = risk_compliance(0.0, penalty_cap=1.0)
    mid = risk_compliance(-0.3, penalty_cap=1.0)
    deep = risk_compliance(-0.8, penalty_cap=1.0)
    assert base > mid > deep


# ---------------------------------------------------------------------------
# Guarantee 4: dither in scalar modes
# ---------------------------------------------------------------------------


def test_scalar_mode_dithers_unanimous_group() -> None:
    """Identical completions across a group must trigger dither (std < threshold)
    so GRPO advantages do not collapse to zero."""
    payload = _payload()
    same = _good_completion()
    completions = [same, same, same, same]
    task_col = [payload] * 4

    fn = build_reward_funcs(RewardModeConfig(mode="scalar_current"))[0]
    values = fn(completions, task=task_col)

    # Verify dither was applied: at least one pair must differ.
    distinct = len({round(v, 9) for v in values})
    assert distinct > 1, "dither must perturb a unanimous group"
    # And the perturbation must stay within the documented magnitude.
    raw = score_completions(completions, task=task_col)[0].reward
    for v in values:
        assert abs(v - raw) <= DITHER_MAGNITUDE + 1e-9


def test_softened_mode_dither_threshold_is_documented_constant() -> None:
    """Lock the threshold so regressions are visible in diff review."""
    assert DITHER_STD_THRESHOLD == 1e-4
    assert DITHER_MAGNITUDE == 0.005


# ---------------------------------------------------------------------------
# Guarantee 5: unknown mode raises early
# ---------------------------------------------------------------------------


def test_unknown_reward_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown reward_mode"):
        build_reward_funcs(RewardModeConfig(mode="not_a_real_mode"))  # type: ignore[arg-type]
