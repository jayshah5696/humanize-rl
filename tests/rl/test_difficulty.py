"""Tests for offline difficulty bucketing (Slice 4).

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 4.

The pure library in ``humanize_rl.rl.difficulty`` is tested without
Modal/vLLM. We build synthetic ``Rollout`` lists that exercise every
bucket and the filter helper.
"""

from __future__ import annotations

import pytest

from humanize_rl.reward.reward import RewardResult
from humanize_rl.rl.difficulty import (
    DifficultyThresholds,
    Rollout,
    filter_task_mix,
    summarize_rollouts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rollout(
    *,
    reward: float = 0.5,
    ridge: float = 0.25,
    det: float = 0.25,
    penalty_sum: float = 0.0,
    penalty_names: tuple[str, ...] = (),
    length: int = 60,
    clipped: bool = False,
    response: str = "ok",
) -> Rollout:
    return Rollout(
        response=response,
        response_length=length,
        reward=reward,
        ridge_rubric=ridge,
        deterministic=det,
        penalty_sum=penalty_sum,
        penalty_names=penalty_names,
        clipped=clipped,
    )


# ---------------------------------------------------------------------------
# Rollout.from_result
# ---------------------------------------------------------------------------


def test_rollout_from_result_extracts_weighted_components_and_penalties() -> None:
    result = RewardResult(
        reward=0.55,
        raw_reward=0.55,
        components={},
        weighted_components={"ridge_rubric": 0.30, "deterministic": 0.25},
        penalties={"wrapper_phrase": -0.20, "ai_tell_phrase": -0.25},
    )
    ro = Rollout.from_result("hello world", result, clipped=False)
    assert ro.reward == pytest.approx(0.55)
    assert ro.ridge_rubric == pytest.approx(0.30)
    assert ro.deterministic == pytest.approx(0.25)
    assert ro.penalty_sum == pytest.approx(-0.45)
    assert ro.penalty_names == ("ai_tell_phrase", "wrapper_phrase")  # sorted
    assert ro.response_length == len("hello world")


# ---------------------------------------------------------------------------
# summarize_rollouts \u2014 buckets
# ---------------------------------------------------------------------------


def test_useful_bucket_when_reward_in_band_and_variance_present() -> None:
    rollouts = [
        _rollout(reward=0.30, penalty_names=("a",)),
        _rollout(reward=0.55, penalty_names=("b",)),
        _rollout(reward=0.70, penalty_names=()),
    ]
    diff = summarize_rollouts("t1", rollouts)
    assert diff.bucket == "useful"
    assert diff.kept is True
    assert diff.reward_mean == pytest.approx(0.5166666, abs=1e-4)


def test_too_easy_when_mean_at_or_above_max() -> None:
    rollouts = [_rollout(reward=0.85), _rollout(reward=0.90), _rollout(reward=0.95)]
    diff = summarize_rollouts("t_easy", rollouts)
    assert diff.bucket == "too_easy"
    assert not diff.kept
    assert any("reward_mean" in r for r in diff.reasons)


def test_too_hard_when_mean_at_or_below_min() -> None:
    rollouts = [_rollout(reward=0.10), _rollout(reward=0.15), _rollout(reward=0.20)]
    diff = summarize_rollouts("t_hard", rollouts)
    assert diff.bucket == "too_hard"
    assert not diff.kept


def test_dead_when_reward_std_below_threshold() -> None:
    """Strict zero variance \u2014 every reward identical."""
    rollouts = [_rollout(reward=0.50) for _ in range(8)]
    diff = summarize_rollouts("t_dead", rollouts)
    assert diff.bucket == "dead"
    assert diff.reward_std == pytest.approx(0.0)


def test_clipped_dominates_other_buckets() -> None:
    """If too many completions are length-clipped, that diagnoses first."""
    rollouts = [_rollout(reward=0.5, clipped=True) for _ in range(8)]
    diff = summarize_rollouts("t_clip", rollouts)
    assert diff.bucket == "clipped"
    assert diff.clipped_rate == pytest.approx(1.0)


def test_same_pattern_when_all_share_penalty_names() -> None:
    rollouts = [
        _rollout(reward=0.30, penalty_names=("wrapper_phrase",)),
        _rollout(reward=0.40, penalty_names=("wrapper_phrase",)),
        _rollout(reward=0.55, penalty_names=("wrapper_phrase",)),
    ]
    diff = summarize_rollouts("t_pattern", rollouts)
    assert diff.bucket == "same_pattern"
    assert diff.penalty_pattern_diversity == 1


def test_check_same_pattern_can_be_disabled_for_softened_reward() -> None:
    """Plan §6 Option C: under softened reward, same penalty pattern is
    not a useful exclusion signal because ridge dim variation still
    differentiates rollouts. Caller can opt out of the same_pattern rule."""
    rollouts = [
        _rollout(reward=0.30, penalty_names=("wrapper_phrase",)),
        _rollout(reward=0.45, penalty_names=("wrapper_phrase",)),
        _rollout(reward=0.55, penalty_names=("wrapper_phrase",)),
    ]
    strict = summarize_rollouts("t_pattern", rollouts)
    permissive = summarize_rollouts(
        "t_pattern",
        rollouts,
        DifficultyThresholds(check_same_pattern=False),
    )
    assert strict.bucket == "same_pattern"
    assert permissive.bucket == "useful"
    assert permissive.kept


def test_useful_when_penalty_patterns_differ() -> None:
    rollouts = [
        _rollout(reward=0.30, penalty_names=("wrapper_phrase",)),
        _rollout(reward=0.55, penalty_names=("ai_tell_phrase",)),
        _rollout(reward=0.65, penalty_names=()),
    ]
    diff = summarize_rollouts("t_div", rollouts)
    assert diff.bucket == "useful"
    assert diff.penalty_pattern_diversity == 3


def test_penalty_rate_counts_only_negative_penalty_sums() -> None:
    rollouts = [
        _rollout(reward=0.30, penalty_sum=0.0),
        _rollout(reward=0.55, penalty_sum=-0.2),
        _rollout(reward=0.65, penalty_sum=-0.3),
    ]
    diff = summarize_rollouts("t_pen", rollouts)
    assert diff.penalty_rate == pytest.approx(2 / 3)


def test_thresholds_are_applied() -> None:
    """Tighten min_reward and a borderline-useful task becomes too_hard."""
    rollouts = [
        _rollout(reward=0.22, penalty_names=("a",)),
        _rollout(reward=0.30, penalty_names=("b",)),
        _rollout(reward=0.38, penalty_names=("c",)),
    ]
    relaxed = summarize_rollouts("t_b", rollouts)
    assert relaxed.bucket == "useful", relaxed.reasons
    strict = summarize_rollouts(
        "t_b",
        rollouts,
        DifficultyThresholds(min_reward=0.35),
    )
    assert strict.bucket == "too_hard", strict.reasons


def test_response_length_p95_handles_skewed_distribution() -> None:
    lengths = [10, 12, 14, 16, 18, 20, 22, 200]  # one outlier
    rollouts = [_rollout(reward=0.4 + i * 0.01, length=ln, penalty_names=(f"p{i}",))
                for i, ln in enumerate(lengths)]
    diff = summarize_rollouts("t_len", rollouts)
    assert diff.response_length_p95 == 200.0
    assert diff.response_length_mean < 50  # not skewed by the outlier


def test_empty_rollouts_raises() -> None:
    with pytest.raises(ValueError):
        summarize_rollouts("t_empty", [])


# ---------------------------------------------------------------------------
# filter_task_mix
# ---------------------------------------------------------------------------


def _summary(task_id: str, *, bucket: str, kept: bool):
    from humanize_rl.rl.difficulty import TaskDifficulty
    return TaskDifficulty(
        task_id=task_id,
        n_completions=8,
        reward_mean=0.5,
        reward_std=0.1,
        ridge_mean=0.25,
        ridge_std=0.05,
        det_mean=0.25,
        det_std=0.05,
        penalty_rate=0.3,
        penalty_pattern_diversity=4,
        response_length_mean=80,
        response_length_p95=120,
        clipped_rate=0.0,
        bucket=bucket,  # type: ignore[arg-type]
        kept=kept,
    )


def test_filter_task_mix_keeps_only_useful_and_overwrites_difficulty_fields() -> None:
    mix = [
        {"id": "t1", "dataset_version": "v01", "difficulty_bucket": "unknown"},
        {"id": "t2", "dataset_version": "v02", "difficulty_bucket": "unknown"},
        {"id": "t3", "dataset_version": "v02", "difficulty_bucket": "unknown"},
    ]
    diffs = {
        "t1": _summary("t1", bucket="useful", kept=True),
        "t2": _summary("t2", bucket="too_easy", kept=False),
        "t3": _summary("t3", bucket="dead", kept=False),
    }
    kept, drops = filter_task_mix(mix, diffs)
    assert [row["id"] for row in kept] == ["t1"]
    assert kept[0]["difficulty_bucket"] == "useful"
    assert "difficulty" in kept[0]
    assert drops == {"too_easy": 1, "dead": 1}


def test_filter_task_mix_drops_unscored_tasks() -> None:
    mix = [{"id": "t1", "dataset_version": "v01", "difficulty_bucket": "unknown"}]
    kept, drops = filter_task_mix(mix, {})
    assert kept == []
    assert drops == {"unscored": 1}


def test_filter_task_mix_whitelist_overrides_drop() -> None:
    mix = [{"id": "t1", "dataset_version": "v01", "difficulty_bucket": "unknown"}]
    diffs = {"t1": _summary("t1", bucket="too_easy", kept=False)}
    kept, drops = filter_task_mix(mix, diffs, whitelist={"t1"})
    assert len(kept) == 1
    # Even when kept-by-whitelist, the bucket metadata reflects reality.
    assert kept[0]["difficulty_bucket"] == "too_easy"
    assert drops == {}
