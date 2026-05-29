"""Per-task difficulty bucketing for Prime-style offline filtering.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 4.

Given K scored rollouts per task, decide whether the task is useful for
GRPO. Buckets:

* ``useful``      \u2014 keep for training
* ``too_easy``    \u2014 mean reward >= max_reward (no headroom)
* ``too_hard``    \u2014 mean reward <= min_reward (signal too sparse)
* ``dead``        \u2014 reward_std < min_reward_std (zero-variance group)
* ``clipped``     \u2014 completions clipped > max_clipped_rate
* ``same_pattern`` \u2014 every completion shares the same penalty pattern

The pure functions in this module are deliberately decoupled from
generation (vLLM lives in the Modal entrypoint) so unit tests run
without GPU.
"""

from __future__ import annotations

import statistics
from collections import Counter
from dataclasses import dataclass, field
from typing import Literal

from humanize_rl.reward.reward import RewardResult

DifficultyBucket = Literal[
    "useful",
    "too_easy",
    "too_hard",
    "dead",
    "clipped",
    "same_pattern",
]


@dataclass(frozen=True)
class DifficultyThresholds:
    """Defaults match plan \u00a78 Ablation F."""

    min_reward: float = 0.20
    max_reward: float = 0.80
    min_reward_std: float = 0.03
    max_clipped_rate: float = 0.10
    # When True, mark tasks where every rollout shares the same penalty
    # pattern as ``same_pattern``. Sensible under strict reward; under
    # ``scalar_softened`` ridge dim variation still provides signal so
    # callers may want to disable this.
    check_same_pattern: bool = True


DEFAULT_THRESHOLDS = DifficultyThresholds()


@dataclass
class Rollout:
    """One scored completion."""

    response: str
    response_length: int
    reward: float
    ridge_rubric: float
    deterministic: float
    penalty_sum: float
    penalty_names: tuple[str, ...]
    clipped: bool

    @classmethod
    def from_result(
        cls, response: str, result: RewardResult, clipped: bool
    ) -> Rollout:
        penalty_names = tuple(sorted(result.penalties.keys()))
        return cls(
            response=response,
            response_length=len(response),
            reward=result.reward,
            ridge_rubric=result.weighted_components.get("ridge_rubric", 0.0),
            deterministic=result.weighted_components.get("deterministic", 0.0),
            penalty_sum=sum(result.penalties.values()),
            penalty_names=penalty_names,
            clipped=clipped,
        )


@dataclass(frozen=True)
class TaskDifficulty:
    task_id: str
    n_completions: int
    reward_mean: float
    reward_std: float
    ridge_mean: float
    ridge_std: float
    det_mean: float
    det_std: float
    penalty_rate: float
    penalty_pattern_diversity: int
    response_length_mean: float
    response_length_p95: float
    clipped_rate: float
    bucket: DifficultyBucket
    kept: bool
    reasons: tuple[str, ...] = field(default_factory=tuple)


def _pstd(values: list[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    rank = max(0, min(len(s) - 1, int(round(0.95 * (len(s) - 1)))))
    return float(s[rank])


def summarize_rollouts(
    task_id: str,
    rollouts: list[Rollout],
    thresholds: DifficultyThresholds = DEFAULT_THRESHOLDS,
) -> TaskDifficulty:
    if not rollouts:
        raise ValueError(f"no rollouts for task {task_id}")

    rewards = [r.reward for r in rollouts]
    ridges = [r.ridge_rubric for r in rollouts]
    dets = [r.deterministic for r in rollouts]
    pens = [r.penalty_sum for r in rollouts]
    lengths = [float(r.response_length) for r in rollouts]
    patterns = Counter(r.penalty_names for r in rollouts)
    clipped_rate = sum(1 for r in rollouts if r.clipped) / len(rollouts)
    penalty_rate = sum(1 for p in pens if p < 0.0) / len(rollouts)

    reward_mean = statistics.fmean(rewards)
    reward_std = _pstd(rewards)

    reasons: list[str] = []
    bucket: DifficultyBucket = "useful"
    if clipped_rate > thresholds.max_clipped_rate:
        bucket = "clipped"
        reasons.append(f"clipped_rate={clipped_rate:.2f}>max={thresholds.max_clipped_rate}")
    elif reward_std < thresholds.min_reward_std:
        bucket = "dead"
        reasons.append(f"reward_std={reward_std:.3f}<min={thresholds.min_reward_std}")
    elif reward_mean <= thresholds.min_reward:
        bucket = "too_hard"
        reasons.append(f"reward_mean={reward_mean:.3f}<=min={thresholds.min_reward}")
    elif reward_mean >= thresholds.max_reward:
        bucket = "too_easy"
        reasons.append(f"reward_mean={reward_mean:.3f}>=max={thresholds.max_reward}")
    elif (
        thresholds.check_same_pattern
        and len(rollouts) >= 2
        and len(patterns) == 1
    ):
        bucket = "same_pattern"
        reasons.append(f"all_{len(rollouts)}_rollouts_share_penalties={list(patterns)[0]}")

    return TaskDifficulty(
        task_id=task_id,
        n_completions=len(rollouts),
        reward_mean=reward_mean,
        reward_std=reward_std,
        ridge_mean=statistics.fmean(ridges),
        ridge_std=_pstd(ridges),
        det_mean=statistics.fmean(dets),
        det_std=_pstd(dets),
        penalty_rate=penalty_rate,
        penalty_pattern_diversity=len(patterns),
        response_length_mean=statistics.fmean(lengths),
        response_length_p95=_p95(lengths),
        clipped_rate=clipped_rate,
        bucket=bucket,
        kept=(bucket == "useful"),
        reasons=tuple(reasons),
    )


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------


def filter_task_mix(
    mix_rows: list[dict],
    difficulties: dict[str, TaskDifficulty],
    whitelist: set[str] | None = None,
) -> tuple[list[dict], dict[str, int]]:
    """Apply difficulty bucketing to a Slice-3 mix.

    Returns (filtered_rows, drop_counts_by_bucket).
    Always preserves rows whose ``task_id`` is in ``whitelist``.
    """
    whitelist = whitelist or set()
    kept: list[dict] = []
    drops: Counter[str] = Counter()
    for row in mix_rows:
        tid = row["id"]
        diff = difficulties.get(tid)
        if diff is None:
            # No rollouts \u2014 conservative: drop and count as "unscored".
            drops["unscored"] += 1
            continue
        row_copy = dict(row)
        row_copy["difficulty_bucket"] = diff.bucket
        row_copy["difficulty"] = {
            "reward_mean": diff.reward_mean,
            "reward_std": diff.reward_std,
            "penalty_rate": diff.penalty_rate,
            "clipped_rate": diff.clipped_rate,
            "penalty_pattern_diversity": diff.penalty_pattern_diversity,
            "response_length_mean": diff.response_length_mean,
            "response_length_p95": diff.response_length_p95,
            "n_completions": diff.n_completions,
        }
        if diff.kept or tid in whitelist:
            kept.append(row_copy)
        else:
            drops[diff.bucket] += 1
    return kept, dict(drops)
