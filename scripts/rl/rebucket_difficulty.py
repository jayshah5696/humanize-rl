#!/usr/bin/env python3
"""Re-bucket Slice 4 rollouts under alternative reward modes.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md §6 / Slice 4 follow-up.

Reads ``rollouts.jsonl`` from a prior Slice 4 run and re-scores each
rollout under every supported ``RewardMode`` using only the stored
``ridge_rubric`` / ``deterministic`` / ``penalty_sum`` fields (no model,
no ridge scorer needed). Then re-runs the bucketing pipeline and writes:

  * ``rebucket_<mode>.jsonl`` (per-task summaries)
  * ``rebucket_summary.json`` (bucket-count comparison across modes)

Run::

  uv run python scripts/rl/rebucket_difficulty.py \\
    --rollouts outputs/rl_difficulty/mix_v1/rollouts.jsonl \\
    --mix data/rl/humanize_tasks_rl_mix_v1.jsonl \\
    --output-dir outputs/rl_difficulty/mix_v1
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import click

from humanize_rl.reward.grpo_rewards import (
    DEFAULT_DET_WEIGHT_SOFT,
    DEFAULT_PENALTY_CAP,
    DEFAULT_RIDGE_WEIGHT_SOFT,
    DEFAULT_RISK_WEIGHT_SOFT,
    risk_compliance,
)
from humanize_rl.reward.reward import DETERMINISTIC_WEIGHT, RIDGE_WEIGHT, clip
from humanize_rl.rl.difficulty import (
    DEFAULT_THRESHOLDS,
    DifficultyThresholds,
    Rollout,
    filter_task_mix,
    summarize_rollouts,
)

# ``scalar_softened_permissive`` is the same softened reward but with
# lower min_reward_std and ``check_same_pattern=False``. It is what
# Slice 4 follow-up analysis recommends actually using for training.
MODES = (
    "current_components",
    "scalar_current",
    "scalar_softened",
    "scalar_softened_permissive",
)


def _recompute_reward(
    mode: str,
    ridge_weighted: float,
    det_weighted: float,
    penalty_sum: float,
    *,
    ridge_w: float,
    det_w: float,
    risk_w: float,
    penalty_cap: float,
) -> float:
    """Re-derive the reward under ``mode`` from stored per-rollout fields.

    ``ridge_weighted`` / ``det_weighted`` are the values already multiplied
    by the legacy RIDGE_WEIGHT / DETERMINISTIC_WEIGHT (50/50). We unwind
    to raw [0, 1] components for the softened formulation.
    """
    if mode in ("current_components", "scalar_current"):
        # Both produce the same scalar; current_components sums 3 funcs,
        # scalar_current returns the sum directly. Both clip to [-1, 1].
        return clip(ridge_weighted + det_weighted + penalty_sum)
    if mode in ("scalar_softened", "scalar_softened_permissive"):
        ridge_raw = ridge_weighted / RIDGE_WEIGHT if RIDGE_WEIGHT > 0 else 0.0
        det_raw = (
            det_weighted / DETERMINISTIC_WEIGHT if DETERMINISTIC_WEIGHT > 0 else 0.0
        )
        compliance = risk_compliance(penalty_sum, penalty_cap)
        return ridge_w * ridge_raw + det_w * det_raw + risk_w * compliance
    raise ValueError(f"unknown mode: {mode}")


def _rollout_from_record(record: dict, recomputed_reward: float) -> Rollout:
    """Build a ``Rollout`` with reward replaced by the recomputed value.

    All other fields (penalty names, response length, clipped flag,
    weighted components) come straight from the stored record.
    """
    return Rollout(
        response=record.get("response", ""),
        response_length=int(record.get("response_length", 0)),
        reward=recomputed_reward,
        ridge_rubric=float(record["ridge_rubric"]),
        deterministic=float(record["deterministic"]),
        penalty_sum=float(record["penalty_sum"]),
        penalty_names=tuple(record.get("penalty_names", []) or ()),
        clipped=bool(record.get("clipped", False)),
    )


def _thresholds_for_mode(mode: str) -> DifficultyThresholds:
    """Bucket thresholds per mode.

    * Strict modes (``current_components`` / ``scalar_current``) keep the
      plan default ``[0.20, 0.80]`` band on the [-1, 1] reward range.
    * ``scalar_softened`` shifts to ``[0.30, 0.90]`` because the floor
      lifts under risk_compliance.
    * ``scalar_softened_permissive`` drops min_reward_std to 0.005 and
      disables the same_pattern guard — under softened reward, ridge
      dim variation across rollouts still differentiates them, so
      identical penalty patterns are no longer a useless-group signal.
    """
    if mode == "scalar_softened":
        return DifficultyThresholds(
            min_reward=0.30,
            max_reward=0.90,
            min_reward_std=0.03,
            max_clipped_rate=0.10,
        )
    if mode == "scalar_softened_permissive":
        return DifficultyThresholds(
            min_reward=0.20,
            max_reward=0.95,
            min_reward_std=0.005,
            max_clipped_rate=0.10,
            check_same_pattern=False,
        )
    return DifficultyThresholds(
        min_reward=DEFAULT_THRESHOLDS.min_reward,
        max_reward=DEFAULT_THRESHOLDS.max_reward,
        min_reward_std=DEFAULT_THRESHOLDS.min_reward_std,
        max_clipped_rate=DEFAULT_THRESHOLDS.max_clipped_rate,
    )


@click.command()
@click.option(
    "--rollouts",
    "rollouts_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--mix",
    "mix_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--output-dir",
    default="outputs/rl_difficulty/mix_v1",
    show_default=True,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--penalty-cap", default=DEFAULT_PENALTY_CAP, show_default=True, type=float
)
@click.option(
    "--ridge-weight", default=DEFAULT_RIDGE_WEIGHT_SOFT, show_default=True, type=float
)
@click.option(
    "--det-weight", default=DEFAULT_DET_WEIGHT_SOFT, show_default=True, type=float
)
@click.option(
    "--risk-weight", default=DEFAULT_RISK_WEIGHT_SOFT, show_default=True, type=float
)
def main(
    rollouts_path: Path,
    mix_path: Path,
    output_dir: Path,
    penalty_cap: float,
    ridge_weight: float,
    det_weight: float,
    risk_weight: float,
) -> None:
    """Re-bucket existing rollouts under all reward modes (no Modal cost)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group rollouts by task_id, preserving order.
    by_task: dict[str, list[dict]] = defaultdict(list)
    with rollouts_path.open() as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                by_task[r["task_id"]].append(r)
    click.echo(
        f"Loaded {sum(len(v) for v in by_task.values())} rollouts across {len(by_task)} tasks"
    )

    mix_rows = [
        json.loads(line) for line in mix_path.read_text().splitlines() if line.strip()
    ]
    task_meta = {row["id"]: row for row in mix_rows}

    summary: dict[str, dict] = {}

    from humanize_rl.rl.difficulty import DifficultyThresholds  # noqa: F401

    for mode in MODES:
        thresholds = _thresholds_for_mode(mode)

        diffs = {}
        per_task_rows = []
        for tid, records in by_task.items():
            rollouts = []
            for rec in records:
                reward = _recompute_reward(
                    mode,
                    rec["ridge_rubric"],
                    rec["deterministic"],
                    rec["penalty_sum"],
                    ridge_w=ridge_weight,
                    det_w=det_weight,
                    risk_w=risk_weight,
                    penalty_cap=penalty_cap,
                )
                rollouts.append(_rollout_from_record(rec, reward))
            diff = summarize_rollouts(tid, rollouts, thresholds)
            diffs[tid] = diff
            meta = task_meta.get(tid, {})
            per_task_rows.append(
                {
                    "task_id": diff.task_id,
                    "family": meta.get("family", "?"),
                    "reward_profile": meta.get("reward_profile", "?"),
                    "n_completions": diff.n_completions,
                    "reward_mean": diff.reward_mean,
                    "reward_std": diff.reward_std,
                    "ridge_mean": diff.ridge_mean,
                    "ridge_std": diff.ridge_std,
                    "det_mean": diff.det_mean,
                    "det_std": diff.det_std,
                    "penalty_rate": diff.penalty_rate,
                    "penalty_pattern_diversity": diff.penalty_pattern_diversity,
                    "response_length_mean": diff.response_length_mean,
                    "response_length_p95": diff.response_length_p95,
                    "clipped_rate": diff.clipped_rate,
                    "bucket": diff.bucket,
                    "kept": diff.kept,
                    "reasons": list(diff.reasons),
                }
            )

        out_path = output_dir / f"rebucket_{mode}.jsonl"
        with out_path.open("w") as fh:
            for row in per_task_rows:
                fh.write(json.dumps(row) + "\n")

        # Also write a filtered mix per mode so downstream training can
        # consume any of them.
        filtered, drops = filter_task_mix(mix_rows, diffs)
        filtered_path = output_dir / f"rebucket_{mode}_filtered_mix.jsonl"
        with filtered_path.open("w") as fh:
            for row in filtered:
                fh.write(json.dumps(row) + "\n")

        bucket_counts = Counter(d.bucket for d in diffs.values())
        summary[mode] = {
            "bucket_counts": dict(bucket_counts),
            "kept": len(filtered),
            "dropped": dict(drops),
            "thresholds": {
                "min_reward": thresholds.min_reward,
                "max_reward": thresholds.max_reward,
                "min_reward_std": thresholds.min_reward_std,
                "max_clipped_rate": thresholds.max_clipped_rate,
            },
            "per_task_path": str(out_path),
            "filtered_mix_path": str(filtered_path),
        }

    summary["soft_reward_params"] = {
        "ridge_weight": ridge_weight,
        "det_weight": det_weight,
        "risk_weight": risk_weight,
        "penalty_cap": penalty_cap,
    }

    summary_path = output_dir / "rebucket_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    click.echo(f"Wrote {summary_path}")

    # Pretty terminal table
    click.echo("\nBucket comparison across reward modes:")
    all_buckets = sorted(
        {bucket for mode in MODES for bucket in summary[mode]["bucket_counts"]},
        key=lambda b: -max(summary[m]["bucket_counts"].get(b, 0) for m in MODES),
    )
    header = ["bucket"] + list(MODES)
    rows = [header]
    for bucket in all_buckets:
        rows.append(
            [bucket] + [str(summary[m]["bucket_counts"].get(bucket, 0)) for m in MODES]
        )
    rows.append(["KEPT"] + [str(summary[m]["kept"]) for m in MODES])
    widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
    for row in rows:
        click.echo(
            "  " + "  ".join(c.ljust(w) for c, w in zip(row, widths, strict=True))
        )


if __name__ == "__main__":
    main()
