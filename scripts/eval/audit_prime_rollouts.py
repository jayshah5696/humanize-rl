#!/usr/bin/env python3
"""Audit Prime rollout JSON against the local reward implementation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import click

from humanize_rl.reward.env import render_prompt
from humanize_rl.reward.reward import RewardMode, load_ridge_scorer, score_response
from humanize_rl.reward.tasks import RLTask, load_tasks

DEFAULT_TASKSET = Path("data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl")
DEFAULT_HIGH_REWARD_THRESHOLD = 0.75
EVAL_LABEL_SPECS: dict[str, tuple[Path, RewardMode]] = {
    "mix_v2_p5050": (
        Path("data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl"),
        "p50_50_no_penalty",
    ),
    "v02_strict": (
        Path("environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl"),
        "strict",
    ),
    "v03_strict": (
        Path("data/rl/humanize_tasks_v03_filtered.jsonl"),
        "strict",
    ),
}


def resolve_audit_inputs(
    *,
    eval_label: str | None,
    taskset_path: Path | None,
    reward_mode: RewardMode | None,
) -> tuple[Path, RewardMode]:
    """Resolve an eval label shorthand into taskset and reward mode inputs."""
    label_taskset: Path = DEFAULT_TASKSET
    label_reward_mode: RewardMode = "p50_50_no_penalty"
    if eval_label:
        label_taskset, label_reward_mode = EVAL_LABEL_SPECS[eval_label]
    return taskset_path or label_taskset, reward_mode or label_reward_mode


def _as_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _chat_content(value: Any) -> str:
    decoded = _as_json(value)
    if isinstance(decoded, list):
        parts: list[str] = []
        for item in decoded:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(decoded, dict):
        content = decoded.get("content")
        return str(content).strip() if content is not None else ""
    return str(decoded or "").strip()


def _load_samples(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = _as_json(data.get("samples", []))
    else:
        raise ValueError(f"unsupported Prime rollout JSON root: {type(data).__name__}")

    if not isinstance(samples, list):
        raise ValueError("Prime rollout JSON does not contain a samples list")
    return [sample for sample in samples if isinstance(sample, dict)]


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _task_index(taskset_path: Path) -> dict[str, RLTask]:
    return {render_prompt(task): task for task in load_tasks(taskset_path)}


def _failed_diagnostic_names(result: Any) -> list[str]:
    return [
        str(diagnostic["name"])
        for diagnostic in result.diagnostics
        if not bool(diagnostic["passed"])
    ]


def build_rollout_audit(
    rollout_path: Path,
    taskset_path: Path = DEFAULT_TASKSET,
    reward_mode: RewardMode = "p50_50_no_penalty",
    ridge_path: Path | None = None,
    high_reward_threshold: float = DEFAULT_HIGH_REWARD_THRESHOLD,
    top_n: int = 5,
    track_a_scorer: Any | None = None,
) -> dict[str, Any]:
    """Build a deterministic audit report for a saved Prime rollout response."""
    samples = _load_samples(rollout_path)
    tasks_by_prompt = _task_index(taskset_path)
    scorer = track_a_scorer if track_a_scorer is not None else load_ridge_scorer(ridge_path)

    if reward_mode == "p50_50_no_penalty" and scorer is None:
        raise RuntimeError("p50_50_no_penalty audit requires an available ridge scorer")

    sample_rewards: list[float] = []
    rescored_rewards: list[float] = []
    failed_counts: Counter[str] = Counter()
    missing_prompts: list[str] = []
    problem_rewards: dict[str, list[float]] = defaultdict(list)
    high_failed = 0
    high_emoji = 0
    high_all_caps = 0
    high_option_or_wrapper = 0
    audited_rows: list[dict[str, Any]] = []

    for sample in samples:
        prompt_text = _chat_content(sample.get("prompt"))
        completion_text = _chat_content(sample.get("completion"))
        task = tasks_by_prompt.get(prompt_text)
        if task is None:
            missing_prompts.append(prompt_text[:240])
            continue

        if sample.get("reward") is not None:
            sample_rewards.append(float(sample["reward"]))

        result = score_response(task, completion_text, scorer, reward_mode=reward_mode)
        failed = _failed_diagnostic_names(result)
        failed_counts.update(failed)
        rescored_rewards.append(result.reward)
        problem_id = str(sample.get("problem_id", "missing"))
        problem_rewards[problem_id].append(result.reward)

        if result.reward >= high_reward_threshold and failed:
            high_failed += 1
        if result.reward >= high_reward_threshold and "emoji" in failed:
            high_emoji += 1
        if result.reward >= high_reward_threshold and "all_caps" in failed:
            high_all_caps += 1
        if result.reward >= high_reward_threshold and (
            "option_menu" in failed or "wrapper_phrase" in failed
        ):
            high_option_or_wrapper += 1

        audited_rows.append(
            {
                "task_id": task.id,
                "problem_id": problem_id,
                "sample_id": sample.get("sample_id"),
                "sample_reward": sample.get("reward"),
                "rescored_reward": result.reward,
                "failed_diagnostics": failed,
                "completion_preview": completion_text[:500],
            }
        )

    top_samples = sorted(
        audited_rows, key=lambda row: float(row["rescored_reward"]), reverse=True
    )[:top_n]

    return {
        "artifact": "prime_rollout_audit",
        "rollout_path": str(rollout_path),
        "taskset_path": str(taskset_path),
        "reward_mode": reward_mode,
        "high_reward_threshold": high_reward_threshold,
        "rollouts": {
            "row_count": len(samples),
            "matched_task_count": len(audited_rows),
            "missing_task_count": len(missing_prompts),
            "missing_prompt_previews": missing_prompts[:10],
        },
        "prime_sample_reward": _stats(sample_rewards),
        "recomputed_reward": _stats(rescored_rewards),
        "diagnostics": {
            "failed_counts": dict(sorted(failed_counts.items())),
            "high_rescored_with_failed_diagnostics": high_failed,
            "high_rescored_with_emoji": high_emoji,
            "high_rescored_with_all_caps": high_all_caps,
            "high_rescored_with_option_or_wrapper": high_option_or_wrapper,
        },
        "problem_rewards": {
            problem_id: [round(reward, 6) for reward in rewards]
            for problem_id, rewards in sorted(problem_rewards.items())
        },
        "top_samples": top_samples,
    }


@click.command()
@click.option(
    "--rollouts",
    "rollout_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--eval-label",
    type=click.Choice(sorted(EVAL_LABEL_SPECS)),
    default=None,
    help="Shorthand for the taskset and reward mode used by a promotion eval label.",
)
@click.option(
    "--taskset",
    "taskset_path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Override taskset JSONL. Defaults to --eval-label or p50 taskset.",
)
@click.option(
    "--output",
    "output_path",
    required=True,
    type=click.Path(dir_okay=False, path_type=Path),
)
@click.option(
    "--reward-mode",
    type=click.Choice(["strict", "scalar_softened", "p50_50_no_penalty"]),
    default=None,
    help="Override reward mode. Defaults to --eval-label or p50_50_no_penalty.",
)
@click.option(
    "--ridge-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--high-reward-threshold",
    default=DEFAULT_HIGH_REWARD_THRESHOLD,
    show_default=True,
)
@click.option("--top-n", default=5, show_default=True)
def main(
    rollout_path: Path,
    eval_label: str | None,
    taskset_path: Path | None,
    output_path: Path,
    reward_mode: RewardMode | None,
    ridge_path: Path | None,
    high_reward_threshold: float,
    top_n: int,
) -> None:
    """Write a JSON audit for Prime rollout samples."""
    resolved_taskset, resolved_reward_mode = resolve_audit_inputs(
        eval_label=eval_label,
        taskset_path=taskset_path,
        reward_mode=reward_mode,
    )
    report = build_rollout_audit(
        rollout_path=rollout_path,
        taskset_path=resolved_taskset,
        reward_mode=resolved_reward_mode,
        ridge_path=ridge_path,
        high_reward_threshold=high_reward_threshold,
        top_n=top_n,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    click.echo(
        "wrote audit "
        f"rows={report['rollouts']['matched_task_count']}/"
        f"{report['rollouts']['row_count']} "
        f"mean={report['recomputed_reward']['mean']} to {output_path}"
    )


if __name__ == "__main__":
    main()
