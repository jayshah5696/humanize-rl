from __future__ import annotations

import json

from humanize_rl.reward.env import prime_dataset_row
from humanize_rl.reward.tasks import RLTask
from scripts.eval.audit_prime_rollouts import build_rollout_audit, resolve_audit_inputs


class FakePerfectRidgeScorer:
    def predict_proba(self, rows: list[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in rows]

    def predict_rubric(self, rows: list[str]) -> list[list[float]]:
        return [[1.0] * 8 for _ in rows]


def _task() -> RLTask:
    return RLTask.model_validate(
        {
            "id": "rl_v03_000001",
            "family": "tone_shift",
            "domain": "creative_general",
            "mode": "multi_constraint_compose",
            "register": "warm",
            "instruction": (
                "Write a direct social post. MUST_NOT: Use emoji, hashtags, "
                "or all caps. Keep the answer plain."
            ),
            "constraints": {
                "max_words": 40,
                "no_subject_line": True,
                "no_signoff": True,
            },
            "reward_profile": "rewrite_faithful_concise",
            "trap_tags": ["emoji", "all_caps"],
            "split": "train",
            "required_facts": [],
        }
    )


def test_build_rollout_audit_rescores_and_counts_surface_failures(tmp_path) -> None:
    task = _task()
    taskset_path = tmp_path / "tasks.jsonl"
    taskset_path.write_text(task.model_dump_json(by_alias=True) + "\n")

    row = prime_dataset_row(task, example_id=0)
    star = chr(0x1F31F)
    rollout_path = tmp_path / "rollouts.json"
    rollout_path.write_text(
        json.dumps(
            {
                "run_id": "run123",
                "samples": [
                    {
                        "problem_id": 1,
                        "sample_id": 0,
                        "prompt": json.dumps(row.prompt),
                        "completion": json.dumps(
                            [
                                {
                                    "role": "assistant",
                                    "content": f"GET DIRECT ROMANCE PICKS NOW. {star}",
                                }
                            ]
                        ),
                        "reward": 0.90,
                    }
                ],
            }
        )
    )

    report = build_rollout_audit(
        rollout_path=rollout_path,
        taskset_path=taskset_path,
        track_a_scorer=FakePerfectRidgeScorer(),
    )

    assert report["rollouts"]["matched_task_count"] == 1
    assert report["diagnostics"]["failed_counts"]["emoji"] == 1
    assert report["diagnostics"]["failed_counts"]["all_caps"] == 1
    assert report["diagnostics"]["high_rescored_with_failed_diagnostics"] == 0
    assert report["recomputed_reward"]["max"] <= 0.70


def test_resolve_audit_inputs_uses_eval_label_defaults() -> None:
    taskset, reward_mode = resolve_audit_inputs(
        eval_label="v03_strict",
        taskset_path=None,
        reward_mode=None,
    )

    assert taskset.as_posix() == "data/rl/humanize_tasks_v03_filtered.jsonl"
    assert reward_mode == "strict"


def test_resolve_audit_inputs_allows_explicit_overrides(tmp_path) -> None:
    custom_taskset = tmp_path / "custom.jsonl"

    taskset, reward_mode = resolve_audit_inputs(
        eval_label="v02_strict",
        taskset_path=custom_taskset,
        reward_mode="p50_50_no_penalty",
    )

    assert taskset == custom_taskset
    assert reward_mode == "p50_50_no_penalty"
