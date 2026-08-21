from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.data.build.extract_prime_audit_failures import (
    cli,
    failure_rows_from_audit,
)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n")


def test_failure_rows_from_audit_include_diagnostics_and_high_reward(
    tmp_path: Path,
) -> None:
    rollout_path = tmp_path / "run-1" / "rollouts_step50.json"
    audit_path = tmp_path / "run-1" / "audit_step50.json"
    _write_json(
        rollout_path,
        {
            "samples": [
                {
                    "sample_id": 1,
                    "completion": json.dumps(
                        [{"role": "assistant", "content": "We got vague stuff here."}]
                    ),
                },
                {
                    "sample_id": 2,
                    "completion": json.dumps(
                        [{"role": "assistant", "content": "Clean but overrewarded."}]
                    ),
                },
                {
                    "sample_id": 3,
                    "completion": json.dumps(
                        [{"role": "assistant", "content": "Low reward clean sample."}]
                    ),
                },
            ]
        },
    )
    _write_json(
        audit_path,
        {
            "rollout_path": str(rollout_path),
            "top_samples": [
                {
                    "task_id": "task-a",
                    "sample_id": 1,
                    "problem_id": "p1",
                    "rescored_reward": 0.4,
                    "failed_diagnostics": ["fake_casual_phrase"],
                },
                {
                    "task_id": "task-b",
                    "sample_id": 2,
                    "problem_id": "p2",
                    "rescored_reward": 0.91,
                    "failed_diagnostics": [],
                },
                {
                    "task_id": "task-c",
                    "sample_id": 3,
                    "problem_id": "p3",
                    "rescored_reward": 0.2,
                    "failed_diagnostics": [],
                },
            ],
        },
    )

    rows = failure_rows_from_audit(audit_path=audit_path)

    assert [row["task_id"] for row in rows] == ["task-a", "task-b"]
    assert rows[0]["rl_completion_preview"] == "We got vague stuff here."
    assert rows[0]["phrase_hits"] == ["we got", "stuff"]
    assert rows[0]["reasons"] == ["fake_casual_phrase"]
    assert rows[1]["reasons"] == ["high_rescored_reward"]


def test_failure_rows_can_fallback_to_preview_when_rollout_missing(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit.json"
    _write_json(
        audit_path,
        {
            "rollout_path": "missing_rollouts.json",
            "top_samples": [
                {
                    "task_id": "task-a",
                    "sample_id": 1,
                    "rescored_reward": 0.8,
                    "failed_diagnostics": [],
                    "completion_preview": "Preview only.",
                }
            ],
        },
    )

    rows = failure_rows_from_audit(
        audit_path=audit_path,
        allow_missing_rollouts=True,
    )

    assert rows[0]["rl_completion_preview"] == "Preview only."


def test_cli_writes_limited_failure_jsonl(tmp_path: Path) -> None:
    rollout_path = tmp_path / "run-1" / "rollouts.json"
    audit_path = tmp_path / "run-1" / "audit.json"
    output_path = tmp_path / "failures.jsonl"
    _write_json(
        rollout_path,
        {
            "samples": [
                {
                    "sample_id": 1,
                    "completion": json.dumps(
                        [{"role": "assistant", "content": "First bad sample."}]
                    ),
                },
                {
                    "sample_id": 2,
                    "completion": json.dumps(
                        [{"role": "assistant", "content": "Second bad sample."}]
                    ),
                },
            ]
        },
    )
    _write_json(
        audit_path,
        {
            "rollout_path": str(rollout_path),
            "top_samples": [
                {
                    "task_id": "task-a",
                    "sample_id": 1,
                    "rescored_reward": 0.8,
                    "failed_diagnostics": [],
                },
                {
                    "task_id": "task-b",
                    "sample_id": 2,
                    "rescored_reward": 0.9,
                    "failed_diagnostics": [],
                },
            ],
        },
    )

    result = CliRunner().invoke(
        cli,
        [
            "--audit",
            str(audit_path),
            "--output",
            str(output_path),
            "--limit",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert "failure_rows=1" in result.output
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["comparison"].endswith(":rank_1:sample_1")
