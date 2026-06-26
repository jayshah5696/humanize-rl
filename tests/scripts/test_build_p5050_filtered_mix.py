from __future__ import annotations

import json

from click.testing import CliRunner

from scripts.rl.build_p5050_filtered_mix import main


def _write_jsonl(path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_build_p5050_filtered_mix_keeps_useful_tasks(tmp_path) -> None:
    mix = [
        {
            "id": "task_a",
            "split": "train",
            "family": "rewrite_repair",
            "mode": "rewrite",
        },
        {
            "id": "task_b",
            "split": "train",
            "family": "compression",
            "mode": "compression",
        },
        {"id": "task_c", "split": "test", "family": "tone_shift", "mode": "tone_shift"},
    ]
    rollouts = [
        {
            "task_id": "task_a",
            "ridge_rubric": 0.35,
            "deterministic": 0.35,
            "penalty_sum": 0.0,
            "clipped": False,
        },
        {
            "task_id": "task_a",
            "ridge_rubric": 0.36,
            "deterministic": 0.35,
            "penalty_sum": -0.2,
            "clipped": False,
        },
        {
            "task_id": "task_b",
            "ridge_rubric": 0.48,
            "deterministic": 0.48,
            "penalty_sum": 0.0,
            "clipped": False,
        },
        {
            "task_id": "task_b",
            "ridge_rubric": 0.49,
            "deterministic": 0.48,
            "penalty_sum": 0.0,
            "clipped": False,
        },
        {
            "task_id": "task_c",
            "ridge_rubric": 0.35,
            "deterministic": 0.35,
            "penalty_sum": 0.0,
            "clipped": True,
        },
        {
            "task_id": "task_c",
            "ridge_rubric": 0.36,
            "deterministic": 0.35,
            "penalty_sum": 0.0,
            "clipped": False,
        },
    ]
    mix_path = tmp_path / "mix.jsonl"
    rollouts_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "filtered.jsonl"
    summary_path = tmp_path / "summary.json"
    per_task_path = tmp_path / "per_task.jsonl"
    _write_jsonl(mix_path, mix)
    _write_jsonl(rollouts_path, rollouts)

    result = CliRunner().invoke(
        main,
        [
            "--mix",
            str(mix_path),
            "--rollouts",
            str(rollouts_path),
            "--output",
            str(output_path),
            "--summary",
            str(summary_path),
            "--per-task",
            str(per_task_path),
            "--max-reward",
            "0.95",
        ],
    )

    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [row["id"] for row in rows] == ["task_a"]
    assert rows[0]["difficulty"]["reward_mode"] == "p50_50_no_penalty"

    summary = json.loads(summary_path.read_text())
    assert summary["input_rows"] == 3
    assert summary["kept"] == 1
    assert summary["dropped"] == {"too_easy": 1, "clipped": 1}
    assert summary["split_counts"] == {"train": 1}
