from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.data.build.generate_sft_references_from_failures import (
    append_jsonl_row,
    build_reference_messages,
    build_sft_row,
    existing_failure_keys,
    existing_task_ids,
    load_task_index,
    main,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_task_index_maps_task_id(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        task_path,
        [
            {
                "id": "rl_v03_000099",
                "instruction": "Rewrite this.",
                "input_text": "Source text",
                "split": "validation",
                "family": "rewrite_repair",
                "mode": "expansion",
            }
        ],
    )

    index = load_task_index([task_path])

    assert index["rl_v03_000099"]["mode"] == "expansion"


def test_build_reference_messages_include_failure_context() -> None:
    task = {
        "id": "rl_v03_000099",
        "instruction": "Explain the parity check program.",
        "input_text": "Byte A0 contains the value.",
        "constraints": {"max_words": 80, "return_only_answer": True},
        "required_facts": ["A0", "even parity"],
        "forbidden_facts": ["interrupts"],
    }
    failure = {
        "reasons": ["length_drift_ge_100_tokens"],
        "phrase_hits": ["hope you're doing well"],
        "base_completion_preview": "short answer",
        "rl_completion_preview": "very long answer",
    }

    messages = build_reference_messages(task, failure)
    content = messages[-1]["content"]

    assert "Explain the parity check program." in content
    assert "Byte A0 contains the value." in content
    assert "length_drift_ge_100_tokens" in content
    assert "hope you're doing well" in content
    assert "Return only the final answer" in content


def test_build_sft_row_has_instruction_response_and_metadata() -> None:
    task = {
        "id": "rl_v01_000016",
        "instruction": "Rewrite this note.",
        "input_text": "Formal source",
        "family": "rewrite_repair",
        "mode": "rewrite",
        "split": "validation",
    }
    failure = {
        "comparison": "p50_mix_v2_val20",
        "reasons": ["high_reward_humanizer_phrase"],
        "phrase_hits": ["so,"],
    }

    row = build_sft_row(task, failure, response="Clean target.")

    assert row["instruction"] == "Rewrite this note.\n\nSource:\nFormal source"
    assert row["response"] == "Clean target."
    assert row["metadata"]["task_id"] == "rl_v01_000016"
    assert row["metadata"]["failure_reasons"] == ["high_reward_humanizer_phrase"]


def test_append_jsonl_row_creates_parent_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.jsonl"

    append_jsonl_row(path, {"metadata": {"task_id": "a"}, "response": "One"})
    append_jsonl_row(path, {"metadata": {"task_id": "b"}, "response": "Two"})

    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["metadata"]["task_id"] for row in rows] == ["a", "b"]


def test_existing_task_ids_reads_output_metadata(tmp_path: Path) -> None:
    path = tmp_path / "refs.jsonl"
    _write_jsonl(
        path,
        [
            {"metadata": {"task_id": "rl_v03_000099"}, "response": "A"},
            {"metadata": {"task_id": "rl_v01_000369"}, "response": "B"},
        ],
    )

    assert existing_task_ids(path) == {"rl_v03_000099", "rl_v01_000369"}


def test_existing_failure_keys_include_comparison(tmp_path: Path) -> None:
    path = tmp_path / "refs.jsonl"
    _write_jsonl(
        path,
        [
            {
                "metadata": {
                    "task_id": "rl_v01_000016",
                    "failure_comparison": "p50_mix_v2_val20",
                },
                "response": "A",
            },
            {
                "metadata": {
                    "task_id": "rl_v01_000016",
                    "failure_comparison": "strict_v02_smoke_val",
                },
                "response": "B",
            },
        ],
    )

    assert existing_failure_keys(path) == {
        ("rl_v01_000016", "p50_mix_v2_val20"),
        ("rl_v01_000016", "strict_v02_smoke_val"),
    }


def test_cli_resume_uses_task_id_and_comparison(tmp_path: Path) -> None:
    failure_path = tmp_path / "failures.jsonl"
    task_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "refs.jsonl"
    _write_jsonl(
        failure_path,
        [
            {"task_id": "rl_v01_000016", "comparison": "p50_mix_v2_val20"},
            {"task_id": "rl_v01_000016", "comparison": "strict_v02_smoke_val"},
        ],
    )
    _write_jsonl(
        task_path,
        [
            {
                "id": "rl_v01_000016",
                "instruction": "Rewrite this note.",
                "input_text": "Source text",
            }
        ],
    )
    _write_jsonl(
        output_path,
        [
            {
                "metadata": {
                    "task_id": "rl_v01_000016",
                    "failure_comparison": "p50_mix_v2_val20",
                },
                "response": "Existing",
            }
        ],
    )

    result = CliRunner().invoke(
        main,
        [
            "--failure-path",
            str(failure_path),
            "--task-path",
            str(task_path),
            "--output-path",
            str(output_path),
            "--dry-run",
            "--resume",
        ],
    )

    assert result.exit_code == 0
    assert "skipped_existing=1" in result.output
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 2
    assert {
        (row["metadata"]["task_id"], row["metadata"].get("failure_comparison"))
        for row in rows
    } == {
        ("rl_v01_000016", "p50_mix_v2_val20"),
        ("rl_v01_000016", "strict_v02_smoke_val"),
    }


def test_cli_overwrite_removes_existing_output(tmp_path: Path) -> None:
    failure_path = tmp_path / "failures.jsonl"
    task_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "refs.jsonl"
    _write_jsonl(failure_path, [{"task_id": "task-1", "comparison": "eval-a"}])
    _write_jsonl(task_path, [{"id": "task-1", "instruction": "Write one line."}])
    _write_jsonl(output_path, [{"metadata": {"task_id": "old"}, "response": "Old"}])

    result = CliRunner().invoke(
        main,
        [
            "--failure-path",
            str(failure_path),
            "--task-path",
            str(task_path),
            "--output-path",
            str(output_path),
            "--dry-run",
            "--overwrite",
        ],
    )

    assert result.exit_code == 0
    rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert len(rows) == 1
    assert rows[0]["metadata"]["task_id"] == "task-1"


def test_cli_rejects_non_google_model(tmp_path: Path) -> None:
    failure_path = tmp_path / "failures.jsonl"
    task_path = tmp_path / "tasks.jsonl"
    output_path = tmp_path / "refs.jsonl"
    _write_jsonl(failure_path, [{"task_id": "task-1", "comparison": "eval-a"}])
    _write_jsonl(task_path, [{"id": "task-1", "instruction": "Write one line."}])

    result = CliRunner().invoke(
        main,
        [
            "--failure-path",
            str(failure_path),
            "--task-path",
            str(task_path),
            "--output-path",
            str(output_path),
            "--dry-run",
            "--model",
            "openai/gpt-5.4-mini",
        ],
    )

    assert result.exit_code != 0
    assert "Only Google OpenRouter models are allowed here." in result.output
