from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.data.build.prepare_prime_sft_messages_dataset import (
    cli,
    prepare_prime_sft_messages_dataset,
)


def _row(idx: int, *, source: str = "prime_failure_reference_generation") -> dict[str, object]:
    return {
        "id": f"row-{idx}",
        "messages": [
            {"role": "user", "content": f"Write update {idx}"},
            {"role": "assistant", "content": f"Update {idx} is ready."},
        ],
        "source": source,
        "mode": "rewrite",
        "domain": "chat",
        "metadata": {"source": source, "input_index": idx},
        "quality": {"judge_keep": True, "judge_reason": "ok"},
        "license": "project_synthetic",
        "release_eligible": True,
        "split": "train",
        "task_type": "slack_chat",
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_prepare_prime_sft_messages_dataset_writes_hub_folder(tmp_path: Path) -> None:
    input_dir = tmp_path / "built"
    output_dir = tmp_path / "hub"
    _write_jsonl(input_dir / "train.jsonl", [_row(1), _row(2, source="stream_b")])
    _write_jsonl(input_dir / "valid.jsonl", [_row(3)])
    _write_jsonl(input_dir / "test.jsonl", [_row(4)])
    (input_dir / "quality_report.md").write_text("# Quality\n")
    (input_dir / "manifest.json").write_text("{}\n")
    repair_report = tmp_path / "repair.json"
    repair_report.write_text('{"kept_rows": 3}\n')

    report = prepare_prime_sft_messages_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id="user/repo",
        dataset_version="test_v1",
        repair_report=repair_report,
    )

    assert report["row_counts"] == {"train": 2, "validation": 1, "test": 1}
    assert report["repair_reference_rows"] == 3
    assert report["training_columns"] == [
        "id",
        "messages",
        "domain",
        "task_type",
        "mode",
        "source",
        "license",
        "release_eligible",
        "split",
    ]
    assert report["dropped_training_columns"] == ["metadata", "quality"]
    assert (output_dir / "data" / "validation.jsonl").exists()
    train_row = json.loads((output_dir / "data" / "train.jsonl").read_text().splitlines()[0])
    assert set(train_row) == set(report["training_columns"])
    assert "metadata" not in train_row
    assert "quality" not in train_row
    assert (output_dir / "README.md").read_text().startswith("---\nconfigs:")
    assert "user/repo" in (output_dir / "README.md").read_text()
    assert (output_dir / "reports" / "quality_report.md").exists()
    assert (output_dir / "reports" / "repair.json").exists()


def test_prepare_prime_sft_messages_dataset_rejects_bad_messages(tmp_path: Path) -> None:
    input_dir = tmp_path / "built"
    output_dir = tmp_path / "hub"
    _write_jsonl(input_dir / "train.jsonl", [{"messages": []}])
    _write_jsonl(input_dir / "valid.jsonl", [_row(2)])
    _write_jsonl(input_dir / "test.jsonl", [_row(3)])

    result = CliRunner().invoke(
        cli,
        [
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--repo-id",
            "user/repo",
        ],
    )

    assert result.exit_code != 0
    assert "must have two messages" in result.output
