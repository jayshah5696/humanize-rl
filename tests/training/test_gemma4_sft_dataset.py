from __future__ import annotations

import json
from pathlib import Path

from humanize_rl.training.gemma4_sft_dataset import (
    BuildConfig,
    build_dataset,
    normalize_row,
    rejection_reasons,
)


def write_rows(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_normalize_instruction_response_row() -> None:
    pair = normalize_row({"instruction": "Write a note", "response": "Here it is."})

    assert pair.instruction == "Write a note"
    assert pair.response == "Here it is."


def test_normalize_messages_row() -> None:
    pair = normalize_row(
        {
            "messages": [
                {"role": "user", "content": "Write a note"},
                {"role": "assistant", "content": "Here it is."},
            ]
        }
    )

    assert pair.instruction == "Write a note"
    assert pair.response == "Here it is."


def test_rejects_ai_tell_phrase(tmp_path: Path) -> None:
    config = BuildConfig(input_path=tmp_path / "in.jsonl", output_dir=tmp_path / "out")
    pair = normalize_row(
        {"instruction": "Write a note", "response": "Certainly, here is the note."}
    )

    assert "response_ai_tell:certainly" in rejection_reasons(pair, config)


def test_rejects_invented_name_for_missing_cover(tmp_path: Path) -> None:
    config = BuildConfig(input_path=tmp_path / "in.jsonl", output_dir=tmp_path / "out")
    pair = normalize_row(
        {
            "instruction": "Draft an email to my manager requesting PTO, outlining who will cover my projects.",
            "response": "Sarah will cover my projects while I'm out.",
        }
    )

    reasons = rejection_reasons(pair, config)

    assert "missing_placeholder_for_unspecified_cover:Sarah" in reasons


def test_preserves_placeholder_for_missing_cover(tmp_path: Path) -> None:
    config = BuildConfig(input_path=tmp_path / "in.jsonl", output_dir=tmp_path / "out")
    pair = normalize_row(
        {
            "instruction": "Draft an email to my manager requesting PTO, outlining who will cover my projects.",
            "response": "[Colleague] will cover my projects while I'm out.",
        }
    )

    assert rejection_reasons(pair, config) == []


def test_build_dataset_writes_splits_and_manifest(tmp_path: Path) -> None:
    input_path = tmp_path / "input.jsonl"
    output_dir = tmp_path / "out"
    rows = [
        {
            "instruction": f"Write a short update {idx}",
            "response": f"Update {idx} is ready for review today.",
            "source": "safe_expand_raw",
        }
        for idx in range(20)
    ]
    rows.append(
        {
            "instruction": "Write a note",
            "response": "Certainly, here it is.",
            "source": "safe_expand_raw",
        }
    )
    write_rows(input_path, rows)

    result = build_dataset(
        BuildConfig(
            input_path=input_path,
            output_dir=output_dir,
            track_a_scorer_path=None,
            smoke_train_size=5,
            smoke_valid_size=2,
        )
    )

    assert result.report["manifest"]["raw_rows"] == 21
    assert result.report["manifest"]["accepted_rows"] == 20
    assert result.report["manifest"]["rejected_rows"] == 1
    assert (output_dir / "train.jsonl").exists()
    assert (output_dir / "valid.jsonl").exists()
    assert (output_dir / "test.jsonl").exists()
    assert (output_dir / "smoke_train.jsonl").exists()
    assert (output_dir / "manifest.json").exists()
