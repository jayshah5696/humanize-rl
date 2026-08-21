from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.export_detector_mimic_for_pangram import cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_export_detector_mimic_for_pangram_writes_sdk_items(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "detector_rows.jsonl"
    output_path = tmp_path / "pangram_bulk_items.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "human_1",
                "text": "I checked the logs. The service is stable now.",
                "expected_label": "human",
            },
            {
                "id": "ai_1",
                "text": "Certainly, here's a more natural version that unlocks value.",
                "expected_label": "ai",
            },
        ],
    )

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code == 0
    assert "items=2" in result.output
    payload = json.loads(output_path.read_text())
    assert payload["artifact"] == "pangram_bulk_items"
    assert payload["sdk_method"] == "Pangram.submit_bulk(items=payload['items'])"
    assert payload["source_path"] == str(input_path)
    assert payload["item_count"] == 2
    assert payload["items"] == [
        {"id": "human_1", "text": "I checked the logs. The service is stable now."},
        {
            "id": "ai_1",
            "text": "Certainly, here's a more natural version that unlocks value.",
        },
    ]


def test_export_detector_mimic_for_pangram_rejects_duplicate_ids(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "detector_rows.jsonl"
    output_path = tmp_path / "pangram_bulk_items.json"
    _write_jsonl(
        input_path,
        [
            {"id": "dup", "text": "One row.", "expected_label": "human"},
            {"id": "dup", "text": "Another row.", "expected_label": "ai"},
        ],
    )

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code != 0
    assert "duplicate detector row id: dup" in result.output
