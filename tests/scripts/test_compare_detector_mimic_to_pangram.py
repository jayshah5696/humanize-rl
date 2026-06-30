from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.compare_detector_mimic_to_pangram import cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_compare_detector_mimic_to_pangram_cli_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "detector_rows.jsonl"
    pangram_path = tmp_path / "pangram.json"
    output_path = tmp_path / "alignment.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "human_1",
                "text": "I deployed the hotfix and checked the logs. It is stable now.",
                "expected_label": "human",
            },
            {
                "id": "ai_1",
                "text": "Certainly, here's a more natural version that unlocks value.",
                "expected_label": "ai",
            },
        ],
    )
    pangram_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "human_1",
                        "prediction_short": "Human",
                        "fraction_ai": 0.05,
                    },
                    {
                        "id": "ai_1",
                        "prediction_short": "AI",
                        "fraction_ai": 0.90,
                    },
                ]
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--pangram-output",
            str(pangram_path),
            "--output",
            str(output_path),
            "--max-mean-abs-fraction-delta",
            "0.5",
        ],
    )

    assert result.exit_code == 0
    assert "gate=pass" in result.output
    report = json.loads(output_path.read_text())
    assert report["summary"]["matched_rows"] == 2
    assert report["summary"]["gate_passed"] is True


def test_compare_detector_mimic_to_pangram_cli_fails_gate_by_default(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "detector_rows.jsonl"
    pangram_path = tmp_path / "pangram.json"
    output_path = tmp_path / "alignment.json"
    _write_jsonl(
        input_path,
        [
            {
                "id": "ai_1",
                "text": "Certainly, here's a more natural version that unlocks value.",
                "expected_label": "ai",
            }
        ],
    )
    pangram_path.write_text(
        json.dumps(
            {
                "results": [
                    {
                        "id": "ai_1",
                        "prediction_short": "Human",
                        "fraction_ai": 0.02,
                    }
                ]
            }
        )
    )

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--pangram-output",
            str(pangram_path),
            "--output",
            str(output_path),
        ],
    )

    assert result.exit_code != 0
    assert "Pangram alignment gate failed" in result.output
    report = json.loads(output_path.read_text())
    assert report["summary"]["label_disagreement_ids"] == ["ai_1"]
