from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.evaluate_detector_mimic import cli


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def test_detector_mimic_cli_writes_report(tmp_path: Path) -> None:
    input_path = tmp_path / "detector_set.jsonl"
    output_path = tmp_path / "report.json"
    scored_path = tmp_path / "scored.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "human_1",
                "text": "I deployed the hotfix and checked the logs. It is stable now.",
                "expected_label": "human",
                "group": "plain_control",
            },
            {
                "id": "ai_1",
                "text": "Certainly, here's a more natural version that unlocks value.",
                "expected_label": "ai",
                "group": "template_ai",
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
            "--scored-output",
            str(scored_path),
        ],
    )

    assert result.exit_code == 0
    report = json.loads(output_path.read_text())
    assert report["summary"]["gate_passed"] is True
    assert report["summary"]["total_rows"] == 2
    assert len(scored_path.read_text().splitlines()) == 2


def test_detector_mimic_cli_fails_gate_by_default(tmp_path: Path) -> None:
    input_path = tmp_path / "detector_set.jsonl"
    output_path = tmp_path / "report.json"
    scored_path = tmp_path / "scored.jsonl"
    _write_jsonl(
        input_path,
        [
            {
                "id": "mislabeled_human",
                "text": "Certainly, here's a more natural version that unlocks value.",
                "expected_label": "human",
                "group": "plain_control",
            }
        ],
    )

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--scored-output",
            str(scored_path),
        ],
    )

    assert result.exit_code != 0
    assert "Detector mimic gate failed" in result.output
    report = json.loads(output_path.read_text())
    assert report["summary"]["gate_passed"] is False
