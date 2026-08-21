from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.run_pangram_bulk_detection import cli, run_pangram_bulk_detection


class FakePangramClient:
    def __init__(self) -> None:
        self.submitted_items: list[dict[str, str]] | None = None

    def submit_bulk(self, *, items: list[dict[str, str]]) -> dict:
        self.submitted_items = items
        return {
            "bulk_id": "bulk_123",
            "status": "queued",
            "total_items": len(items),
            "accepted_items": [{"index": 0, "id": items[0]["id"], "task_id": "task_1"}],
            "failed_items": [],
        }

    def wait_for_bulk(
        self, bulk_id: str, *, timeout: float, poll_interval: float
    ) -> dict:
        assert bulk_id == "bulk_123"
        assert timeout == 120.0
        assert poll_interval == 2.0
        return {
            "bulk_id": bulk_id,
            "status": "succeeded",
            "total_items": 1,
            "accepted": 1,
            "succeeded": 1,
            "failed": 0,
        }

    def get_bulk_results(self, bulk_id: str) -> dict:
        assert bulk_id == "bulk_123"
        return {
            "bulk_id": bulk_id,
            "total_items": 1,
            "items": [
                {
                    "id": "row_1",
                    "result": {
                        "prediction_short": "Human",
                        "fraction_ai": 0.02,
                        "fraction_ai_assisted": 0.0,
                        "fraction_human": 0.98,
                    },
                }
            ],
            "failed_items": [],
        }


def _write_payload(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "pangram_bulk_items",
                "items": [{"id": "row_1", "text": "I checked the logs."}],
            }
        )
        + "\n"
    )


def test_run_pangram_bulk_detection_submits_waits_and_writes_results(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "pangram_bulk_items.json"
    output_path = tmp_path / "pangram_export.json"
    submit_report_path = tmp_path / "pangram_submit.json"
    _write_payload(input_path)
    client = FakePangramClient()

    report = run_pangram_bulk_detection(
        input_path=input_path,
        output_path=output_path,
        submit_report_path=submit_report_path,
        client=client,
        timeout=120.0,
        poll_interval=2.0,
    )

    assert client.submitted_items == [{"id": "row_1", "text": "I checked the logs."}]
    assert report["artifact"] == "pangram_bulk_results"
    assert report["bulk_id"] == "bulk_123"
    assert report["status"]["status"] == "succeeded"
    assert output_path.exists()
    assert submit_report_path.exists()
    written = json.loads(output_path.read_text())
    assert written["items"][0]["id"] == "row_1"
    assert written["items"][0]["result"]["prediction_short"] == "Human"


def test_run_pangram_bulk_detection_dry_run_cli_without_api_key(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "pangram_bulk_items.json"
    output_path = tmp_path / "pangram_export.json"
    submit_report_path = tmp_path / "pangram_submit.json"
    _write_payload(input_path)
    monkeypatch.delenv("PANGRAM_API_KEY", raising=False)

    result = CliRunner().invoke(
        cli,
        [
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--submit-report",
            str(submit_report_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "pangram_bulk_dry_run" in result.output
    assert not output_path.exists()
    report = json.loads(submit_report_path.read_text())
    assert report["artifact"] == "pangram_bulk_dry_run"
    assert report["item_count"] == 1
