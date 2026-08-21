from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.audit_prime_run_bundle import build_prime_run_audit_bundle, cli


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _write_detector_rows(path: Path) -> None:
    rows = [
        {
            "id": "human_1",
            "text": "I deployed the hotfix and checked the logs. Everything is stable now.",
            "expected_label": "human",
            "group": "plain_control",
        },
        {
            "id": "ai_1",
            "text": "Certainly, here's a more natural version that unlocks value.",
            "expected_label": "ai",
            "group": "template_ai",
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")


def _audit(high_failed: int = 0) -> dict:
    return {
        "artifact": "prime_rollout_audit",
        "rollout_path": "rollouts_step50.json",
        "reward_mode": "p50_50_no_penalty",
        "rollouts": {
            "row_count": 64,
            "matched_task_count": 64,
            "missing_task_count": 0,
        },
        "recomputed_reward": {
            "count": 64,
            "mean": 0.55,
            "min": 0.20,
            "max": 0.74 if high_failed == 0 else 0.91,
        },
        "diagnostics": {
            "failed_counts": {"fake_casual_phrase": 3},
            "high_rescored_with_failed_diagnostics": high_failed,
            "high_rescored_with_emoji": 0,
            "high_rescored_with_all_caps": 0,
            "high_rescored_with_option_or_wrapper": 0,
        },
    }


def test_build_prime_run_audit_bundle_passes_existing_audit(tmp_path: Path) -> None:
    audit_path = tmp_path / "audit_step50.json"
    detector_path = tmp_path / "detector.jsonl"
    output_dir = tmp_path / "bundle"
    _write_json(audit_path, _audit())
    _write_detector_rows(detector_path)

    bundle = build_prime_run_audit_bundle(
        run_id="run123",
        output_dir=output_dir,
        audit_paths=[audit_path],
        detector_input=detector_path,
    )

    assert bundle["promotion_gate"]["passed"] is True
    assert bundle["rollout_audits"][0]["gate_passed"] is True
    assert bundle["detector_mimic"]["summary"]["gate_passed"] is True
    assert (output_dir / "bundle_report.json").exists()
    assert (output_dir / "detector_mimic_report.json").exists()


def test_audit_bundle_cli_fails_when_high_reward_diagnostics_remain(
    tmp_path: Path,
) -> None:
    audit_path = tmp_path / "audit_step50.json"
    detector_path = tmp_path / "detector.jsonl"
    output_dir = tmp_path / "bundle"
    _write_json(audit_path, _audit(high_failed=1))
    _write_detector_rows(detector_path)

    result = CliRunner().invoke(
        cli,
        [
            "--run-id",
            "run123",
            "--audit",
            str(audit_path),
            "--detector-input",
            str(detector_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code != 0
    assert "Prime run audit bundle gate failed" in result.output
    report = json.loads((output_dir / "bundle_report.json").read_text())
    assert report["promotion_gate"]["passed"] is False
    assert "audit_step50.json: high_rescored_with_failed_diagnostics=1" in report[
        "promotion_gate"
    ]["failures"]
