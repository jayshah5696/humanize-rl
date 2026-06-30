import json
from pathlib import Path

from click.testing import CliRunner

from scripts.train.verify_prime_warm_start_checkpoint import (
    build_checkpoint_handoff,
    cli,
)


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _run_payload(model: str = "Qwen/Qwen3.5-2B") -> dict:
    return {
        "run": {
            "id": "run_123",
            "name": "sft-or-hosted-run",
            "status": "COMPLETED",
            "base_model": model,
        }
    }


def _checkpoints_payload(status: str = "READY") -> dict:
    return {
        "checkpoints": [
            {
                "id": "ckpt_ready_123",
                "rft_run_id": "run_123",
                "step": 200,
                "status": status,
                "storage_url": "s3://bucket/checkpoint",
            }
        ]
    }


def test_build_checkpoint_handoff_passes_ready_matching_checkpoint(
    tmp_path: Path,
) -> None:
    output = tmp_path / "checkpoint_handoff.json"

    report = build_checkpoint_handoff(
        run_payload=_run_payload(),
        checkpoints_payload=_checkpoints_payload(),
        checkpoint_id="ckpt_ready_123",
        expected_model="Qwen/Qwen3.5-2B",
        output_path=output,
    )

    assert report["passed"] is True
    assert report["checkpoint_id"] == "ckpt_ready_123"
    assert report["run_id"] == "run_123"
    assert report["checkpoint"]["status"] == "READY"
    assert output.exists()


def test_build_checkpoint_handoff_fails_non_ready_checkpoint(tmp_path: Path) -> None:
    output = tmp_path / "checkpoint_handoff.json"

    report = build_checkpoint_handoff(
        run_payload=_run_payload(),
        checkpoints_payload=_checkpoints_payload(status="PENDING"),
        checkpoint_id="ckpt_ready_123",
        expected_model="Qwen/Qwen3.5-2B",
        output_path=output,
    )

    assert report["passed"] is False
    assert "checkpoint status PENDING is not READY" in report["failures"]


def test_cli_offline_fails_model_mismatch(tmp_path: Path) -> None:
    run_json = tmp_path / "run.json"
    checkpoints_json = tmp_path / "checkpoints.json"
    output = tmp_path / "checkpoint_handoff.json"
    _write_json(run_json, _run_payload(model="Qwen/Qwen3.5-0.8B"))
    _write_json(checkpoints_json, _checkpoints_payload())

    result = CliRunner().invoke(
        cli,
        [
            "--run-json",
            str(run_json),
            "--checkpoints-json",
            str(checkpoints_json),
            "--checkpoint-id",
            "ckpt_ready_123",
            "--expected-model",
            "Qwen/Qwen3.5-2B",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "Prime warm-start checkpoint verification failed" in result.output
    report = json.loads(output.read_text())
    assert "run base_model Qwen/Qwen3.5-0.8B != Qwen/Qwen3.5-2B" in report[
        "failures"
    ]
