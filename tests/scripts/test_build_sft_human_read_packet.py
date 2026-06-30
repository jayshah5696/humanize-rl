import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.build_sft_human_read_packet import build_human_read_packet, cli


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _audit(label: str, count: int) -> dict:
    return {
        "artifact": "prime_rollout_audit",
        "top_samples": [
            {
                "task_id": f"{label}_task_{index}",
                "problem_id": f"{label}_problem_{index}",
                "sample_id": f"{label}_sample_{index}",
                "rescored_reward": 0.9 - (index * 0.01),
                "failed_diagnostics": ["fake_casual_phrase"] if index == 0 else [],
                "completion_preview": f"{label} completion {index}",
            }
            for index in range(count)
        ],
    }


def test_build_human_read_packet_collects_bounded_candidate_samples(
    tmp_path: Path,
) -> None:
    mix = tmp_path / "mix.json"
    strict = tmp_path / "strict.json"
    output = tmp_path / "human_read.json"
    _write_json(mix, _audit("mix", 12))
    _write_json(strict, _audit("strict", 12))

    packet = build_human_read_packet(
        checkpoint_id="ckpt_ready_123",
        candidate_audits={"mix_v2_p5050": mix, "v02_strict": strict},
        output_path=output,
        samples_per_audit=12,
        min_samples=20,
        max_samples=50,
    )

    assert packet["artifact"] == "sft_human_read_packet"
    assert packet["checkpoint_id"] == "ckpt_ready_123"
    assert packet["passed"] is False
    assert packet["review_status"] == "needs_review"
    assert packet["sample_count"] == 24
    assert packet["samples"][0]["label"] == "mix_v2_p5050"
    assert packet["samples"][0]["review_decision"] == "unreviewed"
    assert output.exists()


def test_build_human_read_packet_fails_when_audits_have_too_few_samples(
    tmp_path: Path,
) -> None:
    audit = tmp_path / "candidate.json"
    output = tmp_path / "human_read.json"
    _write_json(audit, _audit("mix", 3))

    result = CliRunner().invoke(
        cli,
        [
            "--checkpoint-id",
            "ckpt_ready_123",
            "--candidate-audit",
            f"mix_v2_p5050={audit}",
            "--output",
            str(output),
            "--min-samples",
            "20",
        ],
    )

    assert result.exit_code == 1
    assert "only 3 review samples available; need at least 20" in result.output
    assert not output.exists()


def test_build_human_read_packet_cli_writes_packet(tmp_path: Path) -> None:
    mix = tmp_path / "mix.json"
    strict = tmp_path / "strict.json"
    output = tmp_path / "human_read.json"
    _write_json(mix, _audit("mix", 11))
    _write_json(strict, _audit("strict", 11))

    result = CliRunner().invoke(
        cli,
        [
            "--checkpoint-id",
            "ckpt_ready_123",
            "--candidate-audit",
            f"mix_v2_p5050={mix}",
            "--candidate-audit",
            f"v02_strict={strict}",
            "--samples-per-audit",
            "11",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "human_read_packet=needs_review samples=22" in result.output
    packet = json.loads(output.read_text())
    assert packet["sample_count"] == 22
