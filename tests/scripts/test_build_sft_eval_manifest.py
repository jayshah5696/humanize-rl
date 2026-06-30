from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.build_sft_eval_manifest import build_sft_eval_manifest, cli


def test_build_sft_eval_manifest_writes_ordered_handoff(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    detector = tmp_path / "detector.json"
    pangram = tmp_path / "pangram_alignment.json"
    promotion_root = tmp_path / "promotion"
    output = promotion_root / "manifest.json"
    config.write_text("[model]\nname = \"Qwen/Qwen3.5-2B\"\n")
    detector.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")
    pangram.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")

    report = build_sft_eval_manifest(
        checkpoint_id="ckpt_ready_123",
        promotion_root=promotion_root,
        config_path=config,
        detector_report_path=detector,
        pangram_alignment_report_path=pangram,
        after_sft_config_path=tmp_path / "after_clean50.toml",
        rl_run_name="after-sft-clean50-<checkpoint_slug>",
        step=200,
        output_path=output,
    )

    assert report["artifact"] == "sft_eval_manifest"
    assert report["checkpoint_id"] == "ckpt_ready_123"
    assert report["config"]["exists"] is True
    assert report["detector_report"]["exists"] is True
    assert report["pangram_alignment_report"]["exists"] is True
    assert report["pangram_alignment_report"]["required_now"] is False
    assert report["gate_order"] == [
        "verify_sft_output",
        "collect_base_and_sft_rollout_audits",
        "build_human_read_packet",
        "build_promotion_gate",
        "verify_warm_start_checkpoint",
        "render_after_sft_rl_config",
    ]
    assert report["required_artifacts"]["baseline_audits"] == {
        "mix_v2_p5050": str(promotion_root / "base_mix_v2_p5050_audit.json"),
        "v02_strict": str(promotion_root / "base_v02_strict_audit.json"),
        "v03_strict": str(promotion_root / "base_v03_strict_audit.json"),
    }
    promotion_command = next(
        row["command"] for row in report["commands"] if row["name"] == "build_promotion_gate"
    )
    assert "--sft-output-verification" in promotion_command
    assert f"--pangram-alignment-report {pangram}" in promotion_command
    assert "ckpt_ready_123" in promotion_command
    render_command = next(
        row["command"]
        for row in report["commands"]
        if row["name"] == "render_after_sft_rl_config"
    )
    assert "after_clean50.toml" in render_command
    assert "after-sft-clean50-<checkpoint_slug>" in render_command
    assert output.exists()


def test_build_sft_eval_manifest_cli_writes_template(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    detector = tmp_path / "detector.json"
    output = tmp_path / "manifest.json"
    config.write_text("[model]\nname = \"Qwen/Qwen3.5-2B\"\n")
    detector.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")

    result = CliRunner().invoke(
        cli,
        [
            "--checkpoint-id",
            "READY_SFT_CHECKPOINT_ID",
            "--promotion-root",
            str(tmp_path / "promotion"),
            "--config",
            str(config),
            "--detector-report",
            str(detector),
            "--after-sft-config",
            str(tmp_path / "after_sft_clean50.toml"),
            "--rl-run-name",
            "after-sft-clean50-<checkpoint_slug>",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "commands=6" in result.output
    report = json.loads(output.read_text())
    assert report["checkpoint_id"] == "READY_SFT_CHECKPOINT_ID"
    assert report["launch_allowed_when"] == [
        "sft_output_verification.passed=true",
        "human_read.passed=true",
        "promotion_gate.promotion_gate.passed=true",
        "checkpoint_handoff.passed=true",
    ]
    render_command = next(
        row["command"]
        for row in report["commands"]
        if row["name"] == "render_after_sft_rl_config"
    )
    assert "after_sft_clean50.toml" in render_command
    assert "after-sft-clean50-<checkpoint_slug>" in render_command
