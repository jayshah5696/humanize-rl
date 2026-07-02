from __future__ import annotations

import hashlib
import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.build_sft_eval_manifest import build_sft_eval_manifest, cli


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_build_sft_eval_manifest_writes_ordered_handoff(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    detector = tmp_path / "detector.json"
    pangram_bulk = tmp_path / "pangram_bulk_items.json"
    pangram = tmp_path / "pangram_alignment.json"
    after_sft_template = tmp_path / "after_sft_template.toml"
    promotion_root = tmp_path / "promotion"
    output = promotion_root / "manifest.json"
    config.write_text("[model]\nname = \"Qwen/Qwen3.5-2B\"\n")
    detector.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")
    pangram_bulk.write_text(
        json.dumps({"artifact": "pangram_bulk_items", "items": []}) + "\n"
    )
    pangram.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")
    after_sft_template.write_text(
        'model = "Qwen/Qwen3.5-2B"\n'
        'checkpoint_id = "FILL_WITH_READY_SFT_CHECKPOINT_ID"\n'
    )

    report = build_sft_eval_manifest(
        checkpoint_id="ckpt_ready_123",
        promotion_root=promotion_root,
        config_path=config,
        detector_report_path=detector,
        pangram_bulk_items_path=pangram_bulk,
        pangram_alignment_report_path=pangram,
        after_sft_template_path=after_sft_template,
        after_sft_config_path=tmp_path / "after_clean50_<checkpoint_slug>.toml",
        rl_run_name="after-sft-clean50-<checkpoint_slug>",
        step=200,
        output_path=output,
    )

    assert report["artifact"] == "sft_eval_manifest"
    assert report["checkpoint_id"] == "ckpt_ready_123"
    assert report["config"]["exists"] is True
    assert report["detector_report"]["exists"] is True
    assert report["pangram_bulk_items"]["exists"] is True
    assert report["pangram_bulk_items"]["required_now"] is False
    assert report["pangram_bulk_items"]["sha256"] == _sha256(pangram_bulk)
    assert report["pangram_alignment_report"]["exists"] is True
    assert report["pangram_alignment_report"]["required_now"] is False
    assert report["after_sft_template"]["path"] == str(after_sft_template)
    assert report["after_sft_template"]["exists"] is True
    assert report["after_sft_template"]["sha256"] == _sha256(after_sft_template)
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
    assert report["required_artifacts"]["baseline_rollouts"] == {
        "mix_v2_p5050": str(promotion_root / "base_mix_v2_p5050_rollouts.json"),
        "v02_strict": str(promotion_root / "base_v02_strict_rollouts.json"),
        "v03_strict": str(promotion_root / "base_v03_strict_rollouts.json"),
    }
    assert report["audit_specs"]["v02_strict"]["reward_mode"] == "strict"
    verify_command = next(
        row["command"] for row in report["commands"] if row["name"] == "verify_sft_output"
    )
    assert f"--sft-eval-manifest {output}" in verify_command
    collect_command = next(
        row["command"]
        for row in report["commands"]
        if row["name"] == "collect_base_and_sft_rollout_audits"
    )
    assert "--eval-label mix_v2_p5050" in collect_command
    assert "--eval-label v02_strict" in collect_command
    assert "--eval-label v03_strict" in collect_command
    assert "sft_v03_strict_rollouts.json" in collect_command
    pangram_export_command = report["external_detector_handoff"]["pangram_export_command"]
    assert "export_detector_mimic_for_pangram.py" in pangram_export_command
    assert f"--output {pangram_bulk}" in pangram_export_command
    pangram_run_command = report["external_detector_handoff"]["pangram_bulk_run_command"]
    assert "run_pangram_bulk_detection.py" in pangram_run_command
    assert f"--input {pangram_bulk}" in pangram_run_command
    assert "--output runs/detector_mimic/pangram_export.json" in pangram_run_command
    human_read_command = next(
        row["command"]
        for row in report["commands"]
        if row["name"] == "build_human_read_packet"
    )
    assert f"--sft-eval-manifest {output}" in human_read_command
    promotion_command = next(
        row["command"] for row in report["commands"] if row["name"] == "build_promotion_gate"
    )
    assert "--sft-output-verification" in promotion_command
    assert f"--sft-eval-manifest {output}" in promotion_command
    assert f"--pangram-alignment-report {pangram}" in promotion_command
    assert "ckpt_ready_123" in promotion_command
    render_command = next(
        row["command"]
        for row in report["commands"]
        if row["name"] == "render_after_sft_rl_config"
    )
    assert f"--template {after_sft_template}" in render_command
    assert f"--sft-eval-manifest {output}" in render_command
    assert "after_clean50_ckpt-ready-123.toml" in render_command
    assert "after-sft-clean50-ckpt-ready-123" in render_command
    assert "<checkpoint_slug>" not in render_command
    assert (
        report["required_artifacts"]["after_sft_template"] == str(after_sft_template)
    )
    assert (
        report["required_artifacts"]["after_sft_config"]
        == str(tmp_path / "after_clean50_ckpt-ready-123.toml")
    )
    assert output.exists()


def test_build_sft_eval_manifest_cli_writes_template(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    detector = tmp_path / "detector.json"
    after_sft_template = tmp_path / "after_sft_template.toml"
    output = tmp_path / "manifest.json"
    config.write_text("[model]\nname = \"Qwen/Qwen3.5-2B\"\n")
    detector.write_text(json.dumps({"summary": {"gate_passed": True}}) + "\n")
    after_sft_template.write_text(
        'model = "Qwen/Qwen3.5-2B"\n'
        'checkpoint_id = "FILL_WITH_READY_SFT_CHECKPOINT_ID"\n'
    )

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
            "--pangram-bulk-items",
            str(tmp_path / "pangram_bulk_items.json"),
            "--after-sft-template",
            str(after_sft_template),
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
    assert f"--template {after_sft_template}" in render_command
    assert f"--sft-eval-manifest {output}" in render_command
