import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.build_sft_promotion_gate import build_sft_promotion_gate, cli


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n")


def _audit(mean: float, *, failed_counts: dict[str, int] | None = None) -> dict:
    return {
        "artifact": "prime_rollout_audit",
        "rollouts": {
            "row_count": 64,
            "matched_task_count": 64,
            "missing_task_count": 0,
        },
        "recomputed_reward": {
            "count": 64,
            "mean": mean,
            "min": 0.2,
            "max": 0.8,
        },
        "diagnostics": {
            "failed_counts": failed_counts or {},
            "high_rescored_with_failed_diagnostics": 0,
            "high_rescored_with_emoji": 0,
            "high_rescored_with_all_caps": 0,
            "high_rescored_with_option_or_wrapper": 0,
        },
    }


def _detector_report(passed: bool = True) -> dict:
    return {"summary": {"gate_passed": passed, "total_rows": 16}}


def _pangram_alignment_report(passed: bool = True) -> dict:
    return {
        "artifact": "pangram_alignment",
        "summary": {
            "gate_passed": passed,
            "matched_rows": 22,
            "label_disagreement_rows": 0 if passed else 1,
            "failures": [] if passed else ["label_disagreement"],
        },
    }


def _human_read(passed: bool = True, sample_count: int = 25) -> dict:
    return {"passed": passed, "sample_count": sample_count, "notes": "looks direct"}


def _sft_output_report(passed: bool = True) -> dict:
    return {
        "artifact": "prime_sft_output_verification",
        "passed": passed,
        "failures": [] if passed else ["no safetensors files under outputs/sft/weights/step_200"],
        "output_dir": "outputs/prime_sft/qwen35_2b_sft_target_messages_env0314_gate_env0315",
        "step": 200,
    }


def test_build_sft_promotion_gate_passes_when_sft_improves_and_gates_pass(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52, failed_counts={"fake_casual_phrase": 2}))
    _write_json(candidate, _audit(0.58, failed_counts={"fake_casual_phrase": 1}))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is True
    assert report["comparisons"][0]["delta"] == 0.06
    assert report["pangram_alignment"]["path"] is None
    assert report["pangram_alignment"]["passed"] is None
    assert output.exists()


def test_build_sft_promotion_gate_accepts_passing_pangram_alignment(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    pangram = tmp_path / "pangram_alignment.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52))
    _write_json(candidate, _audit(0.58))
    _write_json(detector, _detector_report())
    _write_json(pangram, _pangram_alignment_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        pangram_alignment_report_path=pangram,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is True
    assert report["pangram_alignment"]["path"] == str(pangram)
    assert report["pangram_alignment"]["passed"] is True


def test_build_sft_promotion_gate_rejects_failed_pangram_alignment(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    pangram = tmp_path / "pangram_alignment.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52))
    _write_json(candidate, _audit(0.58))
    _write_json(detector, _detector_report())
    _write_json(pangram, _pangram_alignment_report(passed=False))
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        pangram_alignment_report_path=pangram,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "pangram_alignment_gate_failed" in report["promotion_gate"]["failures"]
    assert report["pangram_alignment"]["passed"] is False


def test_build_sft_promotion_gate_fails_on_strict_drop(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_v02.json"
    candidate = tmp_path / "candidate_v02.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.72))
    _write_json(candidate, _audit(0.66))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"v02_strict": baseline},
        candidate_audits={"v02_strict": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
        max_strict_drop=0.05,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "v02_strict delta -0.060000 below allowed -0.050000" in report[
        "promotion_gate"
    ]["failures"]


def test_build_sft_promotion_gate_fails_when_tracked_diagnostic_increases(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52, failed_counts={"low_specificity_substitution": 0}))
    _write_json(candidate, _audit(0.58, failed_counts={"low_specificity_substitution": 2}))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "diagnostic low_specificity_substitution increased 0 -> 2" in report[
        "promotion_gate"
    ]["failures"]


def test_build_sft_promotion_gate_requires_human_read_count(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52))
    _write_json(candidate, _audit(0.58))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read(sample_count=10))
    _write_json(sft_output, _sft_output_report())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "human_read sample_count 10 outside 20..50" in report["promotion_gate"][
        "failures"
    ]


def test_cli_writes_report_and_fails_on_gate_by_default(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.58))
    _write_json(candidate, _audit(0.50))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report())

    result = CliRunner().invoke(
        cli,
        [
            "--checkpoint-id",
            "ckpt_ready_123",
            "--baseline-audit",
            f"mix_v2_p5050={baseline}",
            "--candidate-audit",
            f"mix_v2_p5050={candidate}",
            "--detector-report",
            str(detector),
            "--human-read",
            str(human),
            "--sft-output-verification",
            str(sft_output),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 1
    assert "SFT promotion gate failed" in result.output
    assert output.exists()


def test_build_sft_promotion_gate_requires_sft_output_verification(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52))
    _write_json(candidate, _audit(0.58))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=None,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "sft_output_verification_missing" in report["promotion_gate"]["failures"]


def test_build_sft_promotion_gate_rejects_failed_sft_output_verification(
    tmp_path: Path,
) -> None:
    baseline = tmp_path / "baseline_mix.json"
    candidate = tmp_path / "candidate_mix.json"
    detector = tmp_path / "detector.json"
    human = tmp_path / "human.json"
    sft_output = tmp_path / "sft_output.json"
    output = tmp_path / "gate.json"
    _write_json(baseline, _audit(0.52))
    _write_json(candidate, _audit(0.58))
    _write_json(detector, _detector_report())
    _write_json(human, _human_read())
    _write_json(sft_output, _sft_output_report(passed=False))

    report = build_sft_promotion_gate(
        checkpoint_id="ckpt_ready_123",
        baseline_audits={"mix_v2_p5050": baseline},
        candidate_audits={"mix_v2_p5050": candidate},
        detector_report_path=detector,
        human_read_path=human,
        sft_output_verification_path=sft_output,
        output_path=output,
    )

    assert report["promotion_gate"]["passed"] is False
    assert "sft_output_verification_failed" in report["promotion_gate"]["failures"]
