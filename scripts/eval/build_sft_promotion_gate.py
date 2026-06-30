# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

TRACKED_DIAGNOSTICS = (
    "wrapper_phrase",
    "option_menu",
    "emoji",
    "all_caps",
    "fake_casual_phrase",
    "low_specificity_substitution",
)
HIGH_REWARD_COUNTERS = (
    "high_rescored_with_failed_diagnostics",
    "high_rescored_with_emoji",
    "high_rescored_with_all_caps",
    "high_rescored_with_option_or_wrapper",
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise click.BadParameter("use LABEL=PATH")
    label, raw_path = value.split("=", 1)
    if not label:
        raise click.BadParameter("label must not be empty")
    return label, Path(raw_path)


def _audit_summary(path: Path) -> dict[str, Any]:
    audit = _load_json(path)
    rollouts = audit.get("rollouts", {})
    diagnostics = audit.get("diagnostics", {})
    reward = audit.get("recomputed_reward", {})
    failed_counts = diagnostics.get("failed_counts", {})
    if not isinstance(failed_counts, dict):
        failed_counts = {}
    return {
        "path": str(path),
        "mean_reward": float(reward.get("mean") or 0.0),
        "matched_task_count": int(rollouts.get("matched_task_count") or 0),
        "missing_task_count": int(rollouts.get("missing_task_count") or 0),
        "failed_counts": {
            name: int(failed_counts.get(name) or 0) for name in TRACKED_DIAGNOSTICS
        },
        "high_reward_counters": {
            name: int(diagnostics.get(name) or 0) for name in HIGH_REWARD_COUNTERS
        },
    }


def _detector_passed(path: Path) -> bool:
    report = _load_json(path)
    return bool(report.get("summary", {}).get("gate_passed"))


def _pangram_alignment_gate(path: Path | None) -> tuple[dict[str, Any], list[str]]:
    if path is None:
        return {"path": None, "passed": None, "report": None}, []

    report = _load_json(path)
    passed = bool(report.get("summary", {}).get("gate_passed"))
    gate = {
        "path": str(path),
        "passed": passed,
        "report": report,
    }
    if passed:
        return gate, []
    return gate, ["pangram_alignment_gate_failed"]


def _human_read_gate(path: Path) -> tuple[dict[str, Any], list[str]]:
    report = _load_json(path)
    failures: list[str] = []
    sample_count = int(report.get("sample_count") or 0)
    if not bool(report.get("passed")):
        failures.append("human_read passed=false")
    if not 20 <= sample_count <= 50:
        failures.append(f"human_read sample_count {sample_count} outside 20..50")
    return report, failures


def _sft_output_gate(path: Path | None) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, ["sft_output_verification_missing"]

    report = _load_json(path)
    failures: list[str] = []
    if report.get("artifact") != "prime_sft_output_verification":
        failures.append("sft_output_verification_artifact_invalid")
    if not bool(report.get("passed")):
        failures.append("sft_output_verification_failed")
    return report, failures


def _comparison_allowed_drop(
    label: str,
    min_p50_delta: float,
    max_strict_drop: float,
) -> float:
    if "p50" in label or "mix" in label:
        return min_p50_delta
    return -max_strict_drop


def build_sft_promotion_gate(
    *,
    checkpoint_id: str,
    baseline_audits: dict[str, Path],
    candidate_audits: dict[str, Path],
    detector_report_path: Path,
    human_read_path: Path,
    sft_output_verification_path: Path | None,
    output_path: Path,
    pangram_alignment_report_path: Path | None = None,
    min_p50_delta: float = 0.0,
    max_strict_drop: float = 0.0,
) -> dict[str, Any]:
    """Build a promotion report proving an SFT checkpoint is ready for RL."""
    failures: list[str] = []
    if not checkpoint_id or checkpoint_id == "FILL_WITH_READY_SFT_CHECKPOINT_ID":
        failures.append("checkpoint_id is missing or placeholder")

    sft_output, sft_output_failures = _sft_output_gate(sft_output_verification_path)
    failures.extend(sft_output_failures)

    if set(baseline_audits) != set(candidate_audits):
        failures.append("baseline/candidate audit labels differ")

    comparisons: list[dict[str, Any]] = []
    baseline_summaries: dict[str, dict[str, Any]] = {}
    candidate_summaries: dict[str, dict[str, Any]] = {}
    for label in sorted(set(baseline_audits) & set(candidate_audits)):
        baseline = _audit_summary(baseline_audits[label])
        candidate = _audit_summary(candidate_audits[label])
        baseline_summaries[label] = baseline
        candidate_summaries[label] = candidate

        if candidate["matched_task_count"] <= 0:
            failures.append(f"{label} candidate matched_task_count=0")
        if candidate["missing_task_count"]:
            failures.append(f"{label} candidate missing_task_count={candidate['missing_task_count']}")

        delta = round(candidate["mean_reward"] - baseline["mean_reward"], 6)
        allowed = _comparison_allowed_drop(label, min_p50_delta, max_strict_drop)
        if delta < allowed:
            failures.append(f"{label} delta {delta:+.6f} below allowed {allowed:+.6f}")

        comparisons.append(
            {
                "label": label,
                "baseline_mean": baseline["mean_reward"],
                "candidate_mean": candidate["mean_reward"],
                "delta": delta,
                "allowed_delta": allowed,
                "passed": delta >= allowed,
            }
        )

        for name, count in candidate["high_reward_counters"].items():
            if count:
                failures.append(f"{label} candidate {name}={count}")

        for name in TRACKED_DIAGNOSTICS:
            base_count = baseline["failed_counts"][name]
            cand_count = candidate["failed_counts"][name]
            if cand_count > base_count:
                failures.append(f"diagnostic {name} increased {base_count} -> {cand_count}")

    detector_passed = _detector_passed(detector_report_path)
    if not detector_passed:
        failures.append("detector_mimic_gate_failed")

    pangram_alignment, pangram_failures = _pangram_alignment_gate(
        pangram_alignment_report_path
    )
    failures.extend(pangram_failures)

    human_read, human_failures = _human_read_gate(human_read_path)
    failures.extend(human_failures)

    report = {
        "artifact": "sft_promotion_gate",
        "checkpoint_id": checkpoint_id,
        "baseline_audits": {
            label: str(path) for label, path in sorted(baseline_audits.items())
        },
        "candidate_audits": {
            label: str(path) for label, path in sorted(candidate_audits.items())
        },
        "baseline_summaries": baseline_summaries,
        "candidate_summaries": candidate_summaries,
        "comparisons": comparisons,
        "detector_mimic": {
            "path": str(detector_report_path),
            "passed": detector_passed,
        },
        "pangram_alignment": pangram_alignment,
        "human_read": human_read,
        "sft_output_verification": {
            "path": str(sft_output_verification_path)
            if sft_output_verification_path is not None
            else None,
            "passed": bool(sft_output and sft_output.get("passed")),
            "report": sft_output,
        },
        "promotion_gate": {
            "passed": not failures,
            "failures": failures,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option("--checkpoint-id", required=True, help="READY SFT checkpoint ID.")
@click.option(
    "--baseline-audit",
    multiple=True,
    required=True,
    help="Baseline rollout audit as LABEL=PATH.",
)
@click.option(
    "--candidate-audit",
    multiple=True,
    required=True,
    help="SFT candidate rollout audit as LABEL=PATH.",
)
@click.option(
    "--detector-report",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Detector mimic report JSON.",
)
@click.option(
    "--human-read",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Human read JSON with passed and sample_count.",
)
@click.option(
    "--pangram-alignment-report",
    "pangram_alignment_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional passing report from compare_detector_mimic_to_pangram.py.",
)
@click.option(
    "--sft-output-verification",
    "sft_output_verification_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Passing report from verify_prime_sft_output.py.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Promotion gate report JSON.",
)
@click.option("--min-p50-delta", default=0.0, type=float)
@click.option("--max-strict-drop", default=0.0, type=float)
@click.option("--no-fail-on-gate", is_flag=True)
def cli(
    checkpoint_id: str,
    baseline_audit: tuple[str, ...],
    candidate_audit: tuple[str, ...],
    detector_report: Path,
    human_read: Path,
    pangram_alignment_report_path: Path | None,
    sft_output_verification_path: Path,
    output_path: Path,
    min_p50_delta: float,
    max_strict_drop: float,
    no_fail_on_gate: bool,
) -> None:
    """Build the required SFT-before-RL promotion gate."""
    baseline = dict(_parse_labeled_path(value) for value in baseline_audit)
    candidate = dict(_parse_labeled_path(value) for value in candidate_audit)
    report = build_sft_promotion_gate(
        checkpoint_id=checkpoint_id,
        baseline_audits=baseline,
        candidate_audits=candidate,
        detector_report_path=detector_report,
        pangram_alignment_report_path=pangram_alignment_report_path,
        human_read_path=human_read,
        sft_output_verification_path=sft_output_verification_path,
        output_path=output_path,
        min_p50_delta=min_p50_delta,
        max_strict_drop=max_strict_drop,
    )
    passed = bool(report["promotion_gate"]["passed"])
    click.echo(
        "sft_promotion_gate={gate} comparisons={comparisons} report={report}".format(
            gate="pass" if passed else "fail",
            comparisons=len(report["comparisons"]),
            report=output_path,
        )
    )
    if not passed and not no_fail_on_gate:
        raise click.ClickException("SFT promotion gate failed")


if __name__ == "__main__":
    cli()
