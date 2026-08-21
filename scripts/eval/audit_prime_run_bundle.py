from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

from humanize_rl.scoring.detector_mimic import (
    DetectorMimicRow,
    evaluate_detector_mimic_rows,
)

HIGH_REWARD_DIAGNOSTIC_COUNTERS = (
    "high_rescored_with_failed_diagnostics",
    "high_rescored_with_emoji",
    "high_rescored_with_all_caps",
    "high_rescored_with_option_or_wrapper",
)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _load_detector_rows(path: Path) -> list[DetectorMimicRow]:
    rows: list[DetectorMimicRow] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(DetectorMimicRow.model_validate_json(line))
        except ValueError as exc:
            raise click.ClickException(f"{path}:{line_number}: {exc}") from exc
    return rows


def _audit_gate(path: Path, audit: dict[str, Any]) -> tuple[bool, list[str]]:
    failures: list[str] = []
    rollouts = audit.get("rollouts", {})
    diagnostics = audit.get("diagnostics", {})

    matched = int(rollouts.get("matched_task_count") or 0)
    missing = int(rollouts.get("missing_task_count") or 0)
    if matched <= 0:
        failures.append(f"{path.name}: matched_task_count=0")
    if missing:
        failures.append(f"{path.name}: missing_task_count={missing}")

    for key in HIGH_REWARD_DIAGNOSTIC_COUNTERS:
        count = int(diagnostics.get(key) or 0)
        if count:
            failures.append(f"{path.name}: {key}={count}")

    return not failures, failures


def _write_detector_outputs(
    detector_input: Path, output_dir: Path
) -> dict[str, Any]:
    report = evaluate_detector_mimic_rows(
        _load_detector_rows(detector_input), source=str(detector_input)
    )
    (output_dir / "detector_mimic_report.json").write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    (output_dir / "detector_mimic_scored.jsonl").write_text(
        "\n".join(row.model_dump_json() for row in report.rows) + "\n"
    )
    return report.model_dump(mode="json")


def build_prime_run_audit_bundle(
    run_id: str,
    output_dir: Path,
    audit_paths: list[Path],
    detector_input: Path,
) -> dict[str, Any]:
    """Build a promotion-gate report from saved rollout audits and detector rows."""
    if not audit_paths:
        raise ValueError("at least one rollout audit is required")

    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_reports: list[dict[str, Any]] = []
    failures: list[str] = []
    for path in audit_paths:
        audit = _load_json(path)
        gate_passed, audit_failures = _audit_gate(path, audit)
        failures.extend(audit_failures)
        rollout_reports.append(
            {
                "path": str(path),
                "gate_passed": gate_passed,
                "rollouts": audit.get("rollouts", {}),
                "recomputed_reward": audit.get("recomputed_reward", {}),
                "diagnostics": audit.get("diagnostics", {}),
            }
        )

    detector_report = _write_detector_outputs(detector_input, output_dir)
    detector_passed = bool(detector_report["summary"]["gate_passed"])
    if not detector_passed:
        failures.append("detector_mimic_gate_failed")

    bundle = {
        "artifact": "prime_run_audit_bundle",
        "run_id": run_id,
        "audit_paths": [str(path) for path in audit_paths],
        "detector_input": str(detector_input),
        "rollout_audits": rollout_reports,
        "detector_mimic": detector_report,
        "promotion_gate": {
            "passed": not failures,
            "failures": failures,
        },
    }
    (output_dir / "bundle_report.json").write_text(
        json.dumps(bundle, indent=2, sort_keys=True) + "\n"
    )
    return bundle


@click.command(context_settings={"show_default": True})
@click.option("--run-id", required=True, help="Prime Hosted Training run ID.")
@click.option(
    "--audit",
    "audit_paths",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Saved rollout audit JSON. Pass once per step.",
)
@click.option(
    "--detector-input",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/eval/detector_mimic_v01.jsonl"),
    help="Frozen detector-mimic JSONL input.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Directory for bundle_report.json and detector outputs.",
)
@click.option(
    "--no-fail-on-gate",
    is_flag=True,
    help="Write the bundle but return success even when the gate fails.",
)
def cli(
    run_id: str,
    audit_paths: tuple[Path, ...],
    detector_input: Path,
    output_dir: Path | None,
    no_fail_on_gate: bool,
) -> None:
    """Combine rollout audits and detector-mimic results into one promotion gate."""
    resolved_output_dir = output_dir or Path("runs/prime_training_smoke") / run_id
    try:
        bundle = build_prime_run_audit_bundle(
            run_id=run_id,
            output_dir=resolved_output_dir,
            audit_paths=list(audit_paths),
            detector_input=detector_input,
        )
    except ValueError as exc:
        raise click.ClickException(str(exc)) from exc

    passed = bool(bundle["promotion_gate"]["passed"])
    click.echo(
        "bundle_gate={gate} audits={audits} detector={detector} report={report}".format(
            gate="pass" if passed else "fail",
            audits=len(bundle["rollout_audits"]),
            detector="pass"
            if bundle["detector_mimic"]["summary"]["gate_passed"]
            else "fail",
            report=resolved_output_dir / "bundle_report.json",
        )
    )
    if not passed and not no_fail_on_gate:
        raise click.ClickException("Prime run audit bundle gate failed")


if __name__ == "__main__":
    cli()
