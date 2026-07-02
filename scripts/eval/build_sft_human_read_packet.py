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

DEFAULT_MIN_SAMPLES = 20
DEFAULT_MAX_SAMPLES = 50
DEFAULT_SAMPLES_PER_AUDIT = 20
PLACEHOLDER_CHECKPOINT = "FILL_WITH_READY_SFT_CHECKPOINT_ID"
PLACEHOLDER_CHECKPOINTS = frozenset(
    {
        PLACEHOLDER_CHECKPOINT,
        "READY_SFT_CHECKPOINT_ID",
    }
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


def _validate_checkpoint_id(checkpoint_id: str) -> None:
    if not checkpoint_id or checkpoint_id in PLACEHOLDER_CHECKPOINTS:
        raise click.ClickException(
            "checkpoint-id must be a READY SFT checkpoint, not the placeholder"
        )
    if any(char.isspace() for char in checkpoint_id):
        raise click.ClickException("checkpoint-id must not contain whitespace")


def _string_map(paths: dict[str, Path]) -> dict[str, str]:
    return {label: str(path) for label, path in sorted(paths.items())}


def _eval_manifest_gate(
    path: Path | None,
    *,
    checkpoint_id: str,
    candidate_audits: dict[str, Path],
    output_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, []

    report = _load_json(path)
    failures: list[str] = []
    if report.get("artifact") != "sft_eval_manifest":
        failures.append("SFT eval manifest artifact is invalid")
    if report.get("checkpoint_id") != checkpoint_id:
        failures.append("SFT eval manifest checkpoint_id mismatch")

    required = report.get("required_artifacts")
    if not isinstance(required, dict):
        failures.append("SFT eval manifest missing required_artifacts object")
    else:
        if required.get("candidate_audits") != _string_map(candidate_audits):
            failures.append("SFT eval manifest candidate_audits path mismatch")
        if required.get("human_read_packet") != str(output_path):
            failures.append("SFT eval manifest human_read_packet path mismatch")

    return report, failures


def _candidate_samples(
    *,
    label: str,
    path: Path,
    samples_per_audit: int,
) -> list[dict[str, Any]]:
    audit = _load_json(path)
    top_samples = audit.get("top_samples", [])
    if not isinstance(top_samples, list):
        raise click.ClickException(f"{path} top_samples must be a list")

    samples: list[dict[str, Any]] = []
    for index, row in enumerate(top_samples[:samples_per_audit], start=1):
        if not isinstance(row, dict):
            continue
        samples.append(
            {
                "label": label,
                "audit_path": str(path),
                "rank_in_audit": index,
                "task_id": row.get("task_id"),
                "problem_id": row.get("problem_id"),
                "sample_id": row.get("sample_id"),
                "rescored_reward": row.get("rescored_reward"),
                "failed_diagnostics": row.get("failed_diagnostics", []),
                "completion_preview": row.get("completion_preview", ""),
                "review_decision": "unreviewed",
                "review_notes": "",
            }
        )
    return samples


def build_human_read_packet(
    *,
    checkpoint_id: str,
    candidate_audits: dict[str, Path],
    output_path: Path,
    sft_eval_manifest_path: Path | None = None,
    samples_per_audit: int = DEFAULT_SAMPLES_PER_AUDIT,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    max_samples: int = DEFAULT_MAX_SAMPLES,
) -> dict[str, Any]:
    """Build a bounded human-read packet from SFT candidate audit top samples."""
    _validate_checkpoint_id(checkpoint_id)
    if min_samples < 1:
        raise click.ClickException("min-samples must be positive")
    if max_samples < min_samples:
        raise click.ClickException("max-samples must be >= min-samples")
    if samples_per_audit < 1:
        raise click.ClickException("samples-per-audit must be positive")
    sft_eval_manifest, manifest_failures = _eval_manifest_gate(
        sft_eval_manifest_path,
        checkpoint_id=checkpoint_id,
        candidate_audits=candidate_audits,
        output_path=output_path,
    )
    if manifest_failures:
        raise click.ClickException("; ".join(manifest_failures))

    samples: list[dict[str, Any]] = []
    for label, path in sorted(candidate_audits.items()):
        samples.extend(
            _candidate_samples(
                label=label,
                path=path,
                samples_per_audit=samples_per_audit,
            )
        )

    samples = samples[:max_samples]
    if len(samples) < min_samples:
        raise click.ClickException(
            f"only {len(samples)} review samples available; need at least {min_samples}. "
            "Rerun rollout audits with a larger --top-n."
        )

    packet = {
        "artifact": "sft_human_read_packet",
        "checkpoint_id": checkpoint_id,
        "passed": False,
        "review_status": "needs_review",
        "sample_count": len(samples),
        "candidate_audits": {
            label: str(path) for label, path in sorted(candidate_audits.items())
        },
        "sft_eval_manifest": {
            "path": str(sft_eval_manifest_path)
            if sft_eval_manifest_path is not None
            else None,
            "checkpoint_id": sft_eval_manifest.get("checkpoint_id")
            if sft_eval_manifest
            else None,
            "promotion_root": sft_eval_manifest.get("promotion_root")
            if sft_eval_manifest
            else None,
        },
        "review_requirements": {
            "set_passed_true_only_after_review": True,
            "min_samples": min_samples,
            "max_samples": max_samples,
            "blockers": [
                "template wrapper text",
                "fake casual filler",
                "emoji or all-caps artifacts",
                "option-menu answers",
                "lost user constraints",
            ],
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n")
    return packet


@click.command(context_settings={"show_default": True})
@click.option("--checkpoint-id", required=True, help="READY SFT checkpoint ID.")
@click.option(
    "--candidate-audit",
    multiple=True,
    required=True,
    help="SFT candidate rollout audit as LABEL=PATH.",
)
@click.option(
    "--sft-eval-manifest",
    "sft_eval_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="SFT eval manifest that defines expected human-read artifact paths.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Human-read packet JSON to write.",
)
@click.option("--samples-per-audit", default=DEFAULT_SAMPLES_PER_AUDIT, type=int)
@click.option("--min-samples", default=DEFAULT_MIN_SAMPLES, type=int)
@click.option("--max-samples", default=DEFAULT_MAX_SAMPLES, type=int)
def cli(
    checkpoint_id: str,
    candidate_audit: tuple[str, ...],
    sft_eval_manifest_path: Path | None,
    output_path: Path,
    samples_per_audit: int,
    min_samples: int,
    max_samples: int,
) -> None:
    """Build the human-read packet required before SFT-to-RL promotion."""
    packet = build_human_read_packet(
        checkpoint_id=checkpoint_id,
        candidate_audits=dict(_parse_labeled_path(value) for value in candidate_audit),
        sft_eval_manifest_path=sft_eval_manifest_path,
        output_path=output_path,
        samples_per_audit=samples_per_audit,
        min_samples=min_samples,
        max_samples=max_samples,
    )
    click.echo(
        "human_read_packet={status} samples={samples} report={report}".format(
            status=packet["review_status"],
            samples=packet["sample_count"],
            report=output_path,
        )
    )


if __name__ == "__main__":
    cli()
