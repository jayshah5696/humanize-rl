# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import click

DEFAULT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml"
)
DEFAULT_DETECTOR_REPORT = Path("runs/detector_mimic/detector_mimic_v01_report.json")
DEFAULT_PANGRAM_BULK_ITEMS = Path("runs/detector_mimic/pangram_bulk_items.json")
DEFAULT_PANGRAM_ALIGNMENT_REPORT = Path(
    "runs/detector_mimic/pangram_alignment_report.json"
)
DEFAULT_TEMPLATE_ROOT = Path("runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315")
DEFAULT_OUTPUT = DEFAULT_TEMPLATE_ROOT / "sft_eval_manifest.json"
DEFAULT_AFTER_SFT_TEMPLATE = Path(
    "configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml"
)
DEFAULT_AFTER_SFT_CONFIG = Path(
    "configs/prime/qwen35_2b_p5050_after_sft_env0315_<checkpoint_slug>.toml"
)
DEFAULT_RL_RUN_NAME = "humanize-p5050-qwen35-2b-after-sft-env0315-<checkpoint_slug>"
PLACEHOLDER_CHECKPOINT = "READY_SFT_CHECKPOINT_ID"
EVAL_LABELS = ("mix_v2_p5050", "v02_strict", "v03_strict")
AUDIT_SPECS: dict[str, dict[str, str]] = {
    "mix_v2_p5050": {
        "taskset": "data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl",
        "reward_mode": "p50_50_no_penalty",
    },
    "v02_strict": {
        "taskset": "environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl",
        "reward_mode": "strict",
    },
    "v03_strict": {
        "taskset": "data/rl/humanize_tasks_v03_filtered.jsonl",
        "reward_mode": "strict",
    },
}


def _check_path(path: Path, *, required: bool) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.exists(),
        "required_now": required,
    }
    if path.is_file():
        report["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return report


def _audit_paths(root: Path, prefix: str) -> dict[str, str]:
    return {label: str(root / f"{prefix}_{label}_audit.json") for label in EVAL_LABELS}


def _rollout_paths(root: Path, prefix: str) -> dict[str, str]:
    return {label: str(root / f"{prefix}_{label}_rollouts.json") for label in EVAL_LABELS}


def _command(lines: list[str]) -> str:
    return " \\\n  ".join(lines)


def _checkpoint_slug(checkpoint_id: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", checkpoint_id).strip("-").lower()
    if not slug:
        raise click.ClickException("checkpoint-id does not produce a usable slug")
    return slug


def _materialize_checkpoint_slug(value: str, checkpoint_id: str) -> str:
    if checkpoint_id == PLACEHOLDER_CHECKPOINT:
        return value
    return value.replace("<checkpoint_slug>", _checkpoint_slug(checkpoint_id))


def _audit_command_lines(
    *,
    baseline_rollouts: dict[str, str],
    candidate_rollouts: dict[str, str],
    baseline_audits: dict[str, str],
    candidate_audits: dict[str, str],
) -> list[str]:
    lines = [
        "Save base and SFT rollout JSON files at required_artifacts.baseline_rollouts and required_artifacts.candidate_rollouts, then run:",
    ]
    for prefix, rollouts, audits in (
        ("base", baseline_rollouts, baseline_audits),
        ("sft", candidate_rollouts, candidate_audits),
    ):
        for label in EVAL_LABELS:
            lines.append(
                "PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-audit-min "
                "uv run --no-project --python 3.12 "
                "--with pydantic --with click --with numpy "
                "--with 'scikit-learn>=1.8,<1.9' "
                "scripts/eval/audit_prime_rollouts.py "
                f"--eval-label {label} "
                f"--rollouts {rollouts[label]} "
                f"--output {audits[label]} "
                "--top-n 50"
                f"  # {prefix}"
            )
    return lines


def build_sft_eval_manifest(
    *,
    checkpoint_id: str,
    output_path: Path,
    promotion_root: Path,
    config_path: Path = DEFAULT_CONFIG,
    detector_report_path: Path = DEFAULT_DETECTOR_REPORT,
    pangram_bulk_items_path: Path = DEFAULT_PANGRAM_BULK_ITEMS,
    pangram_alignment_report_path: Path | None = None,
    after_sft_template_path: Path = DEFAULT_AFTER_SFT_TEMPLATE,
    after_sft_config_path: Path = DEFAULT_AFTER_SFT_CONFIG,
    rl_run_name: str = DEFAULT_RL_RUN_NAME,
    step: int = 200,
) -> dict[str, Any]:
    """Write the post-SFT eval/promotion manifest before RL handoff."""
    if not checkpoint_id:
        raise click.ClickException("checkpoint-id must not be empty")

    sft_output_report = promotion_root / "sft_output_verification.json"
    human_read_packet = promotion_root / "human_read_packet.json"
    promotion_gate = promotion_root / "promotion_gate.json"
    checkpoint_handoff = promotion_root / "checkpoint_handoff.json"
    after_sft_config = Path(
        _materialize_checkpoint_slug(str(after_sft_config_path), checkpoint_id)
    )
    materialized_rl_run_name = _materialize_checkpoint_slug(rl_run_name, checkpoint_id)
    baseline_audits = _audit_paths(promotion_root, "base")
    candidate_audits = _audit_paths(promotion_root, "sft")
    baseline_rollouts = _rollout_paths(promotion_root, "base")
    candidate_rollouts = _rollout_paths(promotion_root, "sft")

    promotion_gate_command = [
        "uv run scripts/eval/build_sft_promotion_gate.py",
        f"--sft-eval-manifest {output_path}",
        f"--checkpoint-id {checkpoint_id}",
        f"--baseline-audit mix_v2_p5050={baseline_audits['mix_v2_p5050']}",
        f"--baseline-audit v02_strict={baseline_audits['v02_strict']}",
        f"--baseline-audit v03_strict={baseline_audits['v03_strict']}",
        f"--candidate-audit mix_v2_p5050={candidate_audits['mix_v2_p5050']}",
        f"--candidate-audit v02_strict={candidate_audits['v02_strict']}",
        f"--candidate-audit v03_strict={candidate_audits['v03_strict']}",
        f"--detector-report {detector_report_path}",
    ]
    if pangram_alignment_report_path is not None:
        promotion_gate_command.append(
            f"--pangram-alignment-report {pangram_alignment_report_path}"
        )
    promotion_gate_command.extend(
        [
            f"--human-read {human_read_packet}",
            f"--sft-output-verification {sft_output_report}",
            f"--output {promotion_gate}",
        ]
    )

    commands = [
        {
            "name": "verify_sft_output",
            "produces": str(sft_output_report),
            "command": _command(
                [
                    "uv run scripts/train/verify_prime_sft_output.py",
                    f"--config {config_path}",
                    f"--sft-eval-manifest {output_path}",
                    f"--step {step}",
                    f"--output {sft_output_report}",
                ]
            ),
        },
        {
            "name": "collect_base_and_sft_rollout_audits",
            "produces": [*baseline_audits.values(), *candidate_audits.values()],
            "requires": [*baseline_rollouts.values(), *candidate_rollouts.values()],
            "command": "\n".join(
                _audit_command_lines(
                    baseline_rollouts=baseline_rollouts,
                    candidate_rollouts=candidate_rollouts,
                    baseline_audits=baseline_audits,
                    candidate_audits=candidate_audits,
                )
            ),
        },
        {
            "name": "build_human_read_packet",
            "produces": str(human_read_packet),
            "command": _command(
                [
                    "uv run scripts/eval/build_sft_human_read_packet.py",
                    f"--sft-eval-manifest {output_path}",
                    f"--checkpoint-id {checkpoint_id}",
                    f"--candidate-audit mix_v2_p5050={candidate_audits['mix_v2_p5050']}",
                    f"--candidate-audit v02_strict={candidate_audits['v02_strict']}",
                    f"--candidate-audit v03_strict={candidate_audits['v03_strict']}",
                    f"--output {human_read_packet}",
                ]
            ),
        },
        {
            "name": "build_promotion_gate",
            "produces": str(promotion_gate),
            "command": _command(promotion_gate_command),
        },
        {
            "name": "verify_warm_start_checkpoint",
            "produces": str(checkpoint_handoff),
            "command": _command(
                [
                    "uv run scripts/train/verify_prime_warm_start_checkpoint.py",
                    "--run-id <PRIME_RUN_ID_WITH_READY_CHECKPOINT>",
                    f"--checkpoint-id {checkpoint_id}",
                    f"--output {checkpoint_handoff}",
                ]
            ),
        },
        {
            "name": "render_after_sft_rl_config",
            "produces": str(after_sft_config),
            "command": _command(
                [
                    "uv run scripts/train/prepare_prime_sft_to_rl_config.py",
                    f"--template {after_sft_template_path}",
                    f"--sft-eval-manifest {output_path}",
                    f"--checkpoint-id {checkpoint_id}",
                    f"--checkpoint-handoff-report {checkpoint_handoff}",
                    f"--promotion-gate-report {promotion_gate}",
                    f"--output {after_sft_config}",
                    f"--run-name {materialized_rl_run_name}",
                ]
            ),
        },
    ]

    report = {
        "artifact": "sft_eval_manifest",
        "checkpoint_id": checkpoint_id,
        "checkpoint_slug": None
        if checkpoint_id == PLACEHOLDER_CHECKPOINT
        else _checkpoint_slug(checkpoint_id),
        "config": _check_path(config_path, required=True),
        "detector_report": _check_path(detector_report_path, required=True),
        "pangram_bulk_items": _check_path(pangram_bulk_items_path, required=False),
        "pangram_alignment_report": _check_path(
            pangram_alignment_report_path or DEFAULT_PANGRAM_ALIGNMENT_REPORT,
            required=False,
        ),
        "after_sft_template": _check_path(after_sft_template_path, required=True),
        "audit_specs": AUDIT_SPECS,
        "promotion_root": str(promotion_root),
        "required_artifacts": {
            "baseline_rollouts": baseline_rollouts,
            "candidate_rollouts": candidate_rollouts,
            "baseline_audits": baseline_audits,
            "candidate_audits": candidate_audits,
            "sft_output_verification": str(sft_output_report),
            "human_read_packet": str(human_read_packet),
            "promotion_gate": str(promotion_gate),
            "checkpoint_handoff": str(checkpoint_handoff),
            "after_sft_template": str(after_sft_template_path),
            "after_sft_config": str(after_sft_config),
        },
        "external_detector_handoff": {
            "pangram_sdk_method": "Pangram.submit_bulk(items=payload['items'])",
            "pangram_bulk_items": str(pangram_bulk_items_path),
            "pangram_export": "runs/detector_mimic/pangram_export.json",
            "pangram_export_command": _command(
                [
                    "uv run scripts/eval/export_detector_mimic_for_pangram.py",
                    "--input data/eval/detector_mimic_v01.jsonl",
                    f"--output {pangram_bulk_items_path}",
                ]
            ),
            "pangram_bulk_run_command": _command(
                [
                    "uv run scripts/eval/run_pangram_bulk_detection.py",
                    f"--input {pangram_bulk_items_path}",
                    "--output runs/detector_mimic/pangram_export.json",
                    "--submit-report runs/detector_mimic/pangram_bulk_submit.json",
                    "--timeout 3600",
                    "--poll-interval 2",
                ]
            ),
            "pangram_alignment_command": _command(
                [
                    "uv run scripts/eval/compare_detector_mimic_to_pangram.py",
                    "--input data/eval/detector_mimic_v01.jsonl",
                    "--pangram-output runs/detector_mimic/pangram_export.json",
                    f"--output {pangram_alignment_report_path or DEFAULT_PANGRAM_ALIGNMENT_REPORT}",
                ]
            ),
        },
        "commands": commands,
        "gate_order": [
            "verify_sft_output",
            "collect_base_and_sft_rollout_audits",
            "build_human_read_packet",
            "build_promotion_gate",
            "verify_warm_start_checkpoint",
            "render_after_sft_rl_config",
        ],
        "launch_allowed_when": [
            "sft_output_verification.passed=true",
            "human_read.passed=true",
            "promotion_gate.promotion_gate.passed=true",
            "checkpoint_handoff.passed=true",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option("--checkpoint-id", default=PLACEHOLDER_CHECKPOINT, show_default=True)
@click.option(
    "--promotion-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_TEMPLATE_ROOT,
    help="Directory where SFT promotion artifacts will live.",
)
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_CONFIG,
    help="Prime prime-rl SFT config.",
)
@click.option(
    "--detector-report",
    "detector_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_DETECTOR_REPORT,
    help="Detector mimic report JSON.",
)
@click.option(
    "--pangram-bulk-items",
    "pangram_bulk_items_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_PANGRAM_BULK_ITEMS,
    help="Optional Pangram SDK bulk-items payload JSON.",
)
@click.option(
    "--pangram-alignment-report",
    "pangram_alignment_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional Pangram alignment report JSON.",
)
@click.option(
    "--after-sft-template",
    "after_sft_template_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_AFTER_SFT_TEMPLATE,
    help="Hosted RL template to render after SFT promotion passes.",
)
@click.option(
    "--after-sft-config",
    "after_sft_config_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_AFTER_SFT_CONFIG,
    help="Future hosted RL config path to render after SFT promotion passes.",
)
@click.option(
    "--rl-run-name",
    default=DEFAULT_RL_RUN_NAME,
    help="Future hosted RL run name template for the rendered config.",
)
@click.option("--step", default=200, type=int, show_default=True)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT,
    help="Manifest JSON output.",
)
def cli(
    checkpoint_id: str,
    promotion_root: Path,
    config_path: Path,
    detector_report_path: Path,
    pangram_bulk_items_path: Path,
    pangram_alignment_report_path: Path | None,
    after_sft_template_path: Path,
    after_sft_config_path: Path,
    rl_run_name: str,
    step: int,
    output_path: Path,
) -> None:
    """Write the deterministic post-SFT eval and promotion manifest."""
    report = build_sft_eval_manifest(
        checkpoint_id=checkpoint_id,
        promotion_root=promotion_root,
        config_path=config_path,
        detector_report_path=detector_report_path,
        pangram_bulk_items_path=pangram_bulk_items_path,
        pangram_alignment_report_path=pangram_alignment_report_path,
        after_sft_template_path=after_sft_template_path,
        after_sft_config_path=after_sft_config_path,
        rl_run_name=rl_run_name,
        step=step,
        output_path=output_path,
    )
    click.echo(
        "sft_eval_manifest={report} checkpoint_id={checkpoint} commands={commands}".format(
            report=output_path,
            checkpoint=report["checkpoint_id"],
            commands=len(report["commands"]),
        )
    )


if __name__ == "__main__":
    cli()
