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
import tomllib
from pathlib import Path
from typing import Any

import click

DEFAULT_TEMPLATE = Path(
    "configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml"
)
PLACEHOLDER_CHECKPOINT = "FILL_WITH_READY_SFT_CHECKPOINT_ID"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_checkpoint_id(checkpoint_id: str) -> None:
    if not checkpoint_id or checkpoint_id == PLACEHOLDER_CHECKPOINT:
        raise click.ClickException(
            "checkpoint-id must be a READY SFT checkpoint, not the placeholder"
        )
    if any(char.isspace() for char in checkpoint_id):
        raise click.ClickException("checkpoint-id must not contain whitespace")


def _validate_template(data: dict[str, Any]) -> None:
    if data.get("model") != "Qwen/Qwen3.5-2B":
        raise click.ClickException("template model must be Qwen/Qwen3.5-2B")
    if data.get("checkpoint_id") != PLACEHOLDER_CHECKPOINT:
        raise click.ClickException("template checkpoint_id must be the placeholder")
    if data.get("eval", {}).get("eval_base_model") is not False:
        raise click.ClickException("template eval.eval_base_model must be false")

    env_versions = [env.get("version") for env in data.get("env", [])]
    eval_env_versions = [env.get("version") for env in data.get("eval", {}).get("env", [])]
    if set(env_versions + eval_env_versions) != {"0.3.15"}:
        raise click.ClickException("template env versions must all be 0.3.15")


def _validate_handoff_report(path: Path, checkpoint_id: str) -> None:
    report = _load_json(path)
    if report.get("artifact") != "prime_warm_start_checkpoint_handoff":
        raise click.ClickException("checkpoint handoff report artifact is invalid")
    if not bool(report.get("passed")):
        raise click.ClickException("checkpoint handoff report did not pass")
    report_checkpoint_id = report.get("checkpoint_id")
    if report_checkpoint_id != checkpoint_id:
        raise click.ClickException(
            f"checkpoint handoff report id {report_checkpoint_id} != {checkpoint_id}"
        )


def _validate_promotion_gate_report(path: Path, checkpoint_id: str) -> None:
    report = _load_json(path)
    if report.get("artifact") != "sft_promotion_gate":
        raise click.ClickException("SFT promotion gate report artifact is invalid")
    promotion_gate = report.get("promotion_gate")
    if not isinstance(promotion_gate, dict):
        raise click.ClickException("SFT promotion gate report missing promotion_gate")
    if not bool(promotion_gate.get("passed")):
        raise click.ClickException("SFT promotion gate report did not pass")
    report_checkpoint_id = report.get("checkpoint_id")
    if report_checkpoint_id != checkpoint_id:
        raise click.ClickException(
            f"SFT promotion gate report id {report_checkpoint_id} != {checkpoint_id}"
        )


def _validate_sft_eval_manifest(
    path: Path,
    *,
    checkpoint_id: str,
    template_path: Path,
    output_path: Path,
    checkpoint_handoff_report_path: Path | None,
    promotion_gate_report_path: Path | None,
) -> None:
    report = _load_json(path)
    if report.get("artifact") != "sft_eval_manifest":
        raise click.ClickException("SFT eval manifest artifact is invalid")
    report_checkpoint_id = report.get("checkpoint_id")
    if report_checkpoint_id != checkpoint_id:
        raise click.ClickException(
            f"SFT eval manifest id {report_checkpoint_id} != {checkpoint_id}"
        )

    required = report.get("required_artifacts")
    if not isinstance(required, dict):
        raise click.ClickException("SFT eval manifest missing required_artifacts")
    expected_paths = {
        "after_sft_template": str(template_path),
        "after_sft_config": str(output_path),
    }
    if checkpoint_handoff_report_path is not None:
        expected_paths["checkpoint_handoff"] = str(checkpoint_handoff_report_path)
    if promotion_gate_report_path is not None:
        expected_paths["promotion_gate"] = str(promotion_gate_report_path)

    for key, expected in expected_paths.items():
        actual = required.get(key)
        if actual != expected:
            raise click.ClickException(
                f"SFT eval manifest {key} {actual} != {expected}"
            )

    template_report = report.get("after_sft_template")
    if not isinstance(template_report, dict):
        raise click.ClickException("SFT eval manifest missing after_sft_template")
    if template_report.get("path") != str(template_path):
        raise click.ClickException("SFT eval manifest after_sft_template path mismatch")
    template_sha = template_report.get("sha256")
    if template_sha and template_sha != _sha256(template_path):
        raise click.ClickException(
            "SFT eval manifest after_sft_template sha256 does not match"
        )


def _replace_assignment(text: str, key: str, value: str) -> str:
    pattern = re.compile(rf'^{re.escape(key)}\s*=\s*"[^"]*"', flags=re.MULTILINE)
    replacement = f'{key} = "{value}"'
    updated, count = pattern.subn(replacement, text, count=1)
    if count != 1:
        raise click.ClickException(f"expected exactly one top-level {key} assignment")
    return updated


def prepare_after_sft_config(
    *,
    template_path: Path,
    output_path: Path,
    checkpoint_id: str,
    run_name: str | None = None,
    checkpoint_handoff_report_path: Path | None = None,
    promotion_gate_report_path: Path | None = None,
    sft_eval_manifest_path: Path | None = None,
    allow_unverified_checkpoint: bool = False,
) -> None:
    _validate_checkpoint_id(checkpoint_id)
    if checkpoint_handoff_report_path is None and not allow_unverified_checkpoint:
        raise click.ClickException(
            "checkpoint handoff report is required; pass "
            "--allow-unverified-checkpoint only for template validation"
        )
    if checkpoint_handoff_report_path is not None:
        _validate_handoff_report(checkpoint_handoff_report_path, checkpoint_id)
    if promotion_gate_report_path is None and not allow_unverified_checkpoint:
        raise click.ClickException(
            "SFT promotion gate report is required; pass "
            "--allow-unverified-checkpoint only for template validation"
        )
    if promotion_gate_report_path is not None:
        _validate_promotion_gate_report(promotion_gate_report_path, checkpoint_id)
    if sft_eval_manifest_path is None and not allow_unverified_checkpoint:
        raise click.ClickException(
            "SFT eval manifest is required; pass "
            "--allow-unverified-checkpoint only for template validation"
        )
    if sft_eval_manifest_path is not None:
        _validate_sft_eval_manifest(
            sft_eval_manifest_path,
            checkpoint_id=checkpoint_id,
            template_path=template_path,
            output_path=output_path,
            checkpoint_handoff_report_path=checkpoint_handoff_report_path,
            promotion_gate_report_path=promotion_gate_report_path,
        )
    data = _load_toml(template_path)
    _validate_template(data)

    rendered = template_path.read_text()
    rendered = _replace_assignment(rendered, "checkpoint_id", checkpoint_id)
    if run_name:
        rendered = _replace_assignment(rendered, "name", run_name)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(rendered)

    written = _load_toml(output_path)
    if written.get("checkpoint_id") != checkpoint_id:
        raise click.ClickException("written config checkpoint_id did not validate")


@click.command()
@click.option(
    "--template",
    "template_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_TEMPLATE,
    show_default=True,
    help="SFT-to-RL template with placeholder checkpoint_id.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    required=True,
    help="Concrete hosted RL config to write.",
)
@click.option(
    "--checkpoint-id",
    required=True,
    help="READY SFT checkpoint ID from Prime/open prime-rl handoff.",
)
@click.option(
    "--checkpoint-handoff-report",
    "checkpoint_handoff_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Passing report from verify_prime_warm_start_checkpoint.py.",
)
@click.option(
    "--promotion-gate-report",
    "promotion_gate_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Passing report from build_sft_promotion_gate.py.",
)
@click.option(
    "--sft-eval-manifest",
    "sft_eval_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="SFT eval manifest that defines the post-SFT handoff contract.",
)
@click.option(
    "--run-name",
    default=None,
    help="Optional hosted RL run name override.",
)
@click.option(
    "--allow-unverified-checkpoint",
    is_flag=True,
    help="Allow rendering without handoff/promotion reports for template validation only.",
)
def cli(
    template_path: Path,
    output_path: Path,
    checkpoint_id: str,
    checkpoint_handoff_report_path: Path | None,
    promotion_gate_report_path: Path | None,
    sft_eval_manifest_path: Path | None,
    run_name: str | None,
    allow_unverified_checkpoint: bool,
) -> None:
    """Fill the Qwen 2B after-SFT hosted RL config from a READY SFT checkpoint."""

    prepare_after_sft_config(
        template_path=template_path,
        output_path=output_path,
        checkpoint_id=checkpoint_id,
        checkpoint_handoff_report_path=checkpoint_handoff_report_path,
        promotion_gate_report_path=promotion_gate_report_path,
        sft_eval_manifest_path=sft_eval_manifest_path,
        allow_unverified_checkpoint=allow_unverified_checkpoint,
        run_name=run_name,
    )
    click.echo(f"config={output_path}")
    click.echo(f"checkpoint_id={checkpoint_id}")
    click.echo(f"next=prime --plain train {output_path} --yes --output json")


if __name__ == "__main__":
    cli()
