# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import click

DEFAULT_EXPECTED_MODEL = "Qwen/Qwen3.5-2B"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _run_prime(args: list[str]) -> dict[str, Any]:
    try:
        result = subprocess.run(
            ["prime", "--plain", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        raise click.ClickException(f"prime CLI unavailable: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise click.ClickException(detail or f"prime {' '.join(args)} failed")

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"prime returned non-JSON output: {exc}") from exc
    if not isinstance(data, dict):
        raise click.ClickException("prime returned non-object JSON")
    return data


def _run_from_payload(run_payload: dict[str, Any]) -> dict[str, Any]:
    run = run_payload.get("run")
    if not isinstance(run, dict):
        raise click.ClickException("run payload must contain a run object")
    return run


def _checkpoint_from_payload(
    checkpoints_payload: dict[str, Any], checkpoint_id: str
) -> dict[str, Any] | None:
    checkpoints = checkpoints_payload.get("checkpoints")
    if not isinstance(checkpoints, list):
        raise click.ClickException("checkpoints payload must contain checkpoints list")
    for checkpoint in checkpoints:
        if isinstance(checkpoint, dict) and checkpoint.get("id") == checkpoint_id:
            return checkpoint
    return None


def build_checkpoint_handoff(
    *,
    run_payload: dict[str, Any],
    checkpoints_payload: dict[str, Any],
    checkpoint_id: str,
    expected_model: str,
    output_path: Path,
) -> dict[str, Any]:
    """Verify a Prime checkpoint can be used as a hosted RL warm start."""
    failures: list[str] = []
    run = _run_from_payload(run_payload)
    run_id = str(run.get("id") or "")
    base_model = run.get("base_model") or run.get("model")
    if base_model != expected_model:
        failures.append(f"run base_model {base_model} != {expected_model}")

    checkpoint = _checkpoint_from_payload(checkpoints_payload, checkpoint_id)
    if checkpoint is None:
        failures.append(f"checkpoint {checkpoint_id} not found")
        checkpoint = {}
    else:
        status = checkpoint.get("status")
        if status != "READY":
            failures.append(f"checkpoint status {status} is not READY")
        checkpoint_run_id = checkpoint.get("rft_run_id")
        if run_id and checkpoint_run_id and checkpoint_run_id != run_id:
            failures.append(f"checkpoint rft_run_id {checkpoint_run_id} != {run_id}")

    report = {
        "artifact": "prime_warm_start_checkpoint_handoff",
        "passed": not failures,
        "failures": failures,
        "checkpoint_id": checkpoint_id,
        "expected_model": expected_model,
        "run_id": run_id,
        "run": run,
        "checkpoint": checkpoint,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option("--run-id", default=None, help="Prime Hosted Training run ID.")
@click.option(
    "--run-json",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Offline `prime train get --output json` payload.",
)
@click.option(
    "--checkpoints-json",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Offline `prime train checkpoints --output json` payload.",
)
@click.option("--checkpoint-id", required=True, help="Checkpoint ID to verify.")
@click.option("--expected-model", default=DEFAULT_EXPECTED_MODEL)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Checkpoint handoff report JSON.",
)
@click.option("--no-fail-on-gate", is_flag=True)
def cli(
    run_id: str | None,
    run_json: Path | None,
    checkpoints_json: Path | None,
    checkpoint_id: str,
    expected_model: str,
    output_path: Path,
    no_fail_on_gate: bool,
) -> None:
    """Verify a READY Prime checkpoint before using it for SFT-to-RL."""
    if run_json or checkpoints_json:
        if not (run_json and checkpoints_json):
            raise click.ClickException(
                "--run-json and --checkpoints-json must be passed together"
            )
        run_payload = _load_json(run_json)
        checkpoints_payload = _load_json(checkpoints_json)
    else:
        if not run_id:
            raise click.ClickException("pass --run-id or offline JSON payloads")
        run_payload = _run_prime(["train", "get", run_id, "--output", "json"])
        checkpoints_payload = _run_prime(
            ["train", "checkpoints", run_id, "--output", "json"]
        )

    report = build_checkpoint_handoff(
        run_payload=run_payload,
        checkpoints_payload=checkpoints_payload,
        checkpoint_id=checkpoint_id,
        expected_model=expected_model,
        output_path=output_path,
    )
    click.echo(
        "checkpoint_handoff={gate} checkpoint={checkpoint} report={report}".format(
            gate="pass" if report["passed"] else "fail",
            checkpoint=checkpoint_id,
            report=output_path,
        )
    )
    if not report["passed"] and not no_fail_on_gate:
        raise click.ClickException("Prime warm-start checkpoint verification failed")


if __name__ == "__main__":
    cli()
