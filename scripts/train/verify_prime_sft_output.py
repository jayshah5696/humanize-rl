# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path
from typing import Any

import click

DEFAULT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml"
)
EXPECTED_MODEL = "Qwen/Qwen3.5-2B"
EXPECTED_RENDERER = "qwen3.5"
APPROVED_DATASETS = frozenset(
    {
        "jayshah5696/humanize-rl-prime-sft-messages-env0314",
        "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50",
    }
)
STEP_DIR_RE = re.compile(r"^step_(\d+)$")


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _manifest_gate(
    path: Path | None,
    *,
    config_path: Path,
    output_path: Path,
) -> tuple[dict[str, Any] | None, list[str]]:
    if path is None:
        return None, []

    report = _load_json(path)
    failures: list[str] = []
    if report.get("artifact") != "sft_eval_manifest":
        failures.append("SFT eval manifest artifact is invalid")

    manifest_config = report.get("config")
    if not isinstance(manifest_config, dict):
        failures.append("SFT eval manifest missing config object")
    elif manifest_config.get("path") != str(config_path):
        failures.append("SFT eval manifest config path mismatch")

    required = report.get("required_artifacts")
    if not isinstance(required, dict):
        failures.append("SFT eval manifest missing required_artifacts object")
    elif required.get("sft_output_verification") != str(output_path):
        failures.append("SFT eval manifest sft_output_verification path mismatch")

    return report, failures


def _step_number(path: Path) -> int | None:
    match = STEP_DIR_RE.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def _find_step_dir(output_dir: Path, step: int | None) -> tuple[int | None, Path | None]:
    weights_dir = output_dir / "weights"
    if step is not None:
        step_dir = weights_dir / f"step_{step}"
        return step, step_dir if step_dir.is_dir() else None

    if not weights_dir.is_dir():
        return None, None

    candidates = [
        (number, path)
        for path in weights_dir.iterdir()
        if path.is_dir() and (number := _step_number(path)) is not None
    ]
    if not candidates:
        return None, None
    return max(candidates, key=lambda item: item[0])


def _relative_files(root: Path, files: list[Path]) -> list[str]:
    return [path.relative_to(root).as_posix() for path in sorted(files)]


def _adapter_artifacts(output_dir: Path, step: int | None) -> list[Path]:
    candidates: list[Path] = []
    roots = [output_dir / "adapters"]
    if step is not None:
        roots.extend(
            [
                output_dir / "adapters" / f"step_{step}",
                output_dir / "weights" / f"step_{step}",
            ]
        )
    for root in roots:
        if root.is_dir():
            candidates.extend(
                path
                for path in root.rglob("*")
                if path.is_file() and "adapter" in path.name.lower()
            )
    return sorted(set(candidates))


def build_sft_output_report(
    *,
    config_path: Path,
    output_dir: Path | None,
    step: int | None,
    output_path: Path,
    require_adapter: bool,
    sft_eval_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify a Prime prime-rl SFT output has a usable checkpoint snapshot."""
    failures: list[str] = []
    manifest, manifest_failures = _manifest_gate(
        sft_eval_manifest_path,
        config_path=config_path,
        output_path=output_path,
    )
    failures.extend(manifest_failures)

    config = _load_toml(config_path)
    config_output_dir = Path(str(config.get("output_dir") or ""))
    resolved_output_dir = output_dir or config_output_dir
    model = config.get("model", {}).get("name")
    renderer = config.get("renderer", {}).get("name")
    data = config.get("data", {}).get("name")
    save_adapter_separately = bool(
        config.get("ckpt", {}).get("weights", {}).get("save_adapter_separately")
    )

    if model != EXPECTED_MODEL:
        failures.append(f"model {model} != {EXPECTED_MODEL}")
    if renderer != EXPECTED_RENDERER:
        failures.append(f"renderer {renderer} != {EXPECTED_RENDERER}")
    if data not in APPROVED_DATASETS:
        failures.append(f"data {data} not in approved SFT datasets")
    if not resolved_output_dir:
        failures.append("output_dir missing")
    elif not resolved_output_dir.exists():
        failures.append(f"output_dir missing: {resolved_output_dir}")

    found_step, step_dir = _find_step_dir(resolved_output_dir, step)
    if step_dir is None:
        expected = f"step_{step}" if step is not None else "latest step_*"
        failures.append(f"weights snapshot missing: {resolved_output_dir}/weights/{expected}")
        safetensors_files: list[Path] = []
    else:
        safetensors_files = list(step_dir.rglob("*.safetensors"))
        if not safetensors_files:
            failures.append(f"no safetensors files under {step_dir}")

    adapter_files = _adapter_artifacts(resolved_output_dir, found_step)
    if require_adapter and save_adapter_separately and not adapter_files:
        failures.append("adapter artifacts missing while save_adapter_separately=true")

    report = {
        "artifact": "prime_sft_output_verification",
        "passed": not failures,
        "failures": failures,
        "config_path": str(config_path),
        "output_dir": str(resolved_output_dir),
        "step": found_step,
        "step_dir": str(step_dir) if step_dir is not None else None,
        "config": {
            "model": model,
            "renderer": renderer,
            "data": data,
            "save_adapter_separately": save_adapter_separately,
        },
        "safetensors_count": len(safetensors_files),
        "safetensors_files": _relative_files(resolved_output_dir, safetensors_files),
        "adapter_artifact_count": len(adapter_files),
        "adapter_artifacts": _relative_files(resolved_output_dir, adapter_files),
        "sft_eval_manifest": {
            "path": str(sft_eval_manifest_path)
            if sft_eval_manifest_path is not None
            else None,
            "checkpoint_id": manifest.get("checkpoint_id") if manifest else None,
            "promotion_root": manifest.get("promotion_root") if manifest else None,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_CONFIG,
    help="Prime prime-rl SFT config.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=None,
    help="Override output_dir from the config.",
)
@click.option("--step", type=int, default=None, help="Specific SFT step to verify.")
@click.option(
    "--sft-eval-manifest",
    "sft_eval_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="SFT eval manifest that defines the expected verification report path.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Verification report JSON.",
)
@click.option(
    "--require-adapter/--no-require-adapter",
    default=True,
    help="Require adapter artifacts when the config saves adapters separately.",
)
@click.option("--no-fail-on-gate", is_flag=True)
def cli(
    config_path: Path,
    output_dir: Path | None,
    step: int | None,
    sft_eval_manifest_path: Path | None,
    output_path: Path,
    require_adapter: bool,
    no_fail_on_gate: bool,
) -> None:
    """Verify Prime prime-rl SFT output before eval or checkpoint handoff."""
    report = build_sft_output_report(
        config_path=config_path,
        output_dir=output_dir,
        step=step,
        output_path=output_path,
        require_adapter=require_adapter,
        sft_eval_manifest_path=sft_eval_manifest_path,
    )
    click.echo(
        "sft_output={gate} step={step} safetensors={safetensors} adapters={adapters} report={report}".format(
            gate="pass" if report["passed"] else "fail",
            step=report["step"],
            safetensors=report["safetensors_count"],
            adapters=report["adapter_artifact_count"],
            report=output_path,
        )
    )
    if not report["passed"] and not no_fail_on_gate:
        raise click.ClickException("Prime SFT output verification failed")


if __name__ == "__main__":
    cli()
