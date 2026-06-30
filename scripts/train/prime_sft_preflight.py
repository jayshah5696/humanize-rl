# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
import os
import subprocess
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import click

DEFAULT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml"
)
DATASET_ENV0314 = "jayshah5696/humanize-rl-prime-sft-messages-env0314"
DATASET_ENV0315_CLEAN50 = (
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50"
)
DATASET_VIEWER_BASE_URL = "https://datasets-server.huggingface.co"
EXPECTED_SPLIT_ROWS_BY_DATASET = {
    DATASET_ENV0314: {"train": 4313, "validation": 239, "test": 241},
    DATASET_ENV0315_CLEAN50: {"train": 4358, "validation": 242, "test": 243},
}


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def check_sft_config(path: Path) -> Check:
    if not path.exists():
        return Check("dataset SFT config", False, f"missing {path}")

    try:
        config = _load_toml(path)
    except tomllib.TOMLDecodeError as exc:
        return Check("dataset SFT config", False, f"TOML parse error: {exc}")

    model = config.get("model", {})
    renderer = config.get("renderer", {})
    data = config.get("data", {})
    val_data = config.get("val", {}).get("data", {})
    ckpt_weights = config.get("ckpt", {}).get("weights", {})
    dataset_name = str(data.get("name") or "")

    failures = []
    if model.get("name") != "Qwen/Qwen3.5-2B":
        failures.append("model.name must be Qwen/Qwen3.5-2B")
    if renderer.get("name") != "qwen3.5":
        failures.append("renderer.name must be qwen3.5")
    if dataset_name not in EXPECTED_SPLIT_ROWS_BY_DATASET:
        allowed = ", ".join(sorted(EXPECTED_SPLIT_ROWS_BY_DATASET))
        failures.append(f"data.name must be one of: {allowed}")
    if data.get("splits") != ["train"]:
        failures.append("data.splits must be ['train']")
    if val_data.get("name") != dataset_name:
        failures.append("val.data.name must match data.name")
    if val_data.get("splits") != ["validation"]:
        failures.append("val.data.splits must be ['validation']")
    if not ckpt_weights.get("save_adapter_separately"):
        failures.append("ckpt.weights.save_adapter_separately must be true")
    if "wandb" not in config:
        failures.append("wandb table is required for tracked SFT")

    if failures:
        return Check("dataset SFT config", False, "; ".join(failures))

    return Check(
        "dataset SFT config",
        True,
        f"{model['name']} on {dataset_name}",
    )


def dataset_name_from_config(path: Path) -> str:
    try:
        config = _load_toml(path)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return DATASET_ENV0314
    return str(config.get("data", {}).get("name") or DATASET_ENV0314)


def check_local_secret(env: dict[str, str], name: str, detail: str) -> Check:
    if env.get(name):
        return Check(f"local {name}", True, "set")
    return Check(f"local {name}", False, detail)


def _run_prime(args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["prime", "--plain", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )


def check_prime_auth() -> Check:
    try:
        result = _run_prime(["whoami"])
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return Check("Prime auth", False, f"prime CLI unavailable: {exc}")

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return Check("Prime auth", False, detail or "prime whoami failed")

    return Check("Prime auth", True, "whoami succeeded")


def get_prime_secret_names() -> set[str]:
    result = _run_prime(["secret", "list", "--output", "json"])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "prime secret list failed")
    payload = json.loads(result.stdout)
    return {secret["name"] for secret in payload.get("secrets", [])}


def check_wandb_source(
    env: dict[str, str], prime_secret_names: set[str] | None
) -> Check:
    if env.get("WANDB_API_KEY"):
        return Check("WANDB_API_KEY source", True, "local env")
    if prime_secret_names is not None and "WANDB_API_KEY" in prime_secret_names:
        return Check("WANDB_API_KEY source", True, "Prime secret")
    return Check(
        "WANDB_API_KEY source",
        False,
        "set WANDB_API_KEY or create a Prime secret before tracked SFT",
    )


def fetch_dataset_viewer_json(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_VIEWER_BASE_URL}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "humanize-rl-preflight"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def check_hf_dataset_viewer(
    dataset_name: str = DATASET_ENV0314,
    fetch_json=fetch_dataset_viewer_json,
) -> Check:
    expected_split_rows = EXPECTED_SPLIT_ROWS_BY_DATASET.get(dataset_name)
    if expected_split_rows is None:
        return Check("HF Dataset Viewer", False, f"unsupported dataset {dataset_name}")
    try:
        valid = fetch_json("is-valid", {"dataset": dataset_name})
        if not valid.get("viewer"):
            return Check("HF Dataset Viewer", False, "viewer is not available")

        size = fetch_json("size", {"dataset": dataset_name})
        split_rows = {
            split["split"]: split["num_rows"]
            for split in size.get("size", {}).get("splits", [])
        }
        for split, expected_rows in expected_split_rows.items():
            if split_rows.get(split) != expected_rows:
                return Check(
                    "HF Dataset Viewer",
                    False,
                    f"{split} rows {split_rows.get(split)} != {expected_rows}",
                )

        first_rows = fetch_json(
            "first-rows", {"dataset": dataset_name, "config": "default", "split": "train"}
        )
        feature_names = {feature["name"] for feature in first_rows.get("features", [])}
        if "messages" not in feature_names:
            return Check("HF Dataset Viewer", False, "messages column missing")

    except (OSError, urllib.error.URLError, json.JSONDecodeError, KeyError) as exc:
        return Check("HF Dataset Viewer", False, str(exc))

    detail = " ".join(f"{split}={rows}" for split, rows in expected_split_rows.items())
    return Check("HF Dataset Viewer", True, detail)


def render_checks(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "pass" if check.ok else "fail"
        lines.append(f"{check.name}: {status} - {check.detail}")
    return "\n".join(lines)


def build_preflight_report(config_path: Path, checks: list[Check]) -> dict[str, Any]:
    failed = [check.name for check in checks if not check.ok]
    return {
        "artifact": "prime_sft_preflight",
        "config": str(config_path),
        "checks": [
            {
                "name": check.name,
                "passed": check.ok,
                "detail": check.detail,
            }
            for check in checks
        ],
        "gate": {
            "passed": not failed,
            "failed_checks": failed,
        },
    }


def write_preflight_report(
    output_path: Path, config_path: Path, checks: list[Check]
) -> dict[str, Any]:
    report = build_preflight_report(config_path, checks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Prime prime-rl SFT config to validate.",
)
@click.option(
    "--skip-prime",
    is_flag=True,
    help="Skip Prime CLI auth and secret-store checks.",
)
@click.option(
    "--check-hf-viewer",
    is_flag=True,
    help="Check current Hugging Face Dataset Viewer split counts and messages schema.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=None,
    help="Optional JSON preflight report output.",
)
@click.option(
    "--no-fail-on-gate",
    is_flag=True,
    help="Write/render the report but return success even when a check fails.",
)
def cli(
    config_path: Path,
    skip_prime: bool,
    check_hf_viewer: bool,
    output_path: Path | None,
    no_fail_on_gate: bool,
) -> None:
    """Preflight the Prime Qwen 2B dataset SFT launch."""

    env = dict(os.environ)
    checks = [
        check_sft_config(config_path),
        check_local_secret(env, "HF_TOKEN", "set HF_TOKEN before uploading/running SFT"),
    ]
    if check_hf_viewer:
        checks.append(check_hf_dataset_viewer(dataset_name=dataset_name_from_config(config_path)))

    if skip_prime:
        checks.append(
            check_local_secret(
                env,
                "WANDB_API_KEY",
                "set WANDB_API_KEY or create a Prime secret before tracked SFT",
            )
        )
    else:
        checks.append(check_prime_auth())
        prime_secret_names: set[str] | None = None
        try:
            prime_secret_names = get_prime_secret_names()
        except (RuntimeError, json.JSONDecodeError) as exc:
            checks.append(Check("Prime secret list", False, str(exc)))
        checks.append(check_wandb_source(env, prime_secret_names))

    click.echo(render_checks(checks))
    if output_path is not None:
        write_preflight_report(output_path, config_path, checks)
        click.echo(f"report={output_path}")

    if not all(check.ok for check in checks) and not no_fail_on_gate:
        raise click.ClickException("Prime SFT preflight failed")


if __name__ == "__main__":
    cli()
