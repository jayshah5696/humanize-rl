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
DATASET_ENV0315_CLEAN50_LEGACY = (
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50"
)
DATASET_ENV0315_CLEAN50_PRIMECOMPAT = (
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat"
)
DATASET_ENV0315_CLEAN50 = DATASET_ENV0315_CLEAN50_PRIMECOMPAT
DATASET_VIEWER_BASE_URL = "https://datasets-server.huggingface.co"
EXPECTED_SPLIT_ROWS_BY_DATASET = {
    DATASET_ENV0314: {"train": 4313, "validation": 239, "test": 241},
    DATASET_ENV0315_CLEAN50_LEGACY: {"train": 4358, "validation": 242, "test": 243},
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
    model_lora = model.get("lora", {})
    renderer = config.get("renderer", {})
    data = config.get("data", {})
    data_loss_mask = data.get("loss_mask", {})
    val_data = config.get("val", {}).get("data", {})
    val_loss_mask = val_data.get("loss_mask", {})
    optim = config.get("optim", {})
    ckpt = config.get("ckpt", {})
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
    if config.get("max_steps") != 200:
        failures.append("max_steps must be 200")
    if config.get("loss_impl") != "torch":
        failures.append("loss_impl must be torch")
    if model.get("seq_len") != 4096:
        failures.append("model.seq_len must be 4096")
    if model.get("optimization_dtype") != "bfloat16":
        failures.append("model.optimization_dtype must be bfloat16")
    if model_lora.get("rank") != 32:
        failures.append("model.lora.rank must be 32")
    if model_lora.get("alpha") != 64:
        failures.append("model.lora.alpha must be 64")
    if data.get("seq_len") != 4096:
        failures.append("data.seq_len must be 4096")
    if data.get("batch_size") != 128:
        failures.append("data.batch_size must be 128")
    if data.get("micro_batch_size") != 1:
        failures.append("data.micro_batch_size must be 1")
    if data.get("seed") != 5696:
        failures.append("data.seed must be 5696")
    if data_loss_mask != {
        "assistant": True,
        "user": False,
        "system": False,
        "tool": False,
    }:
        failures.append("data.loss_mask must train assistant tokens only")
    if val_data.get("seq_len") != 4096:
        failures.append("val.data.seq_len must be 4096")
    if val_data.get("batch_size") != 64:
        failures.append("val.data.batch_size must be 64")
    if val_data.get("micro_batch_size") != 1:
        failures.append("val.data.micro_batch_size must be 1")
    if val_loss_mask != {
        "assistant": True,
        "user": False,
        "system": False,
        "tool": False,
    }:
        failures.append("val.data.loss_mask must evaluate assistant tokens only")
    if optim.get("lr") != 2e-5:
        failures.append("optim.lr must be 2e-5")
    if ckpt.get("interval") != 50:
        failures.append("ckpt.interval must be 50")
    if not ckpt.get("weights_only"):
        failures.append("ckpt.weights_only must be true")
    if not ckpt_weights.get("save_sharded"):
        failures.append("ckpt.weights.save_sharded must be true")
    if ckpt_weights.get("save_format") != "safetensors":
        failures.append("ckpt.weights.save_format must be safetensors")

    if failures:
        return Check("dataset SFT config", False, "; ".join(failures))

    return Check(
        "dataset SFT config",
        True,
        (
            f"{model['name']} on {dataset_name}; "
            f"max_steps={config.get('max_steps')} "
            f"train_batch={data.get('batch_size')} "
            f"val_batch={val_data.get('batch_size')} "
            f"lr={optim.get('lr')} "
            f"lora_rank={model_lora.get('rank')}"
        ),
    )


def dataset_name_from_config(path: Path) -> str:
    try:
        config = _load_toml(path)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return DATASET_ENV0314
    return str(config.get("data", {}).get("name") or DATASET_ENV0314)


def model_name_from_config(path: Path) -> str:
    try:
        config = _load_toml(path)
    except (FileNotFoundError, tomllib.TOMLDecodeError):
        return "Qwen/Qwen3.5-2B"
    return str(config.get("model", {}).get("name") or "Qwen/Qwen3.5-2B")


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


def check_prime_cli_version() -> Check:
    try:
        result = _run_prime(["--version"])
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return Check("Prime CLI version", False, f"prime CLI unavailable: {exc}")

    detail = (result.stdout or result.stderr).strip()
    if result.returncode != 0:
        return Check("Prime CLI version", False, detail or "prime --version failed")
    if not detail:
        return Check("Prime CLI version", False, "prime --version returned no output")
    return Check("Prime CLI version", True, detail)


def get_prime_secret_names() -> set[str]:
    result = _run_prime(["secret", "list", "--output", "json"])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "prime secret list failed")
    payload = json.loads(result.stdout)
    return {secret["name"] for secret in payload.get("secrets", [])}


def get_prime_train_models() -> list[dict[str, Any]]:
    result = _run_prime(["train", "models", "--output", "json"])
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or "prime train models failed")
    payload = json.loads(result.stdout)
    models = payload.get("models", [])
    if not isinstance(models, list):
        raise RuntimeError("prime train models returned no models list")
    return [model for model in models if isinstance(model, dict)]


def check_prime_train_model_availability(
    model_name: str, models: list[dict[str, Any]]
) -> Check:
    for model in models:
        if model.get("name") != model_name:
            continue
        if bool(model.get("at_capacity")):
            return Check(
                "Prime train model availability",
                False,
                f"{model_name} is listed but at capacity",
            )
        price = model.get(
            "effective_training_price_per_mtok",
            model.get("training_price_per_mtok"),
        )
        return Check(
            "Prime train model availability",
            True,
            f"{model_name} available; training_price_per_mtok={price}",
        )
    return Check(
        "Prime train model availability",
        False,
        f"{model_name} not listed by prime train models",
    )


def check_wandb_source(
    env: dict[str, str],
    prime_secret_names: set[str] | None,
    *,
    allow_prime_secret: bool = False,
) -> Check:
    if env.get("WANDB_API_KEY"):
        return Check("WANDB_API_KEY source", True, "local env")
    if prime_secret_names is not None and "WANDB_API_KEY" in prime_secret_names:
        if not allow_prime_secret:
            return Check(
                "WANDB_API_KEY source",
                False,
                "Prime secret exists, but sandbox launch requires local WANDB_API_KEY",
            )
        return Check("WANDB_API_KEY source", True, "Prime secret")
    return Check(
        "WANDB_API_KEY source",
        False,
        "set local WANDB_API_KEY before sandbox SFT launch",
    )


def fetch_dataset_viewer_json(endpoint: str, params: dict[str, str]) -> dict[str, Any]:
    query = urllib.parse.urlencode(params)
    url = f"{DATASET_VIEWER_BASE_URL}/{endpoint}?{query}"
    request = urllib.request.Request(url, headers={"User-Agent": "humanize-rl-preflight"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def _feature_types(value: Any) -> list[str]:
    if isinstance(value, dict):
        types = [str(value["_type"])] if "_type" in value else []
        for child in value.values():
            types.extend(_feature_types(child))
        return types
    if isinstance(value, list):
        types: list[str] = []
        for child in value:
            types.extend(_feature_types(child))
        return types
    return []


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
        unsupported_features = [
            feature["name"]
            for feature in first_rows.get("features", [])
            if "Json" in _feature_types(feature.get("type", {}))
        ]
        if unsupported_features:
            return Check(
                "HF Dataset Viewer",
                False,
                "unsupported Prime feature type Json in "
                + ", ".join(sorted(unsupported_features)),
            )

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


def build_launch_policy(*, allow_prime_wandb_secret: bool) -> dict[str, Any]:
    return {
        "runner": "prime_sandbox",
        "wandb_source": {
            "local_env_required": not allow_prime_wandb_secret,
            "prime_secret_allowed": allow_prime_wandb_secret,
        },
    }


def build_preflight_report(
    config_path: Path,
    checks: list[Check],
    *,
    allow_prime_wandb_secret: bool = False,
) -> dict[str, Any]:
    failed = [check.name for check in checks if not check.ok]
    return {
        "artifact": "prime_sft_preflight",
        "config": str(config_path),
        "launch_policy": build_launch_policy(
            allow_prime_wandb_secret=allow_prime_wandb_secret
        ),
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
    output_path: Path,
    config_path: Path,
    checks: list[Check],
    *,
    allow_prime_wandb_secret: bool = False,
) -> dict[str, Any]:
    report = build_preflight_report(
        config_path,
        checks,
        allow_prime_wandb_secret=allow_prime_wandb_secret,
    )
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
@click.option(
    "--allow-prime-wandb-secret",
    is_flag=True,
    help=(
        "Allow WANDB_API_KEY from Prime global secrets. Do not use for sandbox "
        "launch kits unless the secret is explicitly injected into WANDB_API_KEY."
    ),
)
def cli(
    config_path: Path,
    skip_prime: bool,
    check_hf_viewer: bool,
    output_path: Path | None,
    no_fail_on_gate: bool,
    allow_prime_wandb_secret: bool,
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
                "set local WANDB_API_KEY before sandbox SFT launch",
            )
        )
    else:
        checks.append(check_prime_cli_version())
        checks.append(check_prime_auth())
        prime_secret_names: set[str] | None = None
        try:
            prime_secret_names = get_prime_secret_names()
        except (RuntimeError, json.JSONDecodeError) as exc:
            checks.append(Check("Prime secret list", False, str(exc)))
        try:
            prime_train_models = get_prime_train_models()
        except (RuntimeError, json.JSONDecodeError) as exc:
            checks.append(Check("Prime train model availability", False, str(exc)))
        else:
            checks.append(
                check_prime_train_model_availability(
                    model_name_from_config(config_path),
                    prime_train_models,
                )
            )
        checks.append(
            check_wandb_source(
                env,
                prime_secret_names,
                allow_prime_secret=allow_prime_wandb_secret,
            )
        )

    click.echo(render_checks(checks))
    if output_path is not None:
        write_preflight_report(
            output_path,
            config_path,
            checks,
            allow_prime_wandb_secret=allow_prime_wandb_secret,
        )
        click.echo(f"report={output_path}")

    if not all(check.ok for check in checks) and not no_fail_on_gate:
        raise click.ClickException("Prime SFT preflight failed")


if __name__ == "__main__":
    cli()
