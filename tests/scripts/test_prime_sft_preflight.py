import json
import subprocess
from pathlib import Path

from click.testing import CliRunner

import scripts.train.prime_sft_preflight as prime_sft_preflight
from scripts.train.prime_sft_preflight import (
    DATASET_ENV0315_CLEAN50,
    check_hf_dataset_viewer,
    check_prime_cli_version,
    check_prime_train_model_availability,
    check_sft_config,
    check_wandb_source,
    cli,
)


def test_lightweight_train_scripts_declare_inline_uv_dependencies():
    script_paths = [
        "scripts/train/prime_sft_preflight.py",
        "scripts/train/prepare_prime_sft_launch_kit.py",
        "scripts/train/prepare_prime_sft_to_rl_config.py",
        "scripts/train/verify_prime_sft_launch_readiness.py",
    ]

    for script_path in script_paths:
        text = Path(script_path).read_text()
        assert "# /// script" in text[:240]
        assert '"click' in text[:240]


def test_prime_sft_preflight_passes_offline_with_required_env(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.setenv("WANDB_API_KEY", "wandb_test")

    result = CliRunner().invoke(cli, ["--skip-prime"])

    assert result.exit_code == 0
    assert "dataset SFT config: pass" in result.output
    assert "local HF_TOKEN: pass" in result.output
    assert "local WANDB_API_KEY: pass" in result.output


def test_prime_sft_preflight_fails_when_wandb_missing(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)

    result = CliRunner().invoke(cli, ["--skip-prime"])

    assert result.exit_code == 1
    assert "local WANDB_API_KEY: fail" in result.output
    assert "set local WANDB_API_KEY before sandbox SFT launch" in result.output


def test_prime_sft_preflight_writes_failed_json_report(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_test")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    output_path = tmp_path / "preflight.json"

    result = CliRunner().invoke(
        cli,
        [
            "--skip-prime",
            "--output",
            str(output_path),
            "--no-fail-on-gate",
        ],
    )

    assert result.exit_code == 0
    report = json.loads(output_path.read_text())
    assert report["artifact"] == "prime_sft_preflight"
    assert report["gate"]["passed"] is False
    assert report["gate"]["failed_checks"] == ["local WANDB_API_KEY"]
    assert report["launch_policy"] == {
        "runner": "prime_sandbox",
        "wandb_source": {
            "local_env_required": True,
            "prime_secret_allowed": False,
        },
    }


def test_check_prime_cli_version_records_installed_version(monkeypatch) -> None:
    def fake_run_prime(args: list[str]) -> subprocess.CompletedProcess[str]:
        assert args == ["--version"]
        return subprocess.CompletedProcess(
            ["prime", "--plain", "--version"],
            0,
            stdout="Prime CLI version: 0.6.14\n",
            stderr="",
        )

    monkeypatch.setattr(prime_sft_preflight, "_run_prime", fake_run_prime)

    check = check_prime_cli_version()

    assert check.ok
    assert check.detail == "Prime CLI version: 0.6.14"


def test_check_wandb_source_rejects_prime_secret_for_sandbox_default() -> None:
    check = check_wandb_source({}, {"WANDB_API_KEY"})

    assert not check.ok
    assert "Prime secret exists" in check.detail
    assert "sandbox launch requires local WANDB_API_KEY" in check.detail


def test_check_wandb_source_can_allow_prime_secret_for_non_sandbox_path() -> None:
    check = check_wandb_source(
        {},
        {"WANDB_API_KEY"},
        allow_prime_secret=True,
    )

    assert check.ok
    assert check.detail == "Prime secret"


def test_check_hf_dataset_viewer_passes_with_expected_messages_dataset():
    def fake_fetch(endpoint, params):
        if endpoint == "is-valid":
            return {"viewer": True}
        if endpoint == "size":
            return {
                "size": {
                    "splits": [
                        {"split": "train", "num_rows": 4313},
                        {"split": "validation", "num_rows": 239},
                        {"split": "test", "num_rows": 241},
                    ]
                }
            }
        if endpoint == "first-rows":
            return {
                "features": [
                    {"name": "id"},
                    {"name": "messages"},
                    {"name": "split"},
                ]
            }
        raise AssertionError(endpoint)

    check = check_hf_dataset_viewer(fetch_json=fake_fetch)

    assert check.ok
    assert "train=4313" in check.detail
    assert "validation=239" in check.detail
    assert "test=241" in check.detail


def test_check_hf_dataset_viewer_passes_for_env0315_clean50_dataset():
    def fake_fetch(endpoint, params):
        assert params["dataset"] == DATASET_ENV0315_CLEAN50
        if endpoint == "is-valid":
            return {"viewer": True}
        if endpoint == "size":
            return {
                "size": {
                    "splits": [
                        {"split": "train", "num_rows": 4358},
                        {"split": "validation", "num_rows": 242},
                        {"split": "test", "num_rows": 243},
                    ]
                }
            }
        if endpoint == "first-rows":
            return {
                "features": [
                    {"name": "id"},
                    {"name": "messages"},
                    {"name": "split"},
                ]
            }
        raise AssertionError(endpoint)

    check = check_hf_dataset_viewer(
        dataset_name=DATASET_ENV0315_CLEAN50,
        fetch_json=fake_fetch,
    )

    assert check.ok
    assert "train=4358" in check.detail
    assert "validation=242" in check.detail
    assert "test=243" in check.detail


def test_check_hf_dataset_viewer_rejects_json_feature_type():
    def fake_fetch(endpoint, params):
        if endpoint == "is-valid":
            return {"viewer": True}
        if endpoint == "size":
            return {
                "size": {
                    "splits": [
                        {"split": "train", "num_rows": 4358},
                        {"split": "validation", "num_rows": 242},
                        {"split": "test", "num_rows": 243},
                    ]
                }
            }
        if endpoint == "first-rows":
            return {
                "features": [
                    {"name": "id", "type": {"dtype": "string", "_type": "Value"}},
                    {"name": "messages", "type": {"_type": "List"}},
                    {"name": "quality", "type": {"_type": "Json"}},
                ]
            }
        raise AssertionError(endpoint)

    check = check_hf_dataset_viewer(
        dataset_name=DATASET_ENV0315_CLEAN50,
        fetch_json=fake_fetch,
    )

    assert not check.ok
    assert "unsupported Prime feature type Json in quality" in check.detail


def test_s2_prime_sft_config_passes_static_preflight() -> None:
    check = check_sft_config(
        Path("configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml")
    )

    assert check.ok
    assert DATASET_ENV0315_CLEAN50 in check.detail
    assert "max_steps=200" in check.detail
    assert "train_batch=128" in check.detail
    assert "val_batch=64" in check.detail
    assert "lr=2e-05" in check.detail


def test_s2_prime_sft_config_rejects_smoke_step_count(tmp_path: Path) -> None:
    config = tmp_path / "s2_smoke.toml"
    source = Path(
        "configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml"
    )
    config.write_text(source.read_text().replace("max_steps = 200", "max_steps = 50"))

    check = check_sft_config(config)

    assert not check.ok
    assert "max_steps must be 200" in check.detail


def test_check_prime_train_model_availability_passes_when_model_is_available():
    check = check_prime_train_model_availability(
        "Qwen/Qwen3.5-2B",
        [
            {
                "name": "Qwen/Qwen3.5-2B",
                "at_capacity": False,
                "effective_training_price_per_mtok": 0.15,
            }
        ],
    )

    assert check.ok
    assert "Qwen/Qwen3.5-2B available" in check.detail
    assert "training_price_per_mtok=0.15" in check.detail


def test_check_prime_train_model_availability_fails_when_model_is_at_capacity():
    check = check_prime_train_model_availability(
        "Qwen/Qwen3.5-2B",
        [{"name": "Qwen/Qwen3.5-2B", "at_capacity": True}],
    )

    assert not check.ok
    assert "at capacity" in check.detail


def test_check_prime_train_model_availability_fails_when_model_is_missing():
    check = check_prime_train_model_availability(
        "Qwen/Qwen3.5-2B",
        [{"name": "Qwen/Qwen3.5-4B", "at_capacity": False}],
    )

    assert not check.ok
    assert "not listed" in check.detail


def test_check_hf_dataset_viewer_fails_without_messages_column():
    def fake_fetch(endpoint, params):
        if endpoint == "is-valid":
            return {"viewer": True}
        if endpoint == "size":
            return {
                "size": {
                    "splits": [
                        {"split": "train", "num_rows": 4313},
                        {"split": "validation", "num_rows": 239},
                        {"split": "test", "num_rows": 241},
                    ]
                }
            }
        if endpoint == "first-rows":
            return {"features": [{"name": "prompt"}, {"name": "completion"}]}
        raise AssertionError(endpoint)

    check = check_hf_dataset_viewer(fetch_json=fake_fetch)

    assert not check.ok
    assert "messages column missing" in check.detail
