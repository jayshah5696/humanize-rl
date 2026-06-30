import json
from pathlib import Path

from click.testing import CliRunner

from scripts.train.prime_sft_preflight import (
    DATASET_ENV0315_CLEAN50,
    check_hf_dataset_viewer,
    check_sft_config,
    cli,
)


def test_lightweight_train_scripts_declare_inline_uv_dependencies():
    script_paths = [
        "scripts/train/prime_sft_preflight.py",
        "scripts/train/prepare_prime_sft_launch_kit.py",
        "scripts/train/prepare_prime_sft_to_rl_config.py",
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
    assert "set WANDB_API_KEY or create a Prime secret before tracked SFT" in result.output


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


def test_s2_prime_sft_config_passes_static_preflight() -> None:
    check = check_sft_config(
        Path("configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml")
    )

    assert check.ok
    assert DATASET_ENV0315_CLEAN50 in check.detail


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
