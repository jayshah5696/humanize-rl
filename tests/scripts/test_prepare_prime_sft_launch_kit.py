import json
import tarfile
from pathlib import Path

from click.testing import CliRunner

from scripts.train.prepare_prime_sft_launch_kit import (
    build_manifest,
    cli,
    write_launch_kit,
)


def _write_config(path: Path) -> None:
    _write_config_for_dataset(
        path,
        dataset="jayshah5696/humanize-rl-prime-sft-messages-env0314",
    )


def _write_config_for_dataset(path: Path, *, dataset: str) -> None:
    path.write_text(
        f"""
max_steps = 200
output_dir = "outputs/prime_sft/test"
clean_output_dir = false
loss_impl = "torch"

[model]
name = "Qwen/Qwen3.5-2B"

[renderer]
name = "qwen3.5"

[data]
name = "{dataset}"
splits = ["train"]

[val]
interval = 25
eval_on_start = true

[val.data]
name = "{dataset}"
splits = ["validation"]

[ckpt]
interval = 50
weights_only = true

[ckpt.weights]
save_adapter_separately = true

[wandb]
project = "humanize-rl"
name = "test-sft"
tags = ["prime-rl", "sft"]
""".strip()
        + "\n"
    )


def test_build_manifest_keeps_sft_launch_facts(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_config(config)

    manifest = build_manifest(
        config_path=config,
        prime_rl_ref="abc123",
        git_commit="deadbeef",
        git_dirty=True,
    )

    assert manifest["config"]["model"] == "Qwen/Qwen3.5-2B"
    assert manifest["config"]["renderer"] == "qwen3.5"
    assert manifest["config"]["data"] == "jayshah5696/humanize-rl-prime-sft-messages-env0314"
    assert manifest["expected_dataset"]["splits"]["train"] == 4313
    assert manifest["config"]["train_splits"] == ["train"]
    assert manifest["config"]["validation_splits"] == ["validation"]
    assert manifest["prime_rl_ref"] == "abc123"
    assert manifest["git"]["dirty"] is True


def test_build_manifest_uses_clean50_expected_split_counts(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    _write_config_for_dataset(
        config,
        dataset="jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50",
    )

    manifest = build_manifest(
        config_path=config,
        prime_rl_ref="abc123",
        git_commit="deadbeef",
        git_dirty=False,
    )

    assert (
        manifest["expected_dataset"]["name"]
        == "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50"
    )
    assert manifest["expected_dataset"]["splits"] == {
        "train": 4358,
        "validation": 242,
        "test": 243,
    }


def test_write_launch_kit_creates_secret_free_run_files(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    output_dir = tmp_path / "kit"
    _write_config(config)

    archive_path = write_launch_kit(
        config_path=config,
        output_dir=output_dir,
        prime_rl_ref="abc123",
        git_commit="deadbeef",
        git_dirty=False,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    readme = (output_dir / "README.md").read_text()
    runner = (output_dir / "run_sft.sh").read_text()

    assert manifest["artifact"] == "prime_sft_launch_kit"
    assert "prime train" in readme
    assert "uv run sft @" in runner
    assert "WANDB_API_KEY" in runner
    assert "wandb_test" not in readme + runner + json.dumps(manifest)
    assert archive_path.exists()
    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "prime_sft_launch_kit/config.toml" in names
    assert "prime_sft_launch_kit/run_sft.sh" in names
    assert "prime_sft_launch_kit/manifest.json" in names


def test_write_launch_kit_can_package_sft_eval_manifest(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    output_dir = tmp_path / "kit"
    eval_manifest = tmp_path / "sft_eval_manifest.json"
    _write_config_for_dataset(
        config,
        dataset="jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50",
    )
    eval_manifest.write_text(
        json.dumps(
            {
                "artifact": "sft_eval_manifest",
                "checkpoint_id": "READY_SFT_CHECKPOINT_ID",
                "checkpoint_slug": None,
                "promotion_root": "runs/prime_sft_promotion/TEMPLATE",
                "gate_order": ["verify_sft_output", "build_promotion_gate"],
            }
        )
        + "\n"
    )

    archive_path = write_launch_kit(
        config_path=config,
        output_dir=output_dir,
        prime_rl_ref="abc123",
        git_commit="deadbeef",
        git_dirty=False,
        eval_manifest_path=eval_manifest,
    )

    manifest = json.loads((output_dir / "manifest.json").read_text())
    packaged_eval_manifest = output_dir / "sft_eval_manifest.json"

    assert packaged_eval_manifest.read_text() == eval_manifest.read_text()
    assert manifest["sft_eval_manifest"]["source_path"] == str(eval_manifest)
    assert manifest["sft_eval_manifest"]["kit_path"] == "prime_sft_launch_kit/sft_eval_manifest.json"
    assert manifest["sft_eval_manifest"]["checkpoint_id"] == "READY_SFT_CHECKPOINT_ID"
    assert manifest["sft_eval_manifest"]["checkpoint_slug"] is None
    assert manifest["sft_eval_manifest"]["promotion_root"] == "runs/prime_sft_promotion/TEMPLATE"
    assert manifest["sft_eval_manifest"]["gate_order"] == [
        "verify_sft_output",
        "build_promotion_gate",
    ]
    with tarfile.open(archive_path, "r:gz") as tar:
        names = set(tar.getnames())
    assert "prime_sft_launch_kit/sft_eval_manifest.json" in names


def test_cli_writes_launch_kit(tmp_path: Path) -> None:
    config = tmp_path / "config.toml"
    output_dir = tmp_path / "kit"
    _write_config(config)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config),
            "--output-dir",
            str(output_dir),
            "--prime-rl-ref",
            "abc123",
        ],
    )

    assert result.exit_code == 0
    assert "launch_kit=" in result.output
    assert (output_dir / "config.toml").exists()
