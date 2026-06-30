import json
from pathlib import Path

from click.testing import CliRunner

from scripts.train.verify_prime_sft_output import build_sft_output_report, cli


def _write_config(
    path: Path,
    output_dir: Path,
    *,
    dataset: str = "jayshah5696/humanize-rl-prime-sft-messages-env0314",
) -> None:
    path.write_text(
        f"""
output_dir = "{output_dir.as_posix()}"

[model]
name = "Qwen/Qwen3.5-2B"

[renderer]
name = "qwen3.5"

[data]
name = "{dataset}"

[ckpt.weights]
save_adapter_separately = true
""".strip()
        + "\n"
    )


def _write_step(output_dir: Path, step: int = 200) -> Path:
    step_dir = output_dir / "weights" / f"step_{step}"
    step_dir.mkdir(parents=True)
    (step_dir / "model.safetensors").write_text("weights")
    (step_dir / "config.json").write_text("{}\n")
    adapter_dir = output_dir / "adapters" / f"step_{step}"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_model.safetensors").write_text("adapter")
    (adapter_dir / "adapter_config.json").write_text("{}\n")
    return step_dir


def test_build_sft_output_report_passes_complete_step(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "sft"
    config = tmp_path / "config.toml"
    report_path = tmp_path / "report.json"
    _write_config(config, output_dir)
    _write_step(output_dir)

    report = build_sft_output_report(
        config_path=config,
        output_dir=None,
        step=200,
        output_path=report_path,
        require_adapter=True,
    )

    assert report["passed"] is True
    assert report["step"] == 200
    assert report["step_dir"].endswith("weights/step_200")
    assert report["config"]["model"] == "Qwen/Qwen3.5-2B"
    assert report["safetensors_count"] == 1
    assert report["adapter_artifact_count"] == 2
    assert report_path.exists()


def test_build_sft_output_report_accepts_clean50_s2_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "sft"
    config = tmp_path / "config.toml"
    report_path = tmp_path / "report.json"
    _write_config(
        config,
        output_dir,
        dataset="jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50",
    )
    _write_step(output_dir)

    report = build_sft_output_report(
        config_path=config,
        output_dir=None,
        step=200,
        output_path=report_path,
        require_adapter=True,
    )

    assert report["passed"] is True
    assert (
        report["config"]["data"]
        == "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50"
    )


def test_build_sft_output_report_rejects_unapproved_dataset(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "sft"
    config = tmp_path / "config.toml"
    report_path = tmp_path / "report.json"
    _write_config(config, output_dir, dataset="jayshah5696/untracked-sft-dataset")
    _write_step(output_dir)

    report = build_sft_output_report(
        config_path=config,
        output_dir=None,
        step=200,
        output_path=report_path,
        require_adapter=True,
    )

    assert report["passed"] is False
    assert "data jayshah5696/untracked-sft-dataset not in approved SFT datasets" in report[
        "failures"
    ]


def test_build_sft_output_report_fails_missing_adapter_when_required(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "outputs" / "sft"
    config = tmp_path / "config.toml"
    report_path = tmp_path / "report.json"
    _write_config(config, output_dir)
    step_dir = output_dir / "weights" / "step_200"
    step_dir.mkdir(parents=True)
    (step_dir / "model.safetensors").write_text("weights")

    report = build_sft_output_report(
        config_path=config,
        output_dir=None,
        step=200,
        output_path=report_path,
        require_adapter=True,
    )

    assert report["passed"] is False
    assert "adapter artifacts missing while save_adapter_separately=true" in report[
        "failures"
    ]


def test_cli_uses_latest_step_when_step_omitted(tmp_path: Path) -> None:
    output_dir = tmp_path / "outputs" / "sft"
    config = tmp_path / "config.toml"
    report_path = tmp_path / "report.json"
    _write_config(config, output_dir)
    _write_step(output_dir, step=50)
    _write_step(output_dir, step=200)

    result = CliRunner().invoke(
        cli,
        [
            "--config",
            str(config),
            "--output",
            str(report_path),
        ],
    )

    assert result.exit_code == 0
    assert "sft_output=pass step=200" in result.output
    report = json.loads(report_path.read_text())
    assert report["step"] == 200
