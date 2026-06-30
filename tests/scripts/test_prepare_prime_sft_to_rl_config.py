import json
import tomllib
from pathlib import Path

from click.testing import CliRunner

from scripts.train.prepare_prime_sft_to_rl_config import (
    PLACEHOLDER_CHECKPOINT,
    cli,
    prepare_after_sft_config,
)

TEMPLATE = """
name = "humanize-p5050-qwen35-2b-after-sft-full200-env0315-r1"
model = "Qwen/Qwen3.5-2B"
checkpoint_id = "FILL_WITH_READY_SFT_CHECKPOINT_ID"
max_steps = 200

[[env]]
id = "jayshah5696/humanize-rl-env"
name = "train_mix_v2_p5050_qwen35_2b_after_sft_env0315_r1"
version = "0.3.15"
args = { split = "train", task_set = "mix_v2_p5050", reward_mode = "p50_50_no_penalty" }

[eval]
interval = 50
num_examples = 64
rollouts_per_example = 1
eval_base_model = false

[[eval.env]]
id = "jayshah5696/humanize-rl-env"
name = "eval_mix_v2_p5050_env0315"
version = "0.3.15"
args = { split = "validation", task_set = "mix_v2_p5050", reward_mode = "p50_50_no_penalty" }
"""


def _write_template(path: Path, text: str = TEMPLATE) -> None:
    path.write_text(text.strip() + "\n")


def _write_handoff_report(
    path: Path,
    *,
    checkpoint_id: str = "ckpt_ready_123456",
    passed: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "prime_warm_start_checkpoint_handoff",
                "passed": passed,
                "checkpoint_id": checkpoint_id,
                "failures": [],
            }
        )
        + "\n"
    )


def _write_promotion_report(
    path: Path,
    *,
    checkpoint_id: str = "ckpt_ready_123456",
    passed: bool = True,
) -> None:
    path.write_text(
        json.dumps(
            {
                "artifact": "sft_promotion_gate",
                "checkpoint_id": checkpoint_id,
                "promotion_gate": {
                    "passed": passed,
                    "failures": [] if passed else ["detector_mimic_gate_failed"],
                },
            }
        )
        + "\n"
    )


def test_prepare_after_sft_config_replaces_checkpoint_and_run_name(tmp_path: Path):
    template = tmp_path / "template.toml"
    output = tmp_path / "ready.toml"
    _write_template(template)

    prepare_after_sft_config(
        template_path=template,
        output_path=output,
        checkpoint_id="ckpt_ready_123456",
        run_name="humanize-after-sft-ckpt-ready",
        allow_unverified_checkpoint=True,
    )

    data = tomllib.loads(output.read_text())
    assert data["checkpoint_id"] == "ckpt_ready_123456"
    assert data["name"] == "humanize-after-sft-ckpt-ready"
    assert data["eval"]["eval_base_model"] is False
    assert data["env"][0]["version"] == "0.3.15"
    assert PLACEHOLDER_CHECKPOINT not in output.read_text()


def test_prepare_after_sft_config_rejects_placeholder_checkpoint(tmp_path: Path):
    template = tmp_path / "template.toml"
    output = tmp_path / "ready.toml"
    _write_template(template)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            PLACEHOLDER_CHECKPOINT,
        ],
    )

    assert result.exit_code == 1
    assert "checkpoint-id must be a READY SFT checkpoint" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_rejects_template_without_placeholder(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    output = tmp_path / "ready.toml"
    _write_template(template, TEMPLATE.replace(PLACEHOLDER_CHECKPOINT, "already-set"))

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--allow-unverified-checkpoint",
        ],
    )

    assert result.exit_code == 1
    assert "template checkpoint_id must be the placeholder" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_cli_requires_checkpoint_handoff_by_default(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    output = tmp_path / "ready.toml"
    _write_template(template)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
        ],
    )

    assert result.exit_code == 1
    assert "checkpoint handoff report is required" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_cli_writes_with_handoff_report(tmp_path: Path):
    template = tmp_path / "template.toml"
    handoff = tmp_path / "checkpoint_handoff.json"
    promotion = tmp_path / "promotion_gate.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    _write_handoff_report(handoff)
    _write_promotion_report(promotion)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(handoff),
            "--promotion-gate-report",
            str(promotion),
        ],
    )

    assert result.exit_code == 0
    assert output.exists()
    assert f"config={output}" in result.output
    assert f"prime --plain train {output} --yes --output json" in result.output


def test_prepare_after_sft_config_requires_promotion_gate_with_handoff(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    handoff = tmp_path / "checkpoint_handoff.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    _write_handoff_report(handoff)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(handoff),
        ],
    )

    assert result.exit_code == 1
    assert "SFT promotion gate report is required" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_cli_can_allow_unverified_validation(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    output = tmp_path / "ready.toml"
    _write_template(template)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--allow-unverified-checkpoint",
        ],
    )

    assert result.exit_code == 0
    assert output.exists()


def test_prepare_after_sft_config_rejects_failed_checkpoint_handoff(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    report = tmp_path / "checkpoint_handoff.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    report.write_text(
        json.dumps(
            {
                "artifact": "prime_warm_start_checkpoint_handoff",
                "passed": False,
                "checkpoint_id": "ckpt_ready_123456",
                "failures": ["checkpoint status PENDING is not READY"],
            }
        )
        + "\n"
    )

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(report),
        ],
    )

    assert result.exit_code == 1
    assert "checkpoint handoff report did not pass" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_rejects_handoff_checkpoint_mismatch(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    report = tmp_path / "checkpoint_handoff.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    report.write_text(
        json.dumps(
            {
                "artifact": "prime_warm_start_checkpoint_handoff",
                "passed": True,
                "checkpoint_id": "other_checkpoint",
                "failures": [],
            }
        )
        + "\n"
    )

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(report),
        ],
    )

    assert result.exit_code == 1
    assert "checkpoint handoff report id other_checkpoint != ckpt_ready_123456" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_rejects_failed_promotion_gate(tmp_path: Path):
    template = tmp_path / "template.toml"
    handoff = tmp_path / "checkpoint_handoff.json"
    promotion = tmp_path / "promotion_gate.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    _write_handoff_report(handoff)
    _write_promotion_report(promotion, passed=False)

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(handoff),
            "--promotion-gate-report",
            str(promotion),
        ],
    )

    assert result.exit_code == 1
    assert "SFT promotion gate report did not pass" in result.output
    assert not output.exists()


def test_prepare_after_sft_config_rejects_promotion_checkpoint_mismatch(
    tmp_path: Path,
):
    template = tmp_path / "template.toml"
    handoff = tmp_path / "checkpoint_handoff.json"
    promotion = tmp_path / "promotion_gate.json"
    output = tmp_path / "ready.toml"
    _write_template(template)
    _write_handoff_report(handoff)
    _write_promotion_report(promotion, checkpoint_id="other_checkpoint")

    result = CliRunner().invoke(
        cli,
        [
            "--template",
            str(template),
            "--output",
            str(output),
            "--checkpoint-id",
            "ckpt_ready_123456",
            "--checkpoint-handoff-report",
            str(handoff),
            "--promotion-gate-report",
            str(promotion),
        ],
    )

    assert result.exit_code == 1
    assert "SFT promotion gate report id other_checkpoint != ckpt_ready_123456" in result.output
    assert not output.exists()
