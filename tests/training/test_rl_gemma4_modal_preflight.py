from __future__ import annotations

from pathlib import Path

import yaml

from humanize_rl.reward.grpo_dataset import load_grpo_rows

MODAL_SCRIPT = Path("src/humanize_rl/training/rl_gemma4_modal.py")
CONFIG_PATH = Path("configs/rl/gemma4_e2b_rl_smoke.yaml")
TASK_PATH = Path("data/rl/humanize_tasks_v01_smoke.jsonl")


def test_smoke_config_is_tiny_and_uses_merged_sft_model() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())

    assert config["model_name"] == "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
    assert config["max_steps"] <= 2
    assert config["num_generations"] >= 2
    assert config["push_to_hub"] is False
    assert config["sample_before_after"] <= 2
    assert config["generation_batch_size"] >= 2
    assert config["artifact_generation_samples"] <= 2


def test_grpo_rows_use_trl_text_prompt_shape() -> None:
    rows = load_grpo_rows(TASK_PATH, split="validation")

    assert rows
    content = rows[0]["prompt"][0]["content"]
    assert isinstance(content, str)
    assert "Source:" in content


def test_modal_image_imports_do_not_import_project_before_src_mount_path() -> None:
    source = MODAL_SCRIPT.read_text()
    imports_block = source.split("with image.imports():", maxsplit=1)[1].split(
        "@dataclass", maxsplit=1
    )[0]

    assert "humanize_rl." not in imports_block
    assert "from trl import" not in imports_block
    assert "import unsloth" in imports_block


def test_modal_script_uses_detached_spawn_for_every_remote_mode() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "def verify_model_artifact" in source
    assert "def preflight_grpo_setup" in source
    assert "verify_model_artifact.spawn" in source
    assert "preflight_grpo_setup.spawn" in source
    assert "train_grpo.spawn" in source
    assert "verify_model_artifact.remote" not in source
    assert "preflight_grpo_setup.remote" not in source


def test_modal_script_batches_before_after_generation_on_a100() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "generation_batch_size" in source
    assert "def _generation_texts" in source
    assert "padding=True" in source


def test_modal_verification_modes_persist_summaries_to_checkpoint_volume() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "artifact_verification.json" in source
    assert "grpo_preflight_summary.json" in source
    assert "checkpoint_volume.commit()" in source


def test_modal_script_drops_unused_gemma4_mm_token_type_ids_for_grpo_generate() -> None:
    source = MODAL_SCRIPT.read_text()

    assert "def _drop_unused_mm_token_type_ids_for_generate" in source
    assert 'kwargs.pop("mm_token_type_ids", None)' in source
