from __future__ import annotations

from pathlib import Path

import yaml

FULL_CONFIG = Path("configs/rl/gemma4_e2b_rl_a100_full.yaml")
MODAL_SCRIPT = Path("src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py")
VF_EVAL_SCRIPT = Path("scripts/run_vf_eval_modal.py")


def test_full_config_uses_pilot_v3_gpu_maximized_settings() -> None:
    cfg = yaml.safe_load(FULL_CONFIG.read_text())

    assert cfg["num_generations"] == 8
    assert cfg["gradient_accumulation_steps"] == 8
    assert cfg["generation_batch_size"] == 8
    assert cfg["vllm_gpu_memory_utilization"] == 0.65
    assert cfg["max_completion_length"] == 1024
    assert cfg["max_steps"] > 50
    assert cfg["push_to_hub"] is False
    assert cfg["report_to"] == "wandb"
    assert cfg["wandb_project"] == "humanize-rl"


def test_full_config_enables_stratified_batches() -> None:
    cfg = yaml.safe_load(FULL_CONFIG.read_text())

    assert cfg["stratify_batches"] is True
    assert cfg["stratify_by"] == "reward_profile"


def test_modal_entrypoint_mounts_full_config_and_uses_sequential_when_stratified() -> (
    None
):
    source = MODAL_SCRIPT.read_text()

    assert "gemma4_e2b_rl_a100_full.yaml" in source
    assert "stratify_batches" in source
    assert "train_sampling_strategy" in source
    assert "sequential" in source
    assert 'modal.Secret.from_name("wandb")' in source
    assert "WANDB_RUN_ID" in source
    assert "WANDB_RESUME" in source


def test_slice3_vf_eval_modal_script_exists_and_uses_env() -> None:
    source = VF_EVAL_SCRIPT.read_text()

    assert "humanize_rl_env" in source
    assert "verifiers" in source
    assert "SingleTurnEnv" in source
    assert "mean_reward" in source
    assert "risk_penalty_negative_count" in source
