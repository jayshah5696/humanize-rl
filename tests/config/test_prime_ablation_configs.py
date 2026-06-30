from __future__ import annotations

import tomllib
from pathlib import Path

RL_CONFIGS = [
    Path("configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50.toml"),
    Path("configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50_lr5e5.toml"),
    Path("configs/prime/qwen35_2b_p5050_env0315_full200_template.toml"),
    Path("configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml"),
]
SFT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml"
)
SFT_REPAIR_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml"
)


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text())


def test_prime_rl_configs_use_env0315_and_safe_rollout_caps() -> None:
    for path in RL_CONFIGS:
        data = _load(path)
        assert data["env"][0]["version"] == "0.3.15", path
        assert data["sampling"]["max_tokens"] <= 1024, path
        assert data["max_inflight_rollouts"] <= 32, path
        assert data["rollouts_per_example"] <= 8, path
        assert data["eval"]["interval"] <= 50, path
        assert data["eval"]["sampling"]["max_tokens"] <= 1024, path
        for env in data["eval"]["env"]:
            assert env["version"] == "0.3.15", path


def test_prime_qwen_full_configs_keep_checkpoint_boundary_explicit() -> None:
    base = _load(Path("configs/prime/qwen35_2b_p5050_env0315_full200_template.toml"))
    after_sft = _load(
        Path("configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml")
    )

    assert "checkpoint_id" not in base
    assert after_sft["checkpoint_id"] == "FILL_WITH_READY_SFT_CHECKPOINT_ID"
    assert base["eval"]["eval_base_model"] is True
    assert after_sft["eval"]["eval_base_model"] is False


def test_prime_sft_config_keeps_dataset_env_and_gate_env_distinct() -> None:
    data = _load(SFT_CONFIG)

    assert data["model"]["name"] == "Qwen/Qwen3.5-2B"
    assert data["data"]["name"] == "jayshah5696/humanize-rl-prime-sft-messages-env0314"
    assert "gate_env0315" in data["output_dir"]
    assert "data-env0314" in data["wandb"]["tags"]
    assert "gate-env0315" in data["wandb"]["tags"]
    assert data["renderer"]["name"] == "qwen3.5"


def test_prime_sft_repair_config_uses_dedicated_clean50_dataset() -> None:
    data = _load(SFT_REPAIR_CONFIG)

    assert data["model"]["name"] == "Qwen/Qwen3.5-2B"
    assert (
        data["data"]["name"]
        == "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50"
    )
    assert data["val"]["data"]["name"] == data["data"]["name"]
    assert "env0315_clean50" in data["output_dir"]
    assert "data-env0315-clean50" in data["wandb"]["tags"]
    assert "repair-s2" in data["wandb"]["tags"]
    assert data["renderer"]["name"] == "qwen3.5"
