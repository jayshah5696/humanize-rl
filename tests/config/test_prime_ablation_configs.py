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
OPEN_PRIME_RL_AFTER_SFT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_rl_after_sft_env0315_clean50_step200_full200.toml"
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
        == "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat"
    )
    assert data["val"]["data"]["name"] == data["data"]["name"]
    assert "env0315_clean50" in data["output_dir"]
    assert "data-env0315-clean50" in data["wandb"]["tags"]
    assert "repair-s2" in data["wandb"]["tags"]
    assert data["renderer"]["name"] == "qwen3.5"


def test_open_prime_rl_after_sft_config_uses_real_step200_hf_checkpoint() -> None:
    data = _load(OPEN_PRIME_RL_AFTER_SFT_CONFIG)

    assert (
        data["model"]["name"]
        == "jayshah5696/humanize-rl-qwen35-2b-sft-env0315-clean50-primecompat-step200"
    )
    assert data["max_steps"] == 200
    assert data["deployment"]["type"] == "single_node"
    assert data["deployment"]["gpus_per_node"] == 2
    assert data["deployment"]["num_train_gpus"] == 1
    assert data["deployment"]["num_infer_gpus"] == 1
    assert data["weight_broadcast"]["type"] == "filesystem"
    assert data["trainer"]["model"]["lora"]["rank"] == 32
    assert data["trainer"]["model"]["lora"]["alpha"] == 64
    assert data["orchestrator"]["renderer"]["name"] == "qwen3.5"
    assert data["orchestrator"]["renderer"]["enable_thinking"] is False
    assert data["orchestrator"]["batch_size"] == 64
    assert data["orchestrator"]["group_size"] == 8
    assert data["orchestrator"]["max_inflight_rollouts"] == 32
    train_env = data["orchestrator"]["train"]["env"][0]
    assert train_env["id"] == "jayshah5696/humanize-rl-env"
    assert train_env["args"]["task_set"] == "mix_v2_p5050"
    assert train_env["args"]["reward_mode"] == "p50_50_no_penalty"
    eval_envs = data["orchestrator"]["eval"]["env"]
    assert {env["args"]["task_set"] for env in eval_envs} == {
        "mix_v2_p5050",
        "v02_smoke",
        "v03",
    }
    assert data["trainer"]["wandb"]["offline"] is True
    assert data["orchestrator"]["wandb"]["offline"] is True
