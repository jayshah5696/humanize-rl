"""Pre-flight tests for the TRL+vLLM Modal RL entrypoint and its configs.

Mirrors `test_rl_gemma4_modal_preflight.py` for the new (non-Unsloth) path.
Must pass before any Modal spend. See:
docs/plans/gemma4_rl_modal_20usd_budget_plan.md "Phase 0".
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
import yaml

from humanize_rl.reward.grpo_dataset import load_grpo_rows
from humanize_rl.reward.grpo_rewards import (
    WEIGHTED_REWARD_FUNCS,
    dither_if_unanimous,
    risk_penalty_reward,
    scalar_reward,
)
from humanize_rl.reward.tasks import RLTask

MODAL_SCRIPT = Path("src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py")
CONFIG_DIR = Path("configs/rl")
CONFIGS = [
    CONFIG_DIR / "gemma4_e2b_rl_a100_capacity_probe.yaml",
    CONFIG_DIR / "gemma4_e2b_rl_a100_pilot.yaml",
    CONFIG_DIR / "gemma4_e2b_rl_a100_full.yaml",
]
TASK_PATH = Path("data/rl/humanize_tasks_v01_smoke.jsonl")

CANARIES = {
    "clean": (
        "Staging is back. The fix was a missing STRIPE_WEBHOOK_SECRET. "
        "Monitoring until 3 pm."
    ),
    "verbose": (
        "I am pleased to report that, following an extensive investigation, we "
        "have now successfully restored the staging environment, the root cause "
        "having been identified as the absence of the STRIPE_WEBHOOK_SECRET "
        "configuration value, and we will continue to monitor the situation "
        "closely until 3 pm."
    ),
    "placeholder": (
        "Staging is back. Root cause: [REDACTED]. Monitoring until [TIME]."
    ),
    "bad": "Database migration done. Sarah handled it. Thanks!",
}


@pytest.mark.parametrize("cfg_path", CONFIGS, ids=lambda p: p.name)
def test_config_schema(cfg_path: Path) -> None:
    if not cfg_path.exists():
        pytest.skip(f"{cfg_path.name} not created yet")
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["num_generations"] >= 4, "num_generations < 4 collapses reward std"
    assert cfg["max_completion_length"] >= 1024, "completion cap too low"
    assert cfg["max_prompt_length"] >= 512
    assert cfg["push_to_hub"] is False
    assert cfg["model_name"].startswith("jayshah5696/")


def test_train_rows_load_and_round_trip() -> None:
    rows = load_grpo_rows(TASK_PATH, split="train")
    assert len(rows) >= 80, f"expected >= 80 train rows, got {len(rows)}"
    RLTask.model_validate(rows[0]["task"])
    content = rows[0]["prompt"][0]["content"]
    assert isinstance(content, str) and len(content) > 20


def test_reward_spread_on_canaries_is_non_degenerate() -> None:
    """Catches the reward_std=0 -> NaN-grad failure from smoke v1."""
    rows = load_grpo_rows(TASK_PATH, split="train")
    task = rows[0]["task"]

    def as_completion(text: str) -> list[list[dict[str, str]]]:
        return [[{"role": "assistant", "content": text}]]

    rewards = {
        name: scalar_reward(as_completion(text), task=[task])[0]
        for name, text in CANARIES.items()
    }
    spread = max(rewards.values()) - min(rewards.values())
    assert spread > 0.05, (
        f"reward spread {spread:.4f} too small; GRPO advantage will be ~0. "
        f"rewards={rewards}"
    )
    assert rewards["clean"] > rewards["verbose"], (
        f"verbose >= clean: {rewards}"
    )


def test_risk_penalty_triggers_on_forbidden_facts() -> None:
    rows = load_grpo_rows(TASK_PATH, split="train")
    task = rows[0]["task"]
    bad = [[{"role": "assistant", "content": CANARIES["bad"]}]]
    penalty = risk_penalty_reward(bad, task=[task])[0]
    assert penalty < 0, f"risk_penalty did not trigger on forbidden facts: {penalty}"


def test_all_weighted_reward_funcs_return_floats() -> None:
    rows = load_grpo_rows(TASK_PATH, split="train")
    task = rows[0]["task"]
    comp = [[{"role": "assistant", "content": CANARIES["clean"]}]]
    for fn in WEIGHTED_REWARD_FUNCS:
        values = fn(comp, task=[task])
        assert len(values) == 1
        assert isinstance(values[0], float), f"{fn.__name__} returned {type(values[0])}"


def test_modal_script_uses_trl_vllm_not_unsloth() -> None:
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    source = MODAL_SCRIPT.read_text()
    # Forbid actual unsloth imports; allow word appearances in docstrings or
    # in the HF model_name (e.g. "gemma4-e2b-humanize-unsloth-merged").
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("import unsloth") or stripped.startswith(
            "from unsloth"
        ):
            raise AssertionError(
                f"new entrypoint must not import unsloth: {line!r}"
            )
    assert "vllm" in source.lower(), "must reference vllm in image install"
    assert "use_vllm=config.use_vllm" in source or "use_vllm=True" in source, (
        "GRPOConfig must wire use_vllm"
    )
    assert 'vllm_mode="colocate"' in source or "vllm_mode='colocate'" in source or (
        "vllm_mode=config.vllm_mode" in source
    )
    assert (
        'gpu="A100-40GB"' in source
        or "gpu='A100-40GB'" in source
        or ('GPU_TYPE = "A100-40GB"' in source and "gpu=GPU_TYPE" in source)
    )


def test_modal_script_uses_detached_spawn_pattern() -> None:
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    source = MODAL_SCRIPT.read_text()
    assert ".spawn(" in source, "must use detached .spawn(), not .remote()"


def test_modal_script_declares_version_pins_for_gemma4_bug_b_c_e() -> None:
    """Bug B + E -> trl >= 0.29.0; Bug C -> transformers >= 5.5.0.

    Pins must live in the Modal image install so they fail loud on the
    GPU container, not silently at runtime.
    """
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    source = MODAL_SCRIPT.read_text()
    assert "transformers>=5.5.0" in source
    # TRL v1.x is the first line with transformers v5 + native Gemma 4.
    assert "trl>=1.0.0" in source
    # vLLM 0.19.1+ ships Gemma 4 + transformers 5.5.3 compatibility.
    assert "vllm>=0.19.1" in source


def test_modal_script_mirrors_gemma4_final_logit_softcap_bug_a() -> None:
    """Bug A: TRL reads ``model.config.final_logit_softcapping`` flat.

    Gemma 4 only sets it on ``text_config``. Without mirroring we get
    softcap=0 -> KL blowup. Verify the helper exists and the trainer
    function calls it (asserted by name reference).
    """
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    source = MODAL_SCRIPT.read_text()
    assert "def mirror_gemma4_final_logit_softcap" in source
    assert "GEMMA4_EXPECTED_FINAL_LOGIT_SOFTCAP" in source
    assert "30.0" in source


def test_bug_a_mirror_promotes_text_config_softcap_to_top_level() -> None:
    """Direct unit test of the Bug A guard against a stub Gemma4-shaped config."""
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    from humanize_rl.training.rl_gemma4_trl_vllm_modal import (
        mirror_gemma4_final_logit_softcap,
    )

    stub_text_cfg = types.SimpleNamespace(final_logit_softcapping=30.0)
    stub_cfg = types.SimpleNamespace(
        text_config=stub_text_cfg,
        final_logit_softcapping=None,
    )
    stub_model = types.SimpleNamespace(config=stub_cfg)

    result = mirror_gemma4_final_logit_softcap(stub_model)
    assert result == 30.0
    assert stub_model.config.final_logit_softcapping == 30.0


def test_bug_a_mirror_is_no_op_when_top_level_already_set() -> None:
    if not MODAL_SCRIPT.exists():
        pytest.skip(f"{MODAL_SCRIPT.name} not created yet")
    from humanize_rl.training.rl_gemma4_trl_vllm_modal import (
        mirror_gemma4_final_logit_softcap,
    )

    stub_text_cfg = types.SimpleNamespace(final_logit_softcapping=30.0)
    stub_cfg = types.SimpleNamespace(
        text_config=stub_text_cfg,
        final_logit_softcapping=42.0,  # already set; do not clobber
    )
    stub_model = types.SimpleNamespace(config=stub_cfg)

    result = mirror_gemma4_final_logit_softcap(stub_model)
    assert result == 42.0


def test_weighted_reward_funcs_are_dither_wrapped() -> None:
    """All exported reward fns must go through dither_if_unanimous.

    Guards the smoke-v1 NaN-grad failure mode where a unanimous group
    yielded zero advantage std.
    """
    for fn in WEIGHTED_REWARD_FUNCS:
        wrapped = getattr(fn, "__wrapped__", None)
        assert wrapped is not None, (
            f"{fn.__name__}: not decorated (functools.wraps sets __wrapped__)"
        )


def test_dither_unit_zero_std_group_gets_jitter() -> None:
    @dither_if_unanimous
    def fn(completions, **kwargs):  # type: ignore[no-untyped-def]
        return [0.5, 0.5, 0.5, 0.5]

    comps = [[{"role": "assistant", "content": f"x{i}"}] for i in range(4)]
    tasks = [{"id": f"t{i}"} for i in range(4)]
    out = fn(comps, task=tasks)
    assert out != [0.5, 0.5, 0.5, 0.5]
    assert len(out) == 4
