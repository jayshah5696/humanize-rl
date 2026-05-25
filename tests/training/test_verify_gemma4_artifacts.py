"""Tests for the Gemma 4 artifact verifier and the false-alarm guard.

These tests do not call Modal or HuggingFace; they only assert on the
source structure of ``verify_gemma4_artifacts_modal.py`` and the helper
``_assert_only_shared_kv_keys_missing`` semantics so that we never
regress to silently accepting a checkpoint with real missing weights.
"""

from __future__ import annotations

from pathlib import Path

VERIFIER = Path("src/humanize_rl/training/verify_gemma4_artifacts_modal.py")
RL_SCRIPT = Path("src/humanize_rl/training/rl_gemma4_modal.py")
SMOKE_V2 = Path("configs/rl/gemma4_e2b_rl_smoke_v2.yaml")


def test_verifier_script_uses_detached_spawn_only() -> None:
    source = VERIFIER.read_text()
    assert "verify.spawn(" in source
    assert "verify.remote(" not in source


def test_verifier_documents_kv_shared_layer_design() -> None:
    source = VERIFIER.read_text()
    assert "num_kv_shared_layers" in source
    assert "45328" in source  # transformers PR that omits these keys
    assert "5386" in source  # Unsloth eos_token regression issue


def test_verifier_writes_expected_report_files() -> None:
    source = VERIFIER.read_text()
    assert '"report.json"' in source
    assert '"direct_generations.jsonl"' in source
    assert '"merged_generations.jsonl"' in source


def test_verifier_does_not_push_tokenizer_by_default() -> None:
    source = VERIFIER.read_text()
    # main entry default is push_fixed_tokenizer=False
    assert "push_fixed_tokenizer: bool = False" in source
    assert "fix_tokenizer: bool = False" in source


def test_shared_layer_keys_match_gemma4_e2b_layout() -> None:
    # Re-derive the helper here so the test does not import the modal-only
    # module. The verifier source must keep the same indexing rule.
    def _shared_layer_keys(num_hidden_layers: int, num_kv_shared_layers: int) -> list[str]:
        start = num_hidden_layers - num_kv_shared_layers
        keys: list[str] = []
        for layer in range(start, num_hidden_layers):
            for name in ("k_proj.weight", "v_proj.weight", "k_norm.weight", "v_norm.weight"):
                keys.append(f"model.language_model.layers.{layer}.self_attn.{name}")
        return keys

    keys = _shared_layer_keys(num_hidden_layers=35, num_kv_shared_layers=20)
    assert len(keys) == 20 * 4
    assert "model.language_model.layers.15.self_attn.k_proj.weight" in keys
    assert "model.language_model.layers.34.self_attn.v_norm.weight" in keys
    assert "model.language_model.layers.14.self_attn.k_proj.weight" not in keys

    # And the verifier source must contain the same indexing constants.
    src = VERIFIER.read_text()
    assert "num_hidden_layers - num_kv_shared_layers" in src
    assert "v_norm.weight" in src


def test_rl_loader_guards_against_real_zero_kv_weights() -> None:
    source = RL_SCRIPT.read_text()
    assert "_assert_only_shared_kv_keys_missing" in source
    assert "num_kv_shared_layers" in source
    assert "Non-shared decoder layers have zero k/v weights" in source


def test_smoke_v2_widens_group_size_and_softens_clipping() -> None:
    import yaml

    cfg = yaml.safe_load(SMOKE_V2.read_text())
    assert cfg["num_generations"] >= 4, "num_generations=2 caused KL blow-up"
    assert cfg["epsilon_high"] <= 0.21
    assert cfg["delta"] <= 1.3
    assert cfg["learning_rate"] <= 1e-5
    assert cfg["push_to_hub"] is False
