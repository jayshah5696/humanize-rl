"""Tests for the artifact-hygiene tooling.

Covers:
- ``_model_cards`` templates produce the strings we depend on downstream.
- ``verified_merge_and_push_modal`` enforces gates before push, uses
  ``.spawn()``, and writes the expected report files.
- ``push_model_cards_modal`` exists, uses ``.spawn()``, and re-exports the
  templates from the same source of truth.
"""

from __future__ import annotations

from pathlib import Path

from humanize_rl.training._model_cards import (
    lora_model_card,
    merged_model_card,
)

VERIFIED = Path("src/humanize_rl/training/verified_merge_and_push_modal.py")
PUSH_CARDS = Path("src/humanize_rl/training/push_model_cards_modal.py")
TEMPLATES = Path("src/humanize_rl/training/_model_cards.py")


def test_merged_model_card_documents_kv_shared_layer_design() -> None:
    card = merged_model_card()
    assert "num_kv_shared_layers: 20" in card
    assert "45328" in card  # transformers PR ref
    assert "Layers 15-34" in card
    assert "5386" in card  # Unsloth eos_token issue ref
    assert "<turn|>" in card
    assert "Apache-2.0" in card


def test_merged_model_card_includes_verification_table() -> None:
    card = merged_model_card(
        parity_prompt_count=10,
        parity_mismatch_count=1,
        transformers_missing_keys=0,
        transformers_unexpected_keys=0,
    )
    assert "9/10 identical" in card
    assert "missing_keys" in card
    assert "unexpected_keys" in card
    assert "Verification report" in card


def test_lora_model_card_points_at_merged_repo() -> None:
    card = lora_model_card(lora_rank=8, lora_alpha=8)
    assert "jayshah5696/gemma4-e2b-humanize-unsloth-merged" in card
    assert "LoRA rank: `8`" in card
    assert "KV-shared layers" in card
    assert "45328" in card


def test_verified_merge_script_uses_spawn_only() -> None:
    source = VERIFIED.read_text()
    assert "verified_merge_and_push.spawn(" in source
    assert "verified_merge_and_push.remote(" not in source


def test_verified_merge_script_pushes_only_after_gates_pass() -> None:
    source = VERIFIED.read_text()
    # Tokenizer and structural gates must compose into all_pass.
    assert "all_pass" in source
    assert "gates_pass" in source
    # Push must be guarded by gates_pass and the push flag.
    assert 'reason"] = "gates_failed"' in source
    # Default candidate repo must NOT clobber the production merged repo.
    assert "merged-peft-v2" in source
    assert (
        '"jayshah5696/gemma4-e2b-humanize-unsloth-merged"'
        not in source.split("candidate_repo: str =", maxsplit=1)[1].split(",", maxsplit=1)[0]
    )


def test_verified_merge_script_emits_verification_report_json() -> None:
    source = VERIFIED.read_text()
    assert "verification_report.json" in source
    assert "wrongly_present_shared_kv_keys" in source
    assert "transformers_missing_keys" in source
    assert "tokenizer_config.json" in source


def test_verified_merge_script_restores_eos_token_on_regression() -> None:
    source = VERIFIED.read_text()
    assert 'eos_token_expected"] = "<turn|>"' in source
    assert 'tok_cfg["eos_token"] = "<turn|>"' in source


def test_push_model_cards_script_uses_spawn_and_shared_templates() -> None:
    source = PUSH_CARDS.read_text()
    assert "push_cards.spawn(" in source
    assert "push_cards.remote(" not in source
    assert "from _model_cards import" in source
    assert "merged_model_card" in source
    assert "lora_model_card" in source


def test_templates_module_does_not_import_modal_or_transformers_at_top_level() -> None:
    # Strip docstrings + template literals so example code blocks inside
    # the README templates do not give false positives.
    import ast

    tree = ast.parse(TEMPLATES.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name.split(".", maxsplit=1)[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported.add(node.module.split(".", maxsplit=1)[0])
    assert "modal" not in imported
    assert "torch" not in imported
    assert "transformers" not in imported
    assert "peft" not in imported
