"""Tests for scripts/data/build/verify_rl_task_quality.py — gate evaluators."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_module():
    root = Path(__file__).resolve().parents[2]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    path = root / "scripts" / "data" / "build" / "verify_rl_task_quality.py"
    spec = importlib.util.spec_from_file_location("verify_rl_task_quality", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["verify_rl_task_quality"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_task(**overrides):
    from humanize_rl.reward.tasks import RLTask
    base = {
        "id": "rl_v03_000001",
        "family": "rewrite_repair",
        "domain": "email",
        "mode": "rewrite_humanize",
        "register": "direct",
        "instruction": "x " * 150,  # 150-word instruction
        "input_text": "src.",
        "constraints": {
            "target_words": 50, "register_target": "direct",
            "must_not_use_em_dash": True, "must_include_phrases": ["foo"],
            "forbidden_openers": ["I'm excited"],
            "min_contraction_count": 2,
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
        "task_author_model": "google/gemini-3.1-pro-preview",
    }
    base.update(overrides)
    return RLTask.model_validate(base)


def test_A1_pass_when_p90_meets_threshold() -> None:
    mod = _load_module()
    # Lengths spanning ~50 to ~250 so p10≈70, p50≈150, p90≈230.
    tasks = [_make_task(id=f"rl_v03_{i:06d}", instruction="x " * (50 + i * 2))
             for i in range(1, 121)]
    th = {"hard": {"p90_min": 200, "p10_max": 80},
          "soft": {"p50_min": 110, "p50_max": 170}}
    result = mod.gate_A1(tasks, th)
    assert result["status"] == "PASS"


def test_A1_fail_when_too_short() -> None:
    mod = _load_module()
    tasks = [_make_task(id=f"rl_v03_{i:06d}", instruction="x " * 30) for i in range(1, 11)]
    th = {"hard": {"p90_min": 200, "p10_max": 80},
          "soft": {"p50_min": 110, "p50_max": 170}}
    assert mod.gate_A1(tasks, th)["status"] == "FAIL"


def test_A2_constraint_density() -> None:
    mod = _load_module()
    tasks = [_make_task(id=f"rl_v03_{i:06d}") for i in range(1, 11)]
    th = {"hard_min": 4, "soft_min": 5}
    result = mod.gate_A2(tasks, th)
    assert result["status"] in {"PASS", "WARN"}
    assert result["value"] >= 4


def test_A5_author_balance_pass_when_aligned() -> None:
    mod = _load_module()
    tasks = []
    authors = (["google/gemini-3.1-pro-preview"] * 40
               + ["openai/gpt-5.4-mini"] * 35
               + ["google/gemini-3.1-flash-lite-preview"] * 25)
    for i, a in enumerate(authors, start=1):
        tasks.append(_make_task(id=f"rl_v03_{i:06d}", task_author_model=a))
    th = {
        "hard_max_pp": 10, "soft_max_pp": 5,
        "target_shares": {
            "google/gemini-3.1-pro-preview": 0.40,
            "openai/gpt-5.4-mini": 0.35,
            "google/gemini-3.1-flash-lite-preview": 0.25,
        },
    }
    assert mod.gate_A5(tasks, th)["status"] == "PASS"


def test_B5_penalty_concentration_fails_when_dominant() -> None:
    mod = _load_module()
    rollouts = {f"rl_v03_{i:06d}": [
        {"reward": 0.5, "model": "x", "penalties": {"missing_number": -0.4}}
        for _ in range(3)
    ] for i in range(1, 6)}
    th = {"hard_max": 0.50, "soft_max": 0.40}
    assert mod.gate_B5(rollouts, th)["status"] == "FAIL"


def test_spearman_perfect() -> None:
    mod = _load_module()
    assert mod._spearman_rho([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert mod._spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_render_markdown_includes_all_gates() -> None:
    mod = _load_module()
    report = {
        "n_tasks": 10, "n_rollouts": 30, "hard_pass": 1, "hard_warn": 0,
        "hard_fail": 0,
        "gates": [{"gate": "A1_instruction_length", "status": "PASS",
                   "value": 100, "hard": "h", "soft": "s"}],
    }
    md = mod.render_markdown(report)
    assert "A1_instruction_length" in md
    assert "PASS" in md
