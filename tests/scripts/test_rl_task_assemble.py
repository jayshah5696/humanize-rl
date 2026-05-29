"""Tests for scripts/data/build/rl_task_assemble.py."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_module():
    """Load the script as a module so we can call its helpers."""
    root = Path(__file__).resolve().parents[2]
    src_path = root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    path = root / "scripts" / "data" / "build" / "rl_task_assemble.py"
    spec = importlib.util.spec_from_file_location("rl_task_assemble", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["rl_task_assemble"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_extract_payload_wrapped_json() -> None:
    mod = _load_module()
    payload = {"instruction": "Hi", "input_text": "src", "constraints": {}}
    outer = json.dumps({"text": json.dumps(payload)})
    out = mod._extract_payload(outer)
    assert out == payload


def test_extract_payload_direct_json() -> None:
    mod = _load_module()
    payload = {"instruction": "Hi", "input_text": "src", "constraints": {}}
    out = mod._extract_payload(json.dumps(payload))
    assert out == payload


def test_extract_payload_with_fence() -> None:
    mod = _load_module()
    payload = {"instruction": "Hi", "input_text": "src", "constraints": {}}
    wrapped = f"```json\n{json.dumps({'text': json.dumps(payload)})}\n```"
    assert mod._extract_payload(wrapped) == payload


def test_infer_mode_from_path() -> None:
    mod = _load_module()
    assert mod._infer_mode_from_path(Path("v03_rewrite_humanize_x.jsonl")) == "rewrite_humanize"
    assert mod._infer_mode_from_path(Path("v03_compression_y.jsonl")) == "compression"
    assert mod._infer_mode_from_path(Path("garbage.jsonl")) is None


def test_infer_author_from_path() -> None:
    mod = _load_module()
    assert mod._infer_author_from_path(
        Path("v03_rewrite_humanize_google_gemini-3.1-pro-preview.jsonl")
    ) == "google/gemini-3.1-pro-preview"
    assert mod._infer_author_from_path(Path("v03_rewrite_humanize.jsonl")) is None


def test_build_task_constructs_valid_rltask() -> None:
    mod = _load_module()
    payload = {
        "instruction": "Rewrite this email to sound human.",
        "input_text": "Dear Sir, I am writing.",
        "constraints": {
            "target_words": 50,
            "must_not_use_em_dash": True,
            "register_target": "direct",
        },
        "domain": "email",
        "register": "direct",
        "trap_tags": ["wrapper_phrase"],
        "reward_profile": "rewrite_faithful_concise",
    }
    task = mod._build_task(
        payload, "rewrite_humanize", "google/gemini-3.1-pro-preview",
        "src-ds", "row-1", 7,
    )
    assert task.id == "rl_v03_000007"
    assert task.mode == "rewrite_humanize"
    assert task.constraints.target_words == 50
    assert task.task_author_model == "google/gemini-3.1-pro-preview"


def test_assign_splits_deterministic() -> None:
    mod = _load_module()
    from humanize_rl.reward.tasks import RLTask
    base = {
        "id": "rl_v03_000001",
        "family": "rewrite_repair",
        "domain": "email",
        "mode": "rewrite_humanize",
        "register": "direct",
        "instruction": "Rewrite.",
        "input_text": "src.",
        "constraints": {"target_words": 50},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
    }
    tasks = [
        RLTask.model_validate({**base, "id": f"rl_v03_{i:06d}"})
        for i in range(1, 11)
    ]
    out = mod._assign_splits(tasks, (0.8, 0.1, 0.1), 42)
    splits = [t.split for t in out]
    assert splits.count("train") == 8
    assert splits.count("validation") == 1
    assert splits.count("test") == 1
    # determinism: same seed → same splits
    out2 = mod._assign_splits(tasks, (0.8, 0.1, 0.1), 42)
    assert [t.split for t in out2] == splits


def test_assemble_end_to_end(tmp_path) -> None:
    mod = _load_module()
    inp = tmp_path / "v03_rewrite_humanize_google_gemini-3.1-pro-preview.jsonl"
    payload = {
        "instruction": "Rewrite this email to sound natural and human, 80 words.",
        "input_text": "Dear Sir, I am writing to follow up.",
        "constraints": {
            "target_words": 50, "register_target": "direct",
            "must_not_use_em_dash": True,
        },
        "domain": "email",
        "register": "direct",
        "trap_tags": ["wrapper_phrase"],
        "reward_profile": "rewrite_faithful_concise",
    }
    arka_row = {
        "instruction": "ignored",
        "response": json.dumps({"text": json.dumps(payload)}),
    }
    inp.write_text(json.dumps(arka_row) + "\n")

    out = tmp_path / "tasks.jsonl"
    from click.testing import CliRunner
    runner = CliRunner()
    result = runner.invoke(mod.main, [
        "--inputs", str(inp), "--out", str(out),
        "--split", "1.0", "0.0", "0.0", "--split-seed", "0",
    ])
    assert result.exit_code == 0, result.output
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["mode"] == "rewrite_humanize"
    assert row["task_author_model"] == "google/gemini-3.1-pro-preview"
