"""Tests for v03 RLTask + TaskConstraints extensions.

See docs/plans/v03-rl-tasks-dataset.md §2.3 (constraint vocabulary) and
§2.1 (six task modes).
"""

from __future__ import annotations

import json

import pytest

from humanize_rl.reward.tasks import RLTask, TaskConstraints, load_tasks


def _v03_task(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "rl_v03_000001",
        "family": "rewrite_repair",
        "domain": "email",
        "mode": "rewrite_humanize",
        "register": "direct",
        "instruction": "Rewrite this email to sound human.",
        "input_text": "Dear Sir, I am writing to inform you.",
        "constraints": {
            "target_words": 80,
            "target_tolerance": 0.2,
            "must_include_phrases": ["status update"],
            "must_not_include_phrases": ["unlock", "seamless"],
            "forbidden_openers": ["I hope this finds you well"],
            "must_not_use_em_dash": True,
            "register_target": "direct",
            "min_contraction_count": 2,
            "max_passive_voice_pct": 25,
            "required_section_headings": [],
            "persona_constraint": "senior PM at fintech",
            "audience_constraint": "engineers",
            "must_contain_number_kind": "percent",
        },
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
        "task_author_model": "google/gemini-3.1-pro-preview",
    }
    base.update(overrides)
    return base


def test_v03_task_id_pattern_accepted() -> None:
    task = RLTask.model_validate(_v03_task())
    assert task.id == "rl_v03_000001"


def test_v03_new_modes_accepted() -> None:
    for mode, needs_src in [
        ("rewrite_humanize", True),
        ("compression", True),
        ("tone_shift", True),
        ("expansion", True),
        ("long_form_generate", False),
        ("multi_constraint_compose", False),
    ]:
        kw = _v03_task(mode=mode)
        if not needs_src:
            kw["input_text"] = "Brief: write a launch announcement."
        RLTask.model_validate(kw)


def test_v03_new_registers_accepted() -> None:
    for reg in ["warm", "direct", "formal", "academic", "journalistic", "literary"]:
        RLTask.model_validate(_v03_task(register=reg))


def test_v03_constraints_target_words_with_tolerance() -> None:
    c = TaskConstraints.model_validate(
        {"target_words": 150, "target_tolerance": 0.25}
    )
    assert c.target_words == 150
    assert c.target_tolerance == pytest.approx(0.25)


def test_v03_constraints_must_include_phrases_capped() -> None:
    """Plan §2.3: hard cap of 3 must_include_phrases per task."""
    with pytest.raises(ValueError, match="must_include_phrases"):
        TaskConstraints.model_validate(
            {"must_include_phrases": ["a", "b", "c", "d"]}
        )


def test_v03_constraints_structure_fields() -> None:
    c = TaskConstraints.model_validate(
        {
            "min_sentences": 3,
            "max_sentences": 8,
            "min_paragraphs": 1,
            "max_paragraphs": 4,
            "required_section_headings": ["Background", "Decision"],
            "allow_bullets": False,
            "allow_headings": True,
        }
    )
    assert c.min_sentences == 3
    assert c.max_sentences == 8
    assert c.required_section_headings == ["Background", "Decision"]
    assert c.allow_bullets is False


def test_v03_constraints_register_target_enum() -> None:
    for r in [
        "casual",
        "warm",
        "direct",
        "formal",
        "academic",
        "journalistic",
        "literary",
    ]:
        TaskConstraints.model_validate({"register_target": r})


def test_v03_constraints_must_contain_number_kind_enum() -> None:
    for k in ["percent", "currency", "count", "date", "time", "any"]:
        TaskConstraints.model_validate({"must_contain_number_kind": k})


def test_v03_constraints_format_fields() -> None:
    c = TaskConstraints.model_validate(
        {
            "no_markdown": True,
            "must_not_use_em_dash": True,
            "forbidden_openers": ["I'm excited to"],
            "forbidden_phrases_global": ["moving forward"],
        }
    )
    assert c.must_not_use_em_dash is True
    assert c.forbidden_openers == ["I'm excited to"]


def test_v03_constraints_min_max_sentences_validated() -> None:
    with pytest.raises(ValueError, match="min_sentences"):
        TaskConstraints.model_validate(
            {"min_sentences": 10, "max_sentences": 3}
        )


def test_v03_task_author_model_optional() -> None:
    task = RLTask.model_validate(_v03_task(task_author_model=None))
    assert task.task_author_model is None


def test_v03_load_tasks_round_trip(tmp_path) -> None:
    path = tmp_path / "v03.jsonl"
    path.write_text(json.dumps(_v03_task()) + "\n")
    tasks = load_tasks(path)
    assert tasks[0].id == "rl_v03_000001"
    assert tasks[0].mode == "rewrite_humanize"
    assert tasks[0].task_author_model == "google/gemini-3.1-pro-preview"


def test_v01_task_still_valid() -> None:
    """v03 schema extension must not break v01 tasks."""
    legacy = {
        "id": "rl_v01_000001",
        "family": "rewrite_repair",
        "domain": "slack",
        "mode": "rewrite",
        "register": "casual",
        "instruction": "Clean up.",
        "input_text": "Staging recovered.",
        "constraints": {"max_words": 30, "preserve_numbers": True},
        "reward_profile": "rewrite_faithful_concise",
        "trap_tags": ["wrapper_phrase"],
        "split": "train",
    }
    task = RLTask.model_validate(legacy)
    assert task.id == "rl_v01_000001"
    assert task.mode == "rewrite"


def test_v03_long_form_modes_with_brief_input() -> None:
    """long_form_generate / multi_constraint_compose use short brief as input_text."""
    for mode in ["long_form_generate", "multi_constraint_compose"]:
        task = RLTask.model_validate(
            _v03_task(mode=mode, input_text="Brief: launch post for Q3.")
        )
        assert task.input_text.startswith("Brief:")
