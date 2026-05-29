"""RL task schema and JSONL validation for the humanize reward environment."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

TaskFamily = Literal[
    "rewrite_repair",
    "direct_email",
    "slack_chat",
    "compression",
    "tone_shift",
    "technical_explain",
    "product_copy_cleanup",
    "candidate_customer_comms",
    "adversarial_ai_tell_removal",
    "placeholder_discipline",
]

TaskDomain = Literal[
    "email",
    "slack",
    "technical",
    "product",
    "leadership",
    "support",
    "hiring",
    "creative_general",
]

TaskMode = Literal[
    # v01 modes (kept for backward compatibility)
    "direct_generation",
    "rewrite",
    "compress",
    "tone_shift",
    "repair",
    # v03 modes (see docs/plans/v03-rl-tasks-dataset.md §2.1)
    "long_form_generate",
    "rewrite_humanize",
    "compression",
    "expansion",
    "multi_constraint_compose",
]

TaskRegister = Literal[
    # v01 registers
    "casual",
    "neutral",
    "warm_professional",
    "candid",
    "terse",
    "technical",
    "founder_like",
    # v03 registers (see plan §2.5)
    "warm",
    "direct",
    "formal",
    "academic",
    "journalistic",
    "literary",
]

RegisterTarget = Literal[
    "casual",
    "warm",
    "direct",
    "formal",
    "academic",
    "journalistic",
    "literary",
]

NumberKind = Literal["percent", "currency", "count", "date", "time", "any"]

TaskSplit = Literal["train", "validation", "test"]

RewardProfileName = Literal[
    "rewrite_faithful_concise",
    "direct_workplace_message",
    "compression_update",
    "technical_explain_natural",
    "sensitive_comms",
]


class TaskConstraints(BaseModel):
    """Deterministic constraints attached to one RL task.

    v01 fields kept for backward compatibility. v03 fields added per
    docs/plans/v03-rl-tasks-dataset.md §2.3.
    """

    # ---- v01 length / format ----
    max_words: int | None = Field(default=None, ge=1)
    min_words: int | None = Field(default=None, ge=1)
    exact_sentences: int | None = Field(default=None, ge=1)
    return_only_answer: bool = True
    no_subject_line: bool = True
    no_signoff: bool = True
    preserve_numbers: bool = False
    preserve_entities: bool = False
    allow_placeholders: bool = False
    require_placeholders_for_missing_specifics: bool = False
    allow_markdown: bool = False

    # ---- v03 length ----
    target_words: int | None = Field(default=None, ge=1)
    target_tolerance: float | None = Field(default=None, ge=0.0, le=1.0)

    # ---- v03 structure ----
    min_sentences: int | None = Field(default=None, ge=1)
    max_sentences: int | None = Field(default=None, ge=1)
    min_paragraphs: int | None = Field(default=None, ge=1)
    max_paragraphs: int | None = Field(default=None, ge=1)
    required_section_headings: list[str] = Field(default_factory=list)
    allow_bullets: bool = True
    allow_headings: bool = True

    # ---- v03 inclusion / exclusion ----
    must_include_phrases: list[str] = Field(default_factory=list)
    must_not_include_phrases: list[str] = Field(default_factory=list)
    forbidden_openers: list[str] = Field(default_factory=list)
    forbidden_phrases_global: list[str] = Field(default_factory=list)

    # ---- v03 format ----
    no_markdown: bool = False
    must_not_use_em_dash: bool = False

    # ---- v03 register ----
    register_target: RegisterTarget | None = None
    max_passive_voice_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    min_contraction_count: int | None = Field(default=None, ge=0)

    # ---- v03 persona / audience (ridge-scored only) ----
    persona_constraint: str | None = None
    audience_constraint: str | None = None

    # ---- v03 facts ----
    must_contain_number_kind: NumberKind | None = None

    @model_validator(mode="after")
    def _validate_word_window(self) -> TaskConstraints:
        if (
            self.min_words is not None
            and self.max_words is not None
            and self.min_words > self.max_words
        ):
            raise ValueError("min_words must be <= max_words")
        if (
            self.min_sentences is not None
            and self.max_sentences is not None
            and self.min_sentences > self.max_sentences
        ):
            raise ValueError("min_sentences must be <= max_sentences")
        if (
            self.min_paragraphs is not None
            and self.max_paragraphs is not None
            and self.min_paragraphs > self.max_paragraphs
        ):
            raise ValueError("min_paragraphs must be <= max_paragraphs")
        # Plan §2.3: hard cap of 3 must_include_phrases per task.
        if len(self.must_include_phrases) > 3:
            raise ValueError(
                "must_include_phrases capped at 3 per task (plan §2.3)"
            )
        return self


class RLTask(BaseModel):
    """One stateless single-turn RL environment task."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(..., pattern=r"^rl_v0[13]_[0-9]{6}$")
    family: TaskFamily
    domain: TaskDomain
    mode: TaskMode
    register_: TaskRegister = Field(alias="register", serialization_alias="register")
    instruction: str = Field(..., min_length=1)
    input_text: str = ""
    constraints: TaskConstraints
    reward_profile: RewardProfileName
    trap_tags: list[str] = Field(default_factory=list, min_length=1)
    source: str = "synthetic_template_v01"
    split: TaskSplit
    forbidden_phrases: list[str] = Field(default_factory=list)
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)
    reference_response: str | None = None
    source_group: str = Field(default="synthetic_template_v01", min_length=1)
    license: str = "project_synthetic"
    release_eligible: bool = True
    # v03: track which model authored the instruction (plan §3.2).
    task_author_model: str | None = None

    @field_validator("instruction", "input_text")
    @classmethod
    def _strip_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def _validate_source_for_modes(self) -> RLTask:
        # Modes that REQUIRE input_text. long_form_generate and
        # multi_constraint_compose accept a short brief as input_text but do
        # not strictly require one (kept permissive here so future env change
        # can drop the brief; see plan §2.1 + Q3 decision).
        source_required = {
            "rewrite",
            "compress",
            "tone_shift",
            "repair",
            "rewrite_humanize",
            "compression",
            "expansion",
        }
        if self.mode in source_required and not self.input_text:
            raise ValueError(f"{self.mode} tasks require input_text")
        return self


def load_tasks(path: Path) -> list[RLTask]:
    """Load and validate RL tasks from JSONL."""
    rows: list[RLTask] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped:
            data = json.loads(stripped)
            try:
                rows.append(RLTask.model_validate(data))
            except ValueError as exc:
                raise ValueError(
                    f"Invalid RL task at {path}:{line_number}: {exc}"
                ) from exc
    return rows


def write_tasks(path: Path, tasks: list[RLTask]) -> None:
    """Write tasks as canonical JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        task.model_dump_json(exclude_none=True) + "\n"
        for task in sorted(tasks, key=lambda row: row.id)
    )
    path.write_text(payload)


def summarize_tasks(tasks: list[RLTask]) -> dict[str, dict[str, int]]:
    """Return counts needed for smoke dataset inspection."""
    trap_counts: Counter[str] = Counter()
    for task in tasks:
        trap_counts.update(task.trap_tags)

    return {
        "family": dict(Counter(task.family for task in tasks)),
        "domain": dict(Counter(task.domain for task in tasks)),
        "mode": dict(Counter(task.mode for task in tasks)),
        "register": dict(Counter(task.register_ for task in tasks)),
        "split": dict(Counter(task.split for task in tasks)),
        "trap": dict(trap_counts),
        "reward_profile": dict(Counter(task.reward_profile for task in tasks)),
    }
