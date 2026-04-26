"""Seed schema for v03 benchmark/data rebuild.

A `Seed` is one accepted human-written text plus its provenance and tags.
See docs/plans/v03-seed-benchmark-spec.md sections 5-6 for the rules.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Domain = Literal[
    "email",
    "instruction_technical",
    "blog_opinion",
    "academic",
    "creative",
]

DiscourseRole = Literal[
    "status_update",
    "request",
    "decision_rationale",
    "troubleshooting",
    "instructional_explanation",
    "argument_opinion",
    "anecdote",
    "reflection",
    "methods",
    "literature_review",
    "discussion_limitations",
    "reported_summary",
    "scene",
    "narrative_reflection",
]

LengthBand = Literal["short", "medium", "long"]


class Seed(BaseModel):
    """A single accepted human seed for the v03 pipeline.

    Schema is intentionally flat for easy JSONL round-tripping and arka
    `data_source: seeds` ingestion.
    """

    id: str = Field(..., description="Stable ID, e.g. 'v03_ws_inst_000'.")
    text: str = Field(..., min_length=1)
    domain: Domain
    discourse_role: DiscourseRole
    source_dataset: str = Field(
        ...,
        description="Origin corpus, e.g. 'goodwiki', 'enron', 'curated_paste'.",
    )
    length_band: LengthBand
    word_count: int = Field(..., ge=1)
    anchors_count: int = Field(
        ...,
        ge=0,
        description="Concrete anchors: names, dates, numbers, versions, "
        "commands, citations, quoted phrases.",
    )
    instruction: str = Field(
        default="",
        description="Optional prompt-style instruction. Empty for raw corpus rows.",
    )
    notes: str = Field(default="")

    @field_validator("text")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()


def to_arka_seed_row(seed: Seed) -> dict[str, object]:
    """Convert a `Seed` into the JSONL row shape arka's `seeds` source expects.

    Arka reads `instruction` + `response` as the conversation payload. We keep
    the original seed metadata under top-level keys so downstream stages can
    surface them via `system` / `transform_original`.
    """
    return {
        "instruction": seed.instruction or f"Write in the style of: {seed.domain}.",
        "response": seed.text,
        # Carry-along metadata. Arka ignores extra keys.
        "id": seed.id,
        "domain": seed.domain,
        "discourse_role": seed.discourse_role,
        "source_dataset": seed.source_dataset,
        "length_band": seed.length_band,
        "word_count": seed.word_count,
        "anchors_count": seed.anchors_count,
    }
