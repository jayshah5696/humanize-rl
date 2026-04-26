"""Tests for the v03 Seed schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from humanize_rl.data.seed import Seed, to_arka_seed_row


def _valid_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "id": "v03_ws_inst_000",
        "text": "  This is a sample paragraph with two anchors: Python 3.12 and pytest.  ",
        "domain": "instruction_technical",
        "discourse_role": "instructional_explanation",
        "source_dataset": "curated_paste",
        "length_band": "medium",
        "word_count": 12,
        "anchors_count": 2,
    }
    base.update(overrides)
    return base


def test_seed_strips_text() -> None:
    seed = Seed(**_valid_kwargs())
    assert seed.text.startswith("This is a sample")
    assert not seed.text.endswith(" ")


def test_seed_rejects_unknown_domain() -> None:
    with pytest.raises(ValidationError):
        Seed(**_valid_kwargs(domain="random_bucket"))


def test_seed_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        Seed(**_valid_kwargs(discourse_role="hot_take"))


def test_seed_rejects_negative_anchors() -> None:
    with pytest.raises(ValidationError):
        Seed(**_valid_kwargs(anchors_count=-1))


def test_to_arka_seed_row_round_trip() -> None:
    seed = Seed(**_valid_kwargs())
    row = to_arka_seed_row(seed)
    assert row["response"] == seed.text
    assert row["domain"] == seed.domain
    assert row["id"] == seed.id
    assert row["discourse_role"] == seed.discourse_role
    assert isinstance(row["instruction"], str) and row["instruction"]
