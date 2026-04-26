"""Tests for V-Slice 4 report builder."""

from __future__ import annotations

import json
from pathlib import Path

from humanize_rl.data.report_v03 import (
    _auroc,
    _length_band,
    _per_class_means,
    build_core_split,
    build_diagnostics_split,
    build_ood_ai_split,
)


def test_length_band_thresholds() -> None:
    assert _length_band(50) == "short"
    assert _length_band(100) == "medium"
    assert _length_band(220) == "long"


def test_auroc_perfect_separation() -> None:
    assert _auroc([0.9, 0.8, 0.7], [0.3, 0.2, 0.1]) == 1.0


def test_auroc_inverted() -> None:
    assert _auroc([0.1, 0.2, 0.3], [0.7, 0.8, 0.9]) == 0.0


def test_auroc_chance() -> None:
    score = _auroc([0.5, 0.6], [0.5, 0.6])
    assert 0.4 <= score <= 0.6


def test_per_class_means_handles_missing_scores() -> None:
    rows = [
        {"label": "human", "overall_score": 0.8},
        {"label": "human", "overall_score": 0.9},
        {"label": "ai", "overall_score": None},
        {"label": "ai", "overall_score": 0.3},
    ]
    means = _per_class_means(rows)
    assert means["human"] == 0.8500000000000001 or abs(means["human"] - 0.85) < 1e-9
    assert means["ai"] == 0.3


def test_build_core_split_writes_expected_shape(tmp_path: Path) -> None:
    matched = tmp_path / "matched.jsonl"
    matched.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "triple_000_human",
                        "label": "human",
                        "domain": "email",
                        "text": "x",
                        "overall_score": 0.8,
                        "per_dim": {"opener_pattern": 1.0},
                    }
                ),
            ]
        )
        + "\n"
    )
    out_path = tmp_path / "core.jsonl"
    rows = build_core_split(matched, out_path)
    assert len(rows) == 1
    assert rows[0]["split"] == "v03_core"
    assert out_path.exists()


def test_build_ood_ai_keeps_only_ai_class(tmp_path: Path) -> None:
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "a", "label": "ai", "text": "x", "domain": "blog"},
                {"id": "b", "label": "human", "text": "y", "domain": "blog"},
                {"id": "c", "label": "humanized", "text": "z", "domain": "blog"},
                {"id": "d", "label": "ai", "text": "w", "domain": "essay"},
            ]
        )
        + "\n"
    )
    out_path = tmp_path / "ood.jsonl"
    rows = build_ood_ai_split(legacy, out_path)
    assert len(rows) == 2
    assert {r["label"] for r in rows} == {"ai"}


def test_build_diagnostics_excludes_accepted_triples(tmp_path: Path) -> None:
    matched = tmp_path / "matched.jsonl"
    matched.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "triple_000_human", "label": "human", "domain": "x", "text": ""},
                {"id": "triple_000_ai", "label": "ai", "domain": "x", "text": ""},
                {"id": "triple_001_human", "label": "human", "domain": "y", "text": ""},
                {"id": "triple_001_ai", "label": "ai", "domain": "y", "text": ""},
            ]
        )
        + "\n"
    )
    sft = tmp_path / "sft.jsonl"
    sft.write_text(json.dumps({"id": "triple_000"}) + "\n")
    out_path = tmp_path / "diag.jsonl"
    rows = build_diagnostics_split(matched, sft, out_path)
    triple_ids = {r["triple_id"] for r in rows}
    assert "triple_000" not in triple_ids
    assert "triple_001" in triple_ids
