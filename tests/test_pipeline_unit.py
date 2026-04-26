"""Unit tests for pipeline export and loading helpers."""

from __future__ import annotations

import json
from pathlib import Path

from humanize_rl.pipeline import (
    ScoredTriple,
    export_3class_benchmark,
    export_sft_pairs,
    load_aiify_output,
    load_humanize_output,
)
from humanize_rl.scoring.aggregator import HumannessResult


def _result(overall: float, **per_dim: float) -> HumannessResult:
    return HumannessResult(
        overall=overall,
        per_dim=per_dim or {"opener_pattern": overall},
    )


def test_load_aiify_output_skips_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "aiify.jsonl"
    valid_meta = json.dumps({"transform_original": {"text": "Human source"}})
    broken_meta = "not-json"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "inst-1",
                        "response": "AI rewrite",
                        "system": valid_meta,
                    }
                ),
                json.dumps(
                    {
                        "instruction": "inst-2",
                        "response": "",
                        "system": valid_meta,
                    }
                ),
                json.dumps(
                    {
                        "instruction": "inst-3",
                        "response": "AI rewrite",
                        "system": broken_meta,
                    }
                ),
            ]
        )
    )

    pairs = load_aiify_output(path)

    assert pairs == [
        {
            "id": "pair_000",
            "instruction": "inst-1",
            "original": "Human source",
            "aiified": "AI rewrite",
        }
    ]


def test_load_humanize_output_skips_invalid_records(tmp_path: Path) -> None:
    path = tmp_path / "humanize.jsonl"
    valid_meta = json.dumps({"transform_original": {"text": "AI source"}})
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "instruction": "inst-1",
                        "response": "Humanized rewrite",
                        "system": valid_meta,
                    }
                ),
                json.dumps(
                    {
                        "instruction": "inst-2",
                        "response": "Humanized rewrite",
                        "system": "bad-json",
                    }
                ),
                json.dumps(
                    {
                        "instruction": "inst-3",
                        "response": "",
                        "system": valid_meta,
                    }
                ),
            ]
        )
    )

    triples = load_humanize_output(path)

    assert triples == [
        {
            "id": "triple_000",
            "instruction": "inst-1",
            "aiified": "AI source",
            "humanized": "Humanized rewrite",
        }
    ]


def test_export_3class_benchmark_writes_three_rows_per_triple(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "benchmark.jsonl"
    triples = [
        ScoredTriple(
            id="triple_001",
            instruction="rewrite",
            original_text="Original text",
            aiified_text="AI text",
            humanized_text="Humanized text",
            original_score=_result(0.9, opener_pattern=1.0),
            aiified_score=_result(0.2, opener_pattern=0.0),
            humanized_score=_result(0.8, opener_pattern=0.9),
        )
    ]

    export_3class_benchmark(triples, output_path)

    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 3
    records = [json.loads(line) for line in lines]
    assert [record["label"] for record in records] == [
        "human",
        "ai",
        "humanized",
    ]
    assert records[0]["id"] == "triple_001_human"
    assert records[1]["overall_score"] == 0.2
    assert records[2]["per_dim"] == {"opener_pattern": 0.9}
    assert records[2]["text_preview"] == "Humanized text"


def test_export_sft_pairs_filters_by_delta_and_returns_count(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "sft.jsonl"
    triples = [
        ScoredTriple(
            id="keep-me",
            instruction="original instruction",
            original_text="Original",
            aiified_text="AI draft",
            humanized_text="Human rewrite",
            original_score=_result(0.8),
            aiified_score=_result(0.2),
            humanized_score=_result(0.5),
        ),
        ScoredTriple(
            id="drop-me",
            instruction="other instruction",
            original_text="Original 2",
            aiified_text="AI draft 2",
            humanized_text="Human rewrite 2",
            original_score=_result(0.7),
            aiified_score=_result(0.3),
            humanized_score=_result(0.4),
        ),
    ]

    exported = export_sft_pairs(triples, output_path, min_delta=0.15)

    assert exported == 1
    lines = output_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["input"] == "AI draft"
    assert record["output"] == "Human rewrite"
    assert record["metadata"]["id"] == "keep-me"
    assert record["metadata"]["original_instruction"] == "original instruction"
    assert record["metadata"]["delta"] == 0.3
    assert record["metadata"]["recovery_ratio"] == 0.5
