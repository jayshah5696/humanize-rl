"""Tests for benchmark dataset loading and artifact building."""

from __future__ import annotations

import json
from pathlib import Path

from humanize_rl.benchmark.datasets import (
    BenchmarkDataset,
    build_mvp_benchmark_dataset,
    load_benchmark_dataset,
)


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")


class TestLoadBenchmarkDataset:
    def test_counts_labels_and_domains(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        _write_jsonl(
            dataset_path,
            [
                {
                    "id": "h1",
                    "label": "human",
                    "source": "blog",
                    "text": "Human text",
                },
                {
                    "id": "a1",
                    "label": "ai",
                    "domain": "technical",
                    "source": "chatgpt",
                    "text_preview": "AI preview",
                },
                {
                    "id": "u1",
                    "label": "humanized",
                    "source": "pipeline",
                    "text": "Humanized text",
                },
            ],
        )

        dataset = load_benchmark_dataset(dataset_path)

        assert isinstance(dataset, BenchmarkDataset)
        assert dataset.label_counts == {"human": 1, "ai": 1, "humanized": 1}
        assert dataset.domain_counts == {
            "blog": 1,
            "technical": 1,
            "unknown": 1,
        }
        assert dataset.preview_only_count == 1
        assert dataset.samples[0].domain == "blog"
        assert dataset.samples[1].text == "AI preview"
        assert dataset.samples[1].text_is_preview is True

    def test_filters_by_label(self, tmp_path: Path) -> None:
        dataset_path = tmp_path / "dataset.jsonl"
        _write_jsonl(
            dataset_path,
            [
                {"id": "h1", "label": "human", "text": "h"},
                {"id": "a1", "label": "ai", "text": "a"},
                {"id": "u1", "label": "humanized", "text": "u"},
            ],
        )

        dataset = load_benchmark_dataset(dataset_path)
        ai_only = dataset.filter_by_label("ai")

        assert [sample.id for sample in ai_only.samples] == ["a1"]
        assert ai_only.label_counts == {"ai": 1}


class TestBuildMVPBenchmarkDataset:
    def test_builds_normalized_dataset_from_scored_and_seed_data(
        self,
        tmp_path: Path,
    ) -> None:
        seeds_path = tmp_path / "seeds.jsonl"
        scored_path = tmp_path / "scored.jsonl"
        output_path = tmp_path / "benchmark.jsonl"

        _write_jsonl(
            seeds_path,
            [
                {
                    "instruction": "inst 1",
                    "response": "Original full text 1",
                    "domain": "blog",
                    "source": "curated",
                },
                {
                    "instruction": "inst 2",
                    "response": "Original full text 2",
                    "domain": "email",
                    "source": "curated",
                },
            ],
        )
        _write_jsonl(
            scored_path,
            [
                {
                    "id": "triple_000_human",
                    "label": "human",
                    "overall_score": 0.9,
                    "per_dim": {"opener_pattern": 1.0},
                    "text_preview": "Truncated human preview",
                },
                {
                    "id": "triple_000_ai",
                    "label": "ai",
                    "overall_score": 0.2,
                    "per_dim": {"opener_pattern": 0.0},
                    "text_preview": "AI preview",
                },
                {
                    "id": "triple_001_humanized",
                    "label": "humanized",
                    "overall_score": 0.8,
                    "per_dim": {"opener_pattern": 1.0},
                    "text_preview": "Humanized preview",
                },
            ],
        )

        written = build_mvp_benchmark_dataset(
            scored_path=scored_path,
            seeds_path=seeds_path,
            output_path=output_path,
        )

        assert written == 3
        records = [json.loads(line) for line in output_path.read_text().splitlines()]

        human_record = records[0]
        assert human_record["text"] == "Original full text 1"
        assert human_record["text_is_preview"] is False
        assert human_record["domain"] == "blog"
        assert human_record["instruction"] == "inst 1"
        assert human_record["generator"] == "human"
        assert human_record["source"] == "curated"

        ai_record = records[1]
        assert ai_record["text"] == "AI preview"
        assert ai_record["text_is_preview"] is True
        assert ai_record["domain"] == "blog"
        assert ai_record["generator"] == "google/gemini-3.1-flash-lite-preview"
        assert ai_record["source"] == "aiify-v01"

        humanized_record = records[2]
        assert humanized_record["text"] == "Humanized preview"
        assert humanized_record["text_is_preview"] is True
        assert humanized_record["domain"] == "email"
        assert humanized_record["generator"] == "google/gemini-3.1-pro-preview"
        assert humanized_record["source"] == "humanize-v04"
