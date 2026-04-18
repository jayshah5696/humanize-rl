"""Benchmark dataset loading and normalization helpers."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from humanize_rl.pipeline import load_aiify_output, load_humanize_output

_AI_GENERATOR = "google/gemini-3.1-flash-lite-preview"
_HUMANIZED_GENERATOR = "google/gemini-3.1-pro-preview"
_TRIPLE_ID_RE = re.compile(r"^triple_(\d+)_([a-z]+)$")
_DOMAIN_NAMES = {"blog", "email", "essay", "forum", "social", "technical", "academic"}


@dataclass(frozen=True)
class BenchmarkSample:
    """Normalized benchmark sample."""

    id: str
    label: str
    text: str
    source: str
    domain: str
    generator: str
    instruction: str = ""
    text_is_preview: bool = False
    overall_score: float | None = None
    per_dim: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkDataset:
    """Collection of normalized benchmark samples."""

    samples: list[BenchmarkSample]
    label_counts: dict[str, int] = field(default_factory=dict)
    domain_counts: dict[str, int] = field(default_factory=dict)
    preview_only_count: int = 0

    def filter_by_label(self, label: str) -> BenchmarkDataset:
        """Return a dataset containing only the requested label."""
        filtered = [sample for sample in self.samples if sample.label == label]
        return _build_dataset(filtered)


def _count(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _build_dataset(samples: list[BenchmarkSample]) -> BenchmarkDataset:
    return BenchmarkDataset(
        samples=samples,
        label_counts=_count([sample.label for sample in samples]),
        domain_counts=_count([sample.domain for sample in samples]),
        preview_only_count=sum(sample.text_is_preview for sample in samples),
    )


def _infer_domain(domain: str | None, source: str) -> str:
    if domain:
        return domain
    if source in _DOMAIN_NAMES:
        return source
    return "unknown"


def load_benchmark_dataset(path: Path) -> BenchmarkDataset:
    """Load a normalized 3-class benchmark dataset from JSONL."""
    samples: list[BenchmarkSample] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            has_full_text = bool(record.get("text"))
            text = record.get("text") or record.get("text_preview", "")
            source = record.get("source", "unknown")
            domain = _infer_domain(record.get("domain"), source)
            samples.append(
                BenchmarkSample(
                    id=record["id"],
                    label=record["label"],
                    text=text,
                    source=source,
                    domain=domain,
                    generator=record.get("generator", "unknown"),
                    instruction=record.get("instruction", ""),
                    text_is_preview=record.get("text_is_preview", not has_full_text),
                    overall_score=record.get("overall_score"),
                    per_dim=record.get("per_dim", {}),
                )
            )
    return _build_dataset(samples)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _parse_triple_id(record_id: str) -> tuple[int, str]:
    match = _TRIPLE_ID_RE.match(record_id)
    if match is None:
        raise ValueError(f"Unrecognized benchmark record id: {record_id}")
    index_text, label = match.groups()
    return int(index_text), label


def build_mvp_benchmark_dataset(
    scored_path: Path,
    seeds_path: Path,
    output_path: Path,
    max_per_label: int | None = None,
    aiify_output_path: Path | None = None,
    humanize_output_path: Path | None = None,
) -> int:
    """Build a normalized benchmark dataset from current repo artifacts.

    Human examples are enriched with full seed text. AI and humanized rows fall
    back to the scored artifact's stored preview text because the full transform
    outputs are not committed to the repository.
    """
    seeds = _load_jsonl(seeds_path)
    scored_records = _load_jsonl(scored_path)
    aiify_full = (
        load_aiify_output(aiify_output_path)
        if aiify_output_path is not None and aiify_output_path.exists()
        else []
    )
    humanize_full = (
        load_humanize_output(humanize_output_path)
        if humanize_output_path is not None and humanize_output_path.exists()
        else []
    )

    written = 0
    per_label_written: dict[str, int] = {}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in scored_records:
            seed_index, label = _parse_triple_id(record["id"])
            if max_per_label is not None:
                current = per_label_written.get(label, 0)
                if current >= max_per_label:
                    continue

            seed = seeds[seed_index] if seed_index < len(seeds) else {}
            domain = seed.get("domain", "unknown")
            instruction = seed.get("instruction", "")

            normalized = {
                "id": record["id"],
                "label": label,
                "domain": domain,
                "instruction": instruction,
                "overall_score": record.get("overall_score"),
                "per_dim": record.get("per_dim", {}),
            }

            if label == "human":
                normalized.update(
                    {
                        "text": seed.get("response", record.get("text_preview", "")),
                        "text_is_preview": False,
                        "source": seed.get("source", "curated"),
                        "generator": "human",
                    }
                )
            elif label == "ai":
                ai_text = record.get("text_preview", "")
                ai_is_preview = True
                if seed_index < len(aiify_full):
                    ai_text = aiify_full[seed_index].get("aiified", ai_text)
                    ai_is_preview = False
                normalized.update(
                    {
                        "text": ai_text,
                        "text_is_preview": ai_is_preview,
                        "source": "aiify-v01",
                        "generator": _AI_GENERATOR,
                    }
                )
            elif label == "humanized":
                humanized_text = record.get("text_preview", "")
                humanized_is_preview = True
                if seed_index < len(humanize_full):
                    humanized_text = humanize_full[seed_index].get(
                        "humanized", humanized_text
                    )
                    humanized_is_preview = False
                normalized.update(
                    {
                        "text": humanized_text,
                        "text_is_preview": humanized_is_preview,
                        "source": "humanize-v04",
                        "generator": _HUMANIZED_GENERATOR,
                    }
                )
            else:
                normalized.update(
                    {
                        "text": record.get("text_preview", ""),
                        "text_is_preview": True,
                        "source": "unknown",
                        "generator": "unknown",
                    }
                )

            handle.write(json.dumps(normalized) + "\n")
            per_label_written[label] = per_label_written.get(label, 0) + 1
            written += 1

    return written


def build_repo_benchmark_dataset(
    base_dataset_path: Path,
    human_path: Path,
    ai_path: Path,
    scored_output_path: Path,
    output_path: Path,
) -> int:
    """Expand the benchmark with unique full-text human/AI samples from repo data."""
    base_records = _load_jsonl(base_dataset_path)
    human_records = _load_jsonl(human_path)
    ai_records = _load_jsonl(ai_path)
    scored_records = {
        record["id"]: record for record in _load_jsonl(scored_output_path)
    }

    seen_texts = {record.get("text", "") for record in base_records}
    combined = list(base_records)

    for record in human_records + ai_records:
        text = record.get("text", "")
        if not text or text in seen_texts:
            continue

        scored = scored_records.get(record["id"], {})
        combined.append(
            {
                "id": record["id"],
                "label": record["label"],
                "domain": _infer_domain(None, record.get("source", "unknown")),
                "instruction": "",
                "overall_score": scored.get("overall_score"),
                "per_dim": scored.get("per_dim", {}),
                "text": text,
                "text_is_preview": False,
                "source": record.get("source", "unknown"),
                "generator": "human" if record["label"] == "human" else record.get("source", "unknown"),
            }
        )
        seen_texts.add(text)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in combined:
            source = record.get("source", "unknown")
            record["domain"] = _infer_domain(record.get("domain"), source)
            handle.write(json.dumps(record) + "\n")

    return len(combined)


def format_dataset_summary(dataset: BenchmarkDataset) -> str:
    """Format a compact summary for CLI/reporting."""
    lines = [
        f"Samples: {len(dataset.samples)}",
        f"Labels: {dataset.label_counts}",
        f"Preview-only rows: {dataset.preview_only_count}",
        f"Domains: {dataset.domain_counts}",
    ]
    return "\n".join(lines)
