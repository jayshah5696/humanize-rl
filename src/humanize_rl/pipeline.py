"""Pipeline orchestrator — score pairs and triples, build benchmark dataset.

Handles two flows:
1. AIify pairs: original + AIified → scored pairs
2. Full triples: original + AIified + humanized → 3-class benchmark
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from humanize_rl.scoring.aggregator import HumannessResult, score_text


@dataclass
class ScoredPair:
    """A scored original-vs-AIified pair."""

    id: str
    instruction: str
    domain: str
    original_text: str
    aiified_text: str
    original_score: HumannessResult
    aiified_score: HumannessResult

    @property
    def delta(self) -> float:
        """Score delta: original - aiified. Positive = AIify worked."""
        return self.original_score.overall - self.aiified_score.overall


@dataclass
class ScoredTriple:
    """A scored original + AIified + humanized triple."""

    id: str
    instruction: str
    original_text: str
    aiified_text: str
    humanized_text: str
    original_score: HumannessResult
    aiified_score: HumannessResult
    humanized_score: HumannessResult

    @property
    def aiify_delta(self) -> float:
        return self.original_score.overall - self.aiified_score.overall

    @property
    def humanize_delta(self) -> float:
        return self.humanized_score.overall - self.aiified_score.overall

    @property
    def recovery_ratio(self) -> float:
        """How much of the original score did humanization recover?"""
        if self.aiify_delta == 0:
            return 0.0
        return self.humanize_delta / self.aiify_delta


def load_aiify_output(path: Path) -> list[dict]:
    """Load Arka transform output and extract original + AI-ified pairs."""
    pairs = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            aiified_text = record.get("response", "")
            original_text = ""
            system = record.get("system", "")
            if system:
                try:
                    meta = json.loads(system)
                    original_text = (
                        meta.get("transform_original", {}).get("text", "")
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass
            if not original_text or not aiified_text:
                continue
            pairs.append({
                "id": f"pair_{i:03d}",
                "instruction": record.get("instruction", ""),
                "original": original_text,
                "aiified": aiified_text,
            })
    return pairs


def load_humanize_output(path: Path) -> list[dict]:
    """Load Arka humanize output, extracting AIified input from system field."""
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            humanized_text = record.get("response", "")
            aiified_text = ""
            system = record.get("system", "")
            if system:
                try:
                    meta = json.loads(system)
                    aiified_text = (
                        meta.get("transform_original", {}).get("text", "")
                    )
                except (json.JSONDecodeError, AttributeError):
                    pass
            if not aiified_text or not humanized_text:
                continue
            records.append({
                "id": f"triple_{i:03d}",
                "instruction": record.get("instruction", ""),
                "aiified": aiified_text,
                "humanized": humanized_text,
            })
    return records


def build_triples(
    aiify_path: Path,
    humanize_path: Path,
) -> list[dict]:
    """Join AIify and humanize outputs into triples by index."""
    aiify_data = load_aiify_output(aiify_path)
    humanize_data = load_humanize_output(humanize_path)

    n = min(len(aiify_data), len(humanize_data))
    triples = []
    for i in range(n):
        triples.append({
            "id": f"triple_{i:03d}",
            "instruction": aiify_data[i]["instruction"],
            "original": aiify_data[i]["original"],
            "aiified": aiify_data[i]["aiified"],
            "humanized": humanize_data[i]["humanized"],
        })
    return triples


def score_pairs(pairs: list[dict]) -> list[ScoredPair]:
    """Score both original and AI-ified text for each pair."""
    scored = []
    for pair in pairs:
        original_result = score_text(pair["original"])
        aiified_result = score_text(pair["aiified"])
        scored.append(ScoredPair(
            id=pair["id"],
            instruction=pair["instruction"],
            domain="",
            original_text=pair["original"],
            aiified_text=pair["aiified"],
            original_score=original_result,
            aiified_score=aiified_result,
        ))
    return scored


def score_triples(triples: list[dict]) -> list[ScoredTriple]:
    """Score all 3 versions of each text."""
    scored = []
    for triple in triples:
        original_result = score_text(triple["original"])
        aiified_result = score_text(triple["aiified"])
        humanized_result = score_text(triple["humanized"])
        scored.append(ScoredTriple(
            id=triple["id"],
            instruction=triple["instruction"],
            original_text=triple["original"],
            aiified_text=triple["aiified"],
            humanized_text=triple["humanized"],
            original_score=original_result,
            aiified_score=aiified_result,
            humanized_score=humanized_result,
        ))
    return scored


def export_3class_benchmark(
    scored_triples: list[ScoredTriple],
    output_path: Path,
) -> None:
    """Export scored triples as a 3-class benchmark JSONL."""
    with open(output_path, "w") as f:
        for triple in scored_triples:
            for label, text, result in [
                ("human", triple.original_text, triple.original_score),
                ("ai", triple.aiified_text, triple.aiified_score),
                ("humanized", triple.humanized_text, triple.humanized_score),
            ]:
                record = {
                    "id": f"{triple.id}_{label}",
                    "label": label,
                    "overall_score": round(result.overall, 4),
                    "per_dim": {
                        k: round(v, 4) for k, v in result.per_dim.items()
                    },
                    "text_preview": text[:120],
                }
                f.write(json.dumps(record) + "\n")


def export_sft_pairs(
    scored_triples: list[ScoredTriple],
    output_path: Path,
    min_delta: float = 0.15,
) -> int:
    """Export SFT-ready pairs: (AIified input, humanized output).

    Only pairs where humanization improved the score by min_delta.
    Returns count of exported pairs.
    """
    exported = 0
    with open(output_path, "w") as f:
        for triple in scored_triples:
            if triple.humanize_delta < min_delta:
                continue
            record = {
                "instruction": (
                    "Rewrite the following text to sound natural and "
                    "human-written. Remove AI writing patterns while "
                    "preserving the meaning."
                ),
                "input": triple.aiified_text,
                "output": triple.humanized_text,
                "metadata": {
                    "id": triple.id,
                    "original_instruction": triple.instruction,
                    "aiified_score": round(
                        triple.aiified_score.overall, 4
                    ),
                    "humanized_score": round(
                        triple.humanized_score.overall, 4
                    ),
                    "delta": round(triple.humanize_delta, 4),
                    "recovery_ratio": round(triple.recovery_ratio, 4),
                },
            }
            f.write(json.dumps(record) + "\n")
            exported += 1
    return exported


def print_pair_report(scored_pairs: list[ScoredPair]) -> None:
    """Print a summary of scored pairs."""
    if not scored_pairs:
        print("No pairs to report.")
        return
    original_scores = [p.original_score.overall for p in scored_pairs]
    aiified_scores = [p.aiified_score.overall for p in scored_pairs]
    deltas = [p.delta for p in scored_pairs]
    print("=" * 60)
    print("SCORED PAIRS REPORT")
    print("=" * 60)
    print(f"  Pairs:          {len(scored_pairs)}")
    print(f"  Original mean:  {sum(original_scores)/len(original_scores):.3f}")
    print(f"  AI-ified mean:  {sum(aiified_scores)/len(aiified_scores):.3f}")
    print(f"  Mean delta:     {sum(deltas)/len(deltas):.3f}")
    print(f"  Min delta:      {min(deltas):.3f}")
    print(f"  Max delta:      {max(deltas):.3f}")
    effective = sum(1 for d in deltas if d > 0.1)
    print(f"  Effective AIify: {effective}/{len(scored_pairs)} "
          f"({effective/len(scored_pairs)*100:.0f}%)")
    print("=" * 60)


def print_triple_report(scored_triples: list[ScoredTriple]) -> None:
    """Print a summary of scored triples."""
    if not scored_triples:
        print("No triples to report.")
        return
    orig = [t.original_score.overall for t in scored_triples]
    ai = [t.aiified_score.overall for t in scored_triples]
    hum = [t.humanized_score.overall for t in scored_triples]
    recovery = [t.recovery_ratio for t in scored_triples]

    print("=" * 60)
    print("3-CLASS SCORING REPORT")
    print("=" * 60)
    print(f"  Triples:             {len(scored_triples)}")
    print(f"  Original mean:       {sum(orig)/len(orig):.3f}")
    print(f"  AI-ified mean:       {sum(ai)/len(ai):.3f}")
    print(f"  Humanized mean:      {sum(hum)/len(hum):.3f}")
    print(f"  Mean recovery ratio: {sum(recovery)/len(recovery):.1%}")
    sft_ready = sum(
        1 for t in scored_triples if t.humanize_delta > 0.15
    )
    print(f"  SFT-ready pairs:     {sft_ready}/{len(scored_triples)} "
          f"(delta > 0.15)")
    print("=" * 60)
