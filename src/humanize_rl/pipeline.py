"""Pipeline orchestrator — score AIified outputs and build benchmark dataset.

Takes Arka's AIify output, extracts original + transformed pairs,
scores both with Layer 1, and builds a scored benchmark dataset.
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

            # Extract original from system field (Arka preserve_original)
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

            pairs.append(
                {
                    "id": f"pair_{i:03d}",
                    "instruction": record.get("instruction", ""),
                    "original": original_text,
                    "aiified": aiified_text,
                }
            )
    return pairs


def score_pairs(pairs: list[dict]) -> list[ScoredPair]:
    """Score both original and AI-ified text for each pair."""
    scored = []
    for pair in pairs:
        original_result = score_text(pair["original"])
        aiified_result = score_text(pair["aiified"])

        scored.append(
            ScoredPair(
                id=pair["id"],
                instruction=pair["instruction"],
                domain="",  # TODO: extract from seeds
                original_text=pair["original"],
                aiified_text=pair["aiified"],
                original_score=original_result,
                aiified_score=aiified_result,
            )
        )
    return scored


def export_benchmark_from_pairs(
    scored_pairs: list[ScoredPair],
    output_path: Path,
) -> None:
    """Export scored pairs as a benchmark JSONL (human + AI interleaved)."""
    with open(output_path, "w") as f:
        for pair in scored_pairs:
            # Human sample
            human_record = {
                "id": f"{pair.id}_human",
                "label": "human",
                "source": "curated",
                "overall_score": round(pair.original_score.overall, 4),
                "per_dim": {
                    k: round(v, 4)
                    for k, v in pair.original_score.per_dim.items()
                },
                "text_preview": pair.original_text[:120],
            }
            f.write(json.dumps(human_record) + "\n")

            # AI sample
            ai_record = {
                "id": f"{pair.id}_ai",
                "label": "ai",
                "source": "gemini-3.1-flash-lite",
                "overall_score": round(pair.aiified_score.overall, 4),
                "per_dim": {
                    k: round(v, 4)
                    for k, v in pair.aiified_score.per_dim.items()
                },
                "text_preview": pair.aiified_text[:120],
            }
            f.write(json.dumps(ai_record) + "\n")


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
    print(
        f"  Original mean:  {sum(original_scores)/len(original_scores):.3f}"
    )
    print(
        f"  AI-ified mean:  {sum(aiified_scores)/len(aiified_scores):.3f}"
    )
    print(f"  Mean delta:     {sum(deltas)/len(deltas):.3f}")
    print(f"  Min delta:      {min(deltas):.3f}")
    print(f"  Max delta:      {max(deltas):.3f}")

    # Count how many pairs have positive delta (AIify actually worked)
    effective = sum(1 for d in deltas if d > 0.1)
    print(f"  Effective AIify: {effective}/{len(scored_pairs)} "
          f"({effective/len(scored_pairs)*100:.0f}%)")

    # Show worst pair (smallest delta)
    worst = min(scored_pairs, key=lambda p: p.delta)
    print(f"\n  Worst pair ({worst.id}): delta={worst.delta:.3f}")
    print(f"    Original: {worst.original_text[:80]}...")
    print(f"    AI-ified: {worst.aiified_text[:80]}...")
    print("=" * 60)
