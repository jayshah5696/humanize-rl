"""Score all 3 classes with both Layer 1 and Layer 2, then benchmark.

End-to-end: loads triples, scores with L1, gates, scores with L2,
combines, computes AUROC, exports.

Usage:
    uv run python -m humanize_rl.score_all
    uv run python -m humanize_rl.score_all --model google/gemini-3-flash-preview  # cheaper
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from humanize_rl.benchmark.evaluator import _auroc
from humanize_rl.pipeline import build_triples
from humanize_rl.scoring.aggregator import CombinedResult, combine_scores, score_text
from humanize_rl.scoring.gate import needs_layer2
from humanize_rl.scoring.layer2 import score_layer2_batch


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score all data with Layer 1 + Layer 2 and benchmark"
    )
    parser.add_argument(
        "--aiify-output",
        type=Path,
        default=Path("output/01-aiify-dataset.jsonl"),
    )
    parser.add_argument(
        "--humanize-output",
        type=Path,
        default=Path("output/02-humanize-dataset.jsonl"),
    )
    parser.add_argument(
        "--model",
        default="google/gemini-3.1-pro-preview",
        help="LLM judge model for Layer 2",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/scored_combined_v01.jsonl"),
    )
    parser.add_argument(
        "--skip-gate",
        action="store_true",
        help="Score everything with L2, skip gate logic",
    )
    args = parser.parse_args()

    if not args.aiify_output.exists() or not args.humanize_output.exists():
        print("Error: need both AIify and humanize outputs.")
        print("  just aiify && just humanize")
        return

    # 1. Build triples
    print("Loading triples...")
    triples = build_triples(args.aiify_output, args.humanize_output)
    print(f"  {len(triples)} triples loaded")

    # 2. Score everything with Layer 1
    print("\nScoring with Layer 1 (deterministic)...")
    all_texts: list[str] = []
    all_labels: list[str] = []
    all_contexts: list[str] = []
    all_l1 = []

    for triple in triples:
        for label, text, ctx in [
            ("human", triple["original"], "source_scoring"),
            ("ai", triple["aiified"], "aiified_scoring"),
            ("humanized", triple["humanized"], "humanized_scoring"),
        ]:
            l1 = score_text(text)
            all_texts.append(text)
            all_labels.append(label)
            all_contexts.append(ctx)
            all_l1.append(l1)

    n = len(all_texts)
    print(f"  Scored {n} texts with Layer 1")

    # 3. Gate: decide which need Layer 2
    if args.skip_gate:
        needs_l2_mask = [True] * n
    else:
        needs_l2_mask = [
            needs_layer2(l1.overall, ctx)
            for l1, ctx in zip(all_l1, all_contexts, strict=True)
        ]
    l2_count = sum(needs_l2_mask)
    skipped = n - l2_count
    print(f"\n  Gate: {l2_count} need Layer 2, {skipped} skipped "
          f"({skipped/n*100:.0f}% saved)")

    # 4. Score with Layer 2 (LLM judge)
    l2_texts = [t for t, m in zip(all_texts, needs_l2_mask, strict=True) if m]
    l2_instructions = [
        "Evaluate this text for AI writing patterns."
    ] * len(l2_texts)

    print(f"\nScoring {len(l2_texts)} texts with Layer 2 ({args.model})...")
    l2_results = score_layer2_batch(
        texts=l2_texts,
        instructions=l2_instructions,
        model=args.model,
        max_workers=args.max_workers,
    )
    print("  Layer 2 scoring complete")

    # 5. Combine L1 + L2
    print("\nCombining Layer 1 + Layer 2 scores...")
    combined_results: list[CombinedResult | None] = []
    l2_idx = 0
    for i in range(n):
        if needs_l2_mask[i]:
            l2r = l2_results[l2_idx]
            l2_idx += 1
            combined = combine_scores(
                layer1=all_l1[i],
                layer2_overall=l2r.overall,
                layer2_per_dim=l2r.per_dim,
                layer2_raw=l2r.raw_scores,
                layer2_reasoning=l2r.reasoning,
            )
            combined_results.append(combined)
        else:
            # L2 skipped — use L1 only
            combined_results.append(None)

    # 6. Compute final scores (combined where available, L1-only otherwise)
    final_scores = []
    for i in range(n):
        if combined_results[i] is not None:
            final_scores.append(combined_results[i].overall)
        else:
            final_scores.append(all_l1[i].overall)

    # 7. Print report
    _print_report(all_labels, all_l1, combined_results, final_scores, n)

    # 8. Export
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _export(
        args.output, all_texts, all_labels, all_l1, combined_results,
        final_scores,
    )
    print(f"\nExported to {args.output}")


def _print_report(
    labels: list[str],
    l1_results: list,
    combined: list,
    final_scores: list[float],
    n: int,
) -> None:
    """Print benchmark report with L1, L2, combined AUROC."""
    # Group by label
    by_label: dict[str, list[float]] = {"human": [], "ai": [], "humanized": []}
    l1_by_label: dict[str, list[float]] = {"human": [], "ai": [], "humanized": []}
    l2_by_label: dict[str, list[float]] = {"human": [], "ai": [], "humanized": []}

    for i in range(n):
        lab = labels[i]
        by_label[lab].append(final_scores[i])
        l1_by_label[lab].append(l1_results[i].overall)
        if combined[i] is not None:
            l2_by_label[lab].append(combined[i].layer2_overall)

    print("\n" + "=" * 65)
    print("COMBINED L1+L2 BENCHMARK REPORT")
    print("=" * 65)

    for lab in ["human", "ai", "humanized"]:
        l1_vals = l1_by_label[lab]
        l2_vals = l2_by_label[lab]
        final_vals = by_label[lab]
        l1_mean = sum(l1_vals) / len(l1_vals) if l1_vals else 0
        l2_mean = sum(l2_vals) / len(l2_vals) if l2_vals else 0
        final_mean = sum(final_vals) / len(final_vals) if final_vals else 0
        l2_info = f"L2={l2_mean:.3f}" if l2_vals else "L2=skipped"
        print(
            f"  {lab:<12s}  L1={l1_mean:.3f}  {l2_info}  "
            f"Combined={final_mean:.3f}  (n={len(final_vals)})"
        )

    # AUROC computations
    def auroc_pair(label_a: str, label_b: str, label_name: str) -> float:
        scores = by_label[label_a] + by_label[label_b]
        binary = [1] * len(by_label[label_a]) + [0] * len(by_label[label_b])
        auc = _auroc(scores, binary)
        return auc

    print()
    for a, b, name in [
        ("human", "ai", "human vs AI"),
        ("humanized", "ai", "humanized vs AI"),
        ("human", "humanized", "human vs humanized"),
    ]:
        auc_l1 = _auroc(
            l1_by_label[a] + l1_by_label[b],
            [1] * len(l1_by_label[a]) + [0] * len(l1_by_label[b]),
        )
        auc_combined = auroc_pair(a, b, name)
        print(f"  AUROC {name:<25s}  L1={auc_l1:.4f}  Combined={auc_combined:.4f}")

    print("=" * 65)


def _export(
    path: Path,
    texts: list[str],
    labels: list[str],
    l1_results: list,
    combined: list,
    final_scores: list[float],
) -> None:
    """Export all scored records."""
    with open(path, "w") as f:
        for i in range(len(texts)):
            record: dict = {
                "label": labels[i],
                "l1_overall": round(l1_results[i].overall, 4),
                "l1_per_dim": {
                    k: round(v, 4) for k, v in l1_results[i].per_dim.items()
                },
            }
            if combined[i] is not None:
                record["l2_overall"] = round(combined[i].layer2_overall, 4)
                record["l2_per_dim"] = {
                    k: round(v, 4)
                    for k, v in combined[i].layer2_per_dim.items()
                }
                record["l2_raw"] = combined[i].layer2_raw
                record["l2_reasoning"] = combined[i].layer2_reasoning
            record["combined_overall"] = round(final_scores[i], 4)
            record["text_preview"] = texts[i][:150]
            f.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
