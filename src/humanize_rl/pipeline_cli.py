"""CLI for the full scoring pipeline.

Usage:
    # Score AIify pairs only (Layer 1)
    uv run python -m humanize_rl.pipeline_cli

    # Full 3-class: original → AIified → humanized
    uv run python -m humanize_rl.pipeline_cli --humanize-output output/02-humanize-dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from humanize_rl.benchmark.evaluator import _auroc
from humanize_rl.pipeline import (
    build_triples,
    export_3class_benchmark,
    export_sft_pairs,
    load_aiify_output,
    print_pair_report,
    print_triple_report,
    score_pairs,
    score_triples,
)


def _print_auroc_table(
    scores: list[float],
    labels: list[int],
    dim_data: dict[str, tuple[list[float], list[int]]],
    title: str,
) -> None:
    """Print AUROC table for given scores and labels."""
    auroc = _auroc(scores, labels)
    print(f"\n  {title}: {auroc:.4f}")
    print(f"\n  Per-dimension AUROC ({title}):")
    dim_aurocs = {}
    for dim, (dim_scores, dim_labels) in dim_data.items():
        dim_aurocs[dim] = _auroc(dim_scores, dim_labels)
    for dim, auc in sorted(dim_aurocs.items(), key=lambda x: -x[1]):
        bar = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
        print(f"    {dim:<22s} {bar} {auc:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score pipeline output and compute benchmark metrics"
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
        "--benchmark-output",
        type=Path,
        default=Path("data/benchmark/scored_3class_v01.jsonl"),
    )
    parser.add_argument(
        "--sft-output",
        type=Path,
        default=Path("data/processed/sft_pairs_v01.jsonl"),
    )
    args = parser.parse_args()

    if not args.aiify_output.exists():
        print(f"Error: {args.aiify_output} not found. Run `just aiify` first.")
        return

    has_humanized = args.humanize_output.exists()

    if has_humanized:
        # Full 3-class flow
        print("Building triples (original → AIified → humanized)...")
        triples = build_triples(args.aiify_output, args.humanize_output)
        print(f"Built {len(triples)} triples")

        print("\nScoring all 3 classes with Layer 1...")
        scored = score_triples(triples)
        print_triple_report(scored)

        # 3-class AUROC: human vs AI
        all_scores = [t.original_score.overall for t in scored] + [
            t.aiified_score.overall for t in scored
        ]
        all_labels = [1] * len(scored) + [0] * len(scored)
        auroc_human_ai = _auroc(all_scores, all_labels)

        # 3-class AUROC: humanized vs AI
        hum_ai_scores = [t.humanized_score.overall for t in scored] + [
            t.aiified_score.overall for t in scored
        ]
        hum_ai_labels = [1] * len(scored) + [0] * len(scored)
        auroc_hum_ai = _auroc(hum_ai_scores, hum_ai_labels)

        # Hardest test: human vs humanized
        hum_hum_scores = [t.original_score.overall for t in scored] + [
            t.humanized_score.overall for t in scored
        ]
        hum_hum_labels = [1] * len(scored) + [0] * len(scored)
        auroc_hum_hum = _auroc(hum_hum_scores, hum_hum_labels)

        print(f"\n  AUROC (human vs AI):        {auroc_human_ai:.4f}")
        print(f"  AUROC (humanized vs AI):    {auroc_hum_ai:.4f}")
        print(f"  AUROC (human vs humanized): {auroc_hum_hum:.4f}")

        # Per-dim AUROC for human vs AI
        dim_names = list(scored[0].original_score.per_dim.keys())
        print("\n  Per-dimension AUROC (human vs AI):")
        for dim in dim_names:
            dim_scores = [t.original_score.per_dim[dim] for t in scored] + [
                t.aiified_score.per_dim[dim] for t in scored
            ]
            auc = _auroc(dim_scores, all_labels)
            bar = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
            print(f"    {dim:<22s} {bar} {auc:.4f}")

        # Export 3-class benchmark
        export_3class_benchmark(scored, args.benchmark_output)
        print(f"\n3-class benchmark exported to {args.benchmark_output}")

        # Export SFT pairs
        args.sft_output.parent.mkdir(parents=True, exist_ok=True)
        n_sft = export_sft_pairs(scored, args.sft_output)
        print(f"SFT-ready pairs exported: {n_sft} to {args.sft_output}")

    else:
        # Pairs-only flow
        print("Loading AIify output (no humanized data found)...")
        pairs = load_aiify_output(args.aiify_output)
        print(f"Loaded {len(pairs)} pairs")
        scored_pairs = score_pairs(pairs)
        print_pair_report(scored_pairs)

        all_scores = [p.original_score.overall for p in scored_pairs] + [
            p.aiified_score.overall for p in scored_pairs
        ]
        all_labels = [1] * len(scored_pairs) + [0] * len(scored_pairs)
        auroc = _auroc(all_scores, all_labels)
        print(f"\n  AUROC (Layer 1): {auroc:.4f}")


if __name__ == "__main__":
    main()
