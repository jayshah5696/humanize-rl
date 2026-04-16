"""CLI for the scoring pipeline.

Usage:
    uv run python -m humanize_rl.pipeline_cli
    uv run python -m humanize_rl.pipeline_cli --aiify-output output/01-aiify-dataset.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

from humanize_rl.benchmark.evaluator import _auroc
from humanize_rl.pipeline import (
    export_benchmark_from_pairs,
    load_aiify_output,
    print_pair_report,
    score_pairs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score AIified output and compute benchmark metrics"
    )
    parser.add_argument(
        "--aiify-output",
        type=Path,
        default=Path("output/01-aiify-dataset.jsonl"),
        help="Path to Arka AIify output JSONL",
    )
    parser.add_argument(
        "--benchmark-output",
        type=Path,
        default=Path("data/benchmark/scored_pairs_v01.jsonl"),
        help="Output path for scored benchmark",
    )
    args = parser.parse_args()

    if not args.aiify_output.exists():
        print(f"Error: {args.aiify_output} not found")
        print("Run the AIify pipeline first:")
        print("  uv run arka --config configs/01-aiify.yaml --run-id aiify-v01")
        return

    print("Loading AIify output...")
    pairs = load_aiify_output(args.aiify_output)
    print(f"Loaded {len(pairs)} pairs")

    print("\nScoring with Layer 1...")
    scored_pairs = score_pairs(pairs)

    print_pair_report(scored_pairs)

    # Compute AUROC
    all_scores = (
        [p.original_score.overall for p in scored_pairs]
        + [p.aiified_score.overall for p in scored_pairs]
    )
    all_labels = [1] * len(scored_pairs) + [0] * len(scored_pairs)
    auroc = _auroc(all_scores, all_labels)

    print(f"\n  AUROC (Layer 1 on real LLM-generated AI text): {auroc:.4f}")

    # Per-dimension AUROC
    dim_names = list(scored_pairs[0].original_score.per_dim.keys())
    print("\n  Per-dimension AUROC:")
    dim_aurocs = {}
    for dim in dim_names:
        dim_scores = (
            [p.original_score.per_dim[dim] for p in scored_pairs]
            + [p.aiified_score.per_dim[dim] for p in scored_pairs]
        )
        dim_auc = _auroc(dim_scores, all_labels)
        dim_aurocs[dim] = dim_auc

    for dim, auc in sorted(dim_aurocs.items(), key=lambda x: -x[1]):
        bar = "█" * int(auc * 20) + "░" * (20 - int(auc * 20))
        print(f"    {dim:<22s} {bar} {auc:.4f}")

    # Export
    export_benchmark_from_pairs(scored_pairs, args.benchmark_output)
    print(f"\nScored benchmark exported to {args.benchmark_output}")


if __name__ == "__main__":
    main()
