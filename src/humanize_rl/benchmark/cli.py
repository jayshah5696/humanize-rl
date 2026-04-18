"""CLI for running and building benchmark artifacts.

Usage:
    uv run python -m humanize_rl.benchmark.cli
    uv run python -m humanize_rl.benchmark.cli --output data/benchmark/scored.jsonl
    uv run python -m humanize_rl.benchmark.cli --build-dataset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from humanize_rl.benchmark.datasets import (
    build_mvp_benchmark_dataset,
    build_repo_benchmark_dataset,
    format_dataset_summary,
    load_benchmark_dataset,
)
from humanize_rl.benchmark.evaluator import evaluate, export_scored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Layer 1 benchmark on human vs AI samples"
    )
    parser.add_argument(
        "--human",
        type=Path,
        default=Path("data/benchmark/human_samples.jsonl"),
        help="Path to human samples JSONL",
    )
    parser.add_argument(
        "--ai",
        type=Path,
        default=Path("data/benchmark/ai_samples.jsonl"),
        help="Path to AI samples JSONL",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Export scored samples to JSONL",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="Build the normalized 3-class benchmark dataset from repo artifacts",
    )
    parser.add_argument(
        "--seed-input",
        type=Path,
        default=Path("seeds/human_seeds_v01.jsonl"),
        help="Seed JSONL used to enrich human rows in the built benchmark dataset",
    )
    parser.add_argument(
        "--scored-3class-input",
        type=Path,
        default=Path("data/benchmark/scored_3class_v01.jsonl"),
        help="Scored 3-class JSONL used to build the benchmark dataset",
    )
    parser.add_argument(
        "--scored-binary-input",
        type=Path,
        default=Path("data/benchmark/scored_output.jsonl"),
        help="Binary scored JSONL used to expand the repo benchmark with full-text human/AI rows",
    )
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=Path("data/benchmark/test_set_v02.jsonl"),
        help="Path for the built benchmark dataset",
    )
    args = parser.parse_args()

    if args.build_dataset:
        temp_output = args.dataset_output.with_name("_tmp_test_set_v01.jsonl")
        build_mvp_benchmark_dataset(
            scored_path=args.scored_3class_input,
            seeds_path=args.seed_input,
            output_path=temp_output,
        )
        written = build_repo_benchmark_dataset(
            base_dataset_path=temp_output,
            human_path=args.human,
            ai_path=args.ai,
            scored_output_path=args.scored_binary_input,
            output_path=args.dataset_output,
        )
        dataset = load_benchmark_dataset(args.dataset_output)
        print(f"Built benchmark dataset at {args.dataset_output} ({written} rows)")
        print(format_dataset_summary(dataset))
        if temp_output.exists():
            temp_output.unlink()
        return

    report = evaluate(args.human, args.ai)
    print(report)
    print(f"\nBy label: {report.by_label}")
    print(f"By source: {report.by_source}")
    print(f"By domain: {report.by_domain}")

    if args.output:
        export_scored(report, args.output)
        print(f"\nScored samples exported to {args.output}")


if __name__ == "__main__":
    main()
