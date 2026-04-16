"""CLI for running the benchmark.

Usage:
    uv run python -m humanize_rl.benchmark.cli
    uv run python -m humanize_rl.benchmark.cli --output data/benchmark/scored.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

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
    args = parser.parse_args()

    report = evaluate(args.human, args.ai)
    print(report)

    if args.output:
        export_scored(report, args.output)
        print(f"\nScored samples exported to {args.output}")


if __name__ == "__main__":
    main()
