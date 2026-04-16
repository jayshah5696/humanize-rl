"""CLI for Layer 1 humanness scoring.

Usage:
    uv run python -m humanize_rl.scoring.cli "Your text here"
    uv run python -m humanize_rl.scoring.cli --file path/to/file.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from humanize_rl.scoring.aggregator import score_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Layer 1 humanness scorer — 8 deterministic dimensions"
    )
    parser.add_argument("text", nargs="?", help="Text to score")
    parser.add_argument("--file", "-f", type=Path, help="File to score")
    args = parser.parse_args()

    if args.file:
        text = args.file.read_text()
    elif args.text:
        text = args.text
    else:
        # Read from stdin
        text = sys.stdin.read()

    if not text.strip():
        print("Error: no text provided", file=sys.stderr)
        sys.exit(1)

    result = score_text(text)
    print(result)


if __name__ == "__main__":
    main()
