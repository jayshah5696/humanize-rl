"""Duplicate each row of a JSONL N times.

Used by V-Slice 2 to fan out seeds into multiple candidates: arka's
`TransformGeneratorStage` emits exactly one output per input, so to get
N AIify or humanize candidates per seed we duplicate the source rows.

Usage:
    uv run python scripts/duplicate_seeds.py \
        --input seeds/v03/walking_skeleton.jsonl \
        --output seeds/v03/walking_skeleton_x2.jsonl \
        --copies 2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Duplicate JSONL rows N times.")
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--copies", type=int, default=2)
    args = p.parse_args()

    rows: list[dict] = []
    with args.input.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for r in rows:
            for i in range(args.copies):
                # Tag the copy index so duplicates aren't dropped by any
                # downstream id-based dedup.
                copy = dict(r)
                if "id" in copy:
                    copy["id"] = f"{copy['id']}__c{i}"
                f.write(json.dumps(copy) + "\n")

    print(
        f"Wrote {len(rows) * args.copies} rows ({args.copies}x of {len(rows)}) to {args.output}"
    )


if __name__ == "__main__":
    main()
