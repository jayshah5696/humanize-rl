from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/scorer/l2_labeled_v01.jsonl"))
    parser.add_argument("--backup", action="store_true", default=True)
    args = parser.parse_args()

    rows_by_id = {}
    order = []
    for line in args.input.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row["id"]
        if row_id not in rows_by_id:
            order.append(row_id)
        rows_by_id[row_id] = row

    original_lines = sum(1 for line in args.input.read_text().splitlines() if line.strip())
    if args.backup:
        backup_path = args.input.with_suffix(args.input.suffix + ".bak")
        backup_path.write_text(args.input.read_text())

    with args.input.open("w") as handle:
        for row_id in order:
            handle.write(json.dumps(rows_by_id[row_id]) + "\n")

    print(f"Deduped {args.input}: {original_lines} -> {len(order)} rows")


if __name__ == "__main__":
    main()
