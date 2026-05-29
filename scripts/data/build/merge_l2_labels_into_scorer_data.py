from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/scorer/l2_labeled_v01.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/scorer/l2_labeled_scorer_v01.jsonl"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.input.open() as src, args.output.open("w") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            label = "ai" if row["binary_label"] == 1 else row["label_type"]
            out = {
                "label": label,
                "domain": row.get("domain"),
                "source": row.get("source"),
                "text": row["text"],
                "l1_overall": row.get("l1_overall"),
                "l1_per_dim": row.get("l1_per_dim"),
                "l2_overall": row.get("l2_overall"),
                "l2_per_dim": row.get("l2_per_dim"),
                "l2_raw": row.get("l2_raw"),
                "l2_reasoning": row.get("l2_reasoning"),
            }
            dst.write(json.dumps(out) + "\n")
            n += 1
    print(f"Wrote {n} scorer rows to {args.output}")


if __name__ == "__main__":
    main()
