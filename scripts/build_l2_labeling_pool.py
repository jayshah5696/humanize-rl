from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from humanize_rl.data.scorer_dataset import load_scorer_data


def iter_rows(data: dict, split: str):
    for i, text in enumerate(data["texts"]):
        yield {
            "id": f"{split}_{i:06d}",
            "text": text,
            "binary_label": int(data["labels"][i]),
            "label_type": data["label_types"][i],
            "domain": data["domains"][i],
            "source": data["sources"][i],
            "format_bucket": data["format_buckets"][i],
            "group": data["groups"][i],
            "has_existing_rubric": bool(np.all(data["rubrics"][i] != -1.0)),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/scorer/l2_labeling_pool_v01.jsonl"))
    parser.add_argument("--include-hf", action="store_true")
    parser.add_argument("--balance-format", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    train, val = load_scorer_data(
        hf_dataset_name="gsingh1-py/train" if args.include_hf else None,
        split_strategy="random",
        balance_format=args.balance_format,
    )
    rows = list(iter_rows(train, "train")) + list(iter_rows(val, "val"))
    if args.limit is not None:
        rows = rows[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
