from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from humanize_rl.scoring.aggregator import score_text
from humanize_rl.scoring.layer2 import _default_layer2, score_layer2


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    with path.open() as handle:
        for line in handle:
            if line.strip():
                done.add(json.loads(line)["id"])
    return done


def dedupe_output(path: Path) -> None:
    if not path.exists():
        return
    rows_by_id = {}
    order = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        row_id = row["id"]
        if row_id not in rows_by_id:
            order.append(row_id)
        rows_by_id[row_id] = row
    with path.open("w") as handle:
        for row_id in order:
            handle.write(json.dumps(rows_by_id[row_id]) + "\n")


def batched(items: list[dict], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def score_one(row: dict, model: str):
    try:
        l2 = score_layer2(
            text=row["text"],
            instruction="Evaluate this text for AI writing patterns. Return rubric scores only.",
            model=model,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[WARN] L2 failed for {row['id']}: {exc}")
        l2 = _default_layer2()
    l1 = score_text(row["text"])
    return row, l1, l2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/scorer/l2_labeling_pool_v01.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("data/scorer/l2_labeled_v01.jsonl"))
    parser.add_argument("--model", default="google/gemini-3.1-pro-preview")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep", type=float, default=0.0)
    args = parser.parse_args()

    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    dedupe_output(args.output)
    done = load_done(args.output)
    pending = [row for row in rows if row["id"] not in done]
    if args.limit is not None:
        pending = pending[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    print(f"Total={len(rows)} done={len(done)} pending_this_run={len(pending)}")

    with args.output.open("a") as handle:
        for batch in batched(pending, args.batch_size):
            current_done = load_done(args.output)
            batch = [row for row in batch if row["id"] not in current_done]
            results = []
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = [executor.submit(score_one, row, args.model) for row in batch]
                for future in as_completed(futures):
                    results.append(future.result())
            results.sort(key=lambda item: item[0]["id"])

            for row, l1, l2 in results:
                if row["id"] in current_done:
                    continue
                out = {
                    **row,
                    "l1_overall": l1.overall,
                    "l1_per_dim": l1.per_dim,
                    "l2_overall": l2.overall,
                    "l2_per_dim": l2.per_dim,
                    "l2_raw": l2.raw_scores,
                    "l2_reasoning": l2.reasoning,
                    "judge_model": l2.judge_model,
                    "latency_ms": l2.latency_ms,
                }
                handle.write(json.dumps(out) + "\n")
                current_done.add(row["id"])
            handle.flush()
            print(f"Wrote batch of {len(batch)}; done now ~{len(load_done(args.output))}")
            if args.sleep:
                time.sleep(args.sleep)


if __name__ == "__main__":
    main()
