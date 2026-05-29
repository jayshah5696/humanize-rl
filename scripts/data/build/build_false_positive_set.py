from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset


def clean(text: Any, max_chars: int = 2500) -> str | None:
    if not isinstance(text, str):
        return None
    text = " ".join(text.split())
    if len(text) < 300:
        return None
    return text[:max_chars]


def add(rows: list[dict], text: str | None, source: str, domain: str, fmt: str) -> None:
    if text:
        rows.append(
            {
                "id": f"fp_{len(rows):05d}",
                "text": text,
                "binary_label": 0,
                "label_type": "human_authored",
                "domain": domain,
                "source": source,
                "format_bucket": fmt,
                "group": f"{source}:{len(rows)}",
                "has_existing_rubric": False,
            }
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/scorer/false_positive_human_v01.jsonl"))
    parser.add_argument("--per-source", type=int, default=100)
    args = parser.parse_args()
    rows: list[dict] = []

    # Human formal emails: real Enron-derived, non-commercial license; use for eval only.
    try:
        ds = load_dataset("snap-stanford/humanual-email", split="train", streaming=True)
        for item in ds.take(args.per_source):
            text = clean(item.get("completion") or item.get("email") or item.get("text") or item.get("content"))
            add(rows, text, "snap-stanford/humanual-email", "email", "plain")
    except Exception as exc:
        print(f"skip humanual-email: {exc}")

    # Human academic abstracts, CC0 metadata.
    try:
        ds = load_dataset("open-index/open-arxiv", split="train", streaming=True)
        count = 0
        for item in ds:
            text = clean(item.get("abstract"))
            if text:
                title = item.get("title")
                text = f"# {title}\n\n{text}" if title else text
                add(rows, text, "open-index/open-arxiv", "academic", "markdown_heading")
                count += 1
            if count >= args.per_source:
                break
    except Exception as exc:
        print(f"skip open-arxiv: {exc}")

    # Human StackExchange markdown answers/questions.
    try:
        ds = load_dataset("marin-community/stackexchange-markdown", split="train", streaming=True)
        count = 0
        for item in ds:
            text = clean(item.get("text") or item.get("markdown") or item.get("content"))
            if text:
                add(rows, text, "marin-community/stackexchange-markdown", "technical", "markdown_heading" if "#" in text else "list_or_bullets")
                count += 1
            if count >= args.per_source:
                break
    except Exception as exc:
        print(f"skip stackexchange-markdown: {exc}")

    # Existing human authored rows from our L2 set that are formal/list-like.
    try:
        for line in Path("data/scorer/l2_labeled_v02_10k.jsonl").read_text().splitlines():
            item = json.loads(line)
            if item.get("label_type") == "human_authored" and item.get("domain") in {"academic", "email", "nyt_story"}:
                add(rows, clean(item.get("text")), "humanize-rl/l2_labeled_v02_10k", item.get("domain", "unknown"), item.get("format_bucket", "plain"))
            if len(rows) >= args.per_source * 4:
                break
    except Exception as exc:
        print(f"skip local l2: {exc}")

    seen = set()
    deduped = []
    for row in rows:
        if row["text"] in seen:
            continue
        seen.add(row["text"])
        row["id"] = f"fp_{len(deduped):05d}"
        deduped.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        for row in deduped:
            handle.write(json.dumps(row) + "\n")
    print(f"Wrote {len(deduped)} rows to {args.output}")


if __name__ == "__main__":
    main()
