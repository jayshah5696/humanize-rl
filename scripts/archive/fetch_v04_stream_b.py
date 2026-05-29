"""Fetch real human-written text from approved HF sources for Stream B.

Target: 2000 rows, 200-1500 chars, English, good domain mix.
Sources are all MIT/Apache/CC-BY licensed (see data/source_manifest_v04.json).
"""
import json
import random
from pathlib import Path

DOMAIN_TARGETS = {
    "email": 700,         # wardacoder/business-email-dataset
    "chat": 400,          # nikcane/slack + synthetic workplace chat
    "technical": 400,     # liamdugan/raid human rows (news/wiki/reddit)
    "creative": 250,      # euclaise/writingprompts (high-score only)
    "informative": 250,   # fineweb-edu score>=4
}

MIN_CHARS = 120
MAX_CHARS = 1200


def clean(text: str) -> str:
    import re
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def fetch_emails(limit: int) -> list[dict]:
    from datasets import load_dataset
    rows = []
    ds = load_dataset("wardacoder/business-email-dataset", split="train", streaming=True)
    for row in ds:
        if len(rows) >= limit: break
        body = clean(row.get("output", "") or "")
        if MIN_CHARS <= len(body) <= MAX_CHARS and "I hope this" not in body:
            rows.append({"text": body, "domain": "email",
                         "source_dataset": "wardacoder/business-email-dataset"})
    print(f"email: {len(rows)}")
    return rows


def fetch_slack(limit: int) -> list[dict]:
    from datasets import load_dataset
    rows = []
    ds = load_dataset("nikcane/slack", split="train", streaming=True)
    for row in ds:
        if len(rows) >= limit: break
        text = clean(row.get("text", "") or "")
        if MIN_CHARS <= len(text) <= MAX_CHARS and not text.startswith("<http"):
            rows.append({"text": text, "domain": "chat",
                         "source_dataset": "nikcane/slack"})
    print(f"slack: {len(rows)}")
    return rows


def fetch_raid_human(limit: int) -> list[dict]:
    from datasets import load_dataset
    GOOD_DOMAINS = {"news", "reddit", "wiki", "book", "paper", "review"}
    rows = []
    ds = load_dataset("liamdugan/raid", split="train", streaming=True)
    for row in ds:
        if len(rows) >= limit: break
        if row.get("model") != "human": continue
        d = row.get("domain", "general").lower()
        if not any(g in d for g in GOOD_DOMAINS): continue
        text = clean(row.get("generation", "") or "")
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            rows.append({"text": text, "domain": "general",
                         "source_dataset": "liamdugan/raid"})
    print(f"raid-human: {len(rows)}")
    return rows


def fetch_writingprompts(limit: int) -> list[dict]:
    from datasets import load_dataset
    rows = []
    ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
    for row in ds:
        if len(rows) >= limit * 6: break  # over-sample, filter below
        text = clean(row.get("story", "") or "")
        # take only first ~400 chars of stories
        if len(text) > 400:
            text = text[:text.rfind('.', 200, 500) + 1] if '.' in text[200:500] else text[:400]
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            rows.append({"text": text, "domain": "creative",
                         "source_dataset": "euclaise/writingprompts"})
    random.shuffle(rows)
    rows = rows[:limit]
    print(f"writingprompts: {len(rows)}")
    return rows


def fetch_fineweb(limit: int) -> list[dict]:
    from datasets import load_dataset
    rows = []
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True)
    for row in ds:
        if len(rows) >= limit: break
        if row.get("score", 0) < 4.0: continue
        # take first paragraph or 300 chars
        text = clean(row.get("text", "") or "")
        if len(text) > 600:
            idx = text.rfind('\n', 200, 700)
            text = text[:idx] if idx > 200 else text[:600]
        if MIN_CHARS <= len(text) <= MAX_CHARS:
            rows.append({"text": text, "domain": "informative",
                         "source_dataset": "HuggingFaceFW/fineweb-edu"})
    print(f"fineweb-edu: {len(rows)}")
    return rows


def main():
    out = Path("data/raw/v04_stream_b_sources.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    all_rows += fetch_emails(DOMAIN_TARGETS["email"])
    all_rows += fetch_slack(DOMAIN_TARGETS["chat"])
    all_rows += fetch_raid_human(DOMAIN_TARGETS["technical"])
    all_rows += fetch_writingprompts(DOMAIN_TARGETS["creative"])
    all_rows += fetch_fineweb(DOMAIN_TARGETS["informative"])

    random.shuffle(all_rows)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in all_rows))
    print(f"\nTotal: {len(all_rows)} rows → {out}")


if __name__ == "__main__":
    main()
