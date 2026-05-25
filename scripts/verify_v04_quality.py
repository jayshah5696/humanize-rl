import argparse
import json
import re
from collections import Counter
from pathlib import Path

RESPONSE_REJECT_PHRASES = [
    "certainly",
    "of course",
    "great question",
    "i'd be happy to",
    "it is worth noting",
    "it's worth noting",
    "furthermore",
    "moreover",
    "in conclusion",
    "please don't hesitate",
    "i hope this email finds you well",
]

META_INSTRUCTION_PHRASES = [
    "chatgpt",
    "ai-generated",
    "ai generated",
    "ai-written",
    "ai written",
    "humanize this",
    "human wrote it",
]

COMMON_FAKE_NAMES = ["Sarah", "Marcus", "John", "Jane", "Alice", "Bob", "Charlie"]
PLACEHOLDER_RE = re.compile(r"\[[A-Za-z ]+\]")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def flag_row(row: dict) -> list[str]:
    flags = []
    instruction = row.get("instruction", "")
    response = row.get("response", "")
    lower_i = instruction.lower()
    lower_r = response.lower()

    for phrase in RESPONSE_REJECT_PHRASES:
        if phrase in lower_r:
            flags.append(f"response_reject_phrase:{phrase}")

    for phrase in META_INSTRUCTION_PHRASES:
        if phrase in lower_i:
            flags.append(f"meta_instruction_phrase:{phrase}")

    # For rewrite rows, catch obvious unsupported proper-name insertion.
    quoted = "'" in instruction or '"' in instruction
    if quoted or instruction.lower().startswith(("rewrite", "clean up", "shorten", "make this", "fix", "condense", "tighten")):
        for name in COMMON_FAKE_NAMES:
            if name in response and name not in instruction:
                flags.append(f"possible_fake_name:{name}")

    # For direct drafts, prefer placeholders when missing details are requested.
    asks_cover = "who will cover" in lower_i or "cover my projects" in lower_i
    if asks_cover and not PLACEHOLDER_RE.search(response) and any(name in response for name in COMMON_FAKE_NAMES):
        flags.append("missing_placeholder_for_unspecified_cover")

    if len(response) > 2500:
        flags.append("response_too_long")
    if len(response) < 20:
        flags.append("response_too_short")

    return flags


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("--sample", type=int, default=20)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.path))
    counts = Counter()
    bad = []
    for idx, row in enumerate(rows, start=1):
        flags = flag_row(row)
        for flag in flags:
            counts[flag] += 1
        if flags:
            bad.append((idx, flags, row))

    print(f"rows={len(rows)} flagged_rows={len(bad)}")
    if counts:
        print("flags:")
        for key, value in counts.most_common():
            print(f"  {key}: {value}")
    else:
        print("flags: none")

    print("\nfirst flagged samples:")
    for idx, flags, row in bad[: args.sample]:
        print(f"\n#{idx} {flags}")
        print("I:", row.get("instruction", "")[:300].replace("\n", " "))
        print("R:", row.get("response", "")[:300].replace("\n", " "))


if __name__ == "__main__":
    main()
