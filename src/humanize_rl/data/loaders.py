"""Streaming dataset loaders for compiling the scaled-up seed dataset.

Streams from HF Datasets:
- Email: corbt/enron-emails
- Instruction/Technical: HuggingFaceFW/fineweb-edu
- Blog/Opinion: cnn_dailymail
- Academic: liamdugan/raid (filtered for human abstracts)
- Creative: euclaise/writingprompts
"""

from __future__ import annotations

import argparse
import json
import re
from collections.abc import Generator
from pathlib import Path

from datasets import load_dataset

from humanize_rl.data.seed import DiscourseRole, LengthBand, Seed, to_arka_seed_row

# Anchor definition matching build_walking_skeleton_seeds.py
_ANCHOR_RE = re.compile(
    r"""
    \b\d+(?:\.\d+)+\b               # versions: 3.12, 1.2.3
    | \b\d{4}\b                     # years
    | \b[A-Z]{2,}(?:_[A-Z0-9]+)+\b  # SHOUTY_SNAKE constants
    | \b[A-Z]{2,}\b                 # acronyms (ETL, GDPR, SSO)
    | `[^`]+`                       # backtick code
    | [\w./-]+\.\w{1,5}\b           # filenames / paths
    | \b[a-z]+(?:[A-Z][a-z]+)+\b    # camelCase
    | \b\d+\s*(?:ms|s|MB|GB|KB|%|hr)\b  # metrics
    | \$[\d,]+(?:\.\d+)?            # dollar amounts
    """,
    re.VERBOSE,
)

_WORD_RE = re.compile(r"\b\w+\b")


def get_word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def get_anchor_count(text: str) -> int:
    return len(_ANCHOR_RE.findall(text))


def get_length_band(words: int) -> LengthBand:
    if words < 100:
        return "short"
    if words < 220:
        return "medium"
    return "long"


def clean_email(text: str) -> str:
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        if "-----Original Message-----" in line or "----- Forwarded by" in line:
            break
        cleaned_lines.append(line)
    text = "\n".join(cleaned_lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def clean_creative(text: str) -> str:
    text = text.replace("<br>", "\n").replace("<p>", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def truncate_to_sentences(text: str, min_words: int, max_words: int) -> str | None:
    """Truncates text to end on a sentence boundary within target word range."""
    words = _WORD_RE.findall(text)
    if len(words) < min_words:
        return None

    # Truncate at approx max words, then find last sentence boundary
    sliced_words = words[:max_words]
    char_limit = len(" ".join(sliced_words))
    sliced_text = text[: char_limit + 50]  # give some room to find terminal punctuation

    # Find the last period, question mark, or exclamation mark followed by whitespace or end
    matches = list(re.finditer(r"[.!?](?:\s|$)", sliced_text))
    if not matches:
        return None

    last_idx = matches[-1].end()
    final_text = sliced_text[:last_idx].strip()

    final_wc = get_word_count(final_text)
    if min_words <= final_wc <= max_words:
        return final_text
    return None


def stream_email(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming email seeds from corbt/enron-emails...")
    dataset = load_dataset("corbt/enron-emails", split="train", streaming=True)
    count = 0
    roles: list[DiscourseRole] = [
        "status_update",
        "request",
        "decision_rationale",
        "troubleshooting",
    ]

    for row in dataset:
        body = clean_email(row.get("body", ""))
        wc = get_word_count(body)
        if 80 <= wc <= 200:
            anchors = get_anchor_count(body)
            if anchors >= 2:
                role = roles[count % len(roles)]
                seed_id = f"v03_scale_email_{count:03d}"
                yield Seed(
                    id=seed_id,
                    text=body,
                    domain="email",
                    discourse_role=role,
                    source_dataset="enron",
                    length_band=get_length_band(wc),
                    word_count=wc,
                    anchors_count=anchors,
                    instruction="Write a short professional email from one teammate to another.",
                )
                count += 1
                if count >= target_count:
                    break


def stream_instruction_technical(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming instruction_technical seeds from HuggingFaceFW/fineweb-edu...")
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )
    count = 0

    for row in dataset:
        if row.get("int_score", 0) < 3:
            continue

        text = row.get("text", "").strip()
        # Look for code or configuration indications
        has_tech_keywords = any(
            kw in text.lower()
            for kw in [
                "python",
                "code",
                "database",
                "git",
                "install",
                "error",
                "deploy",
                "api",
                "server",
                "class",
                "function",
            ]
        )
        if not has_tech_keywords and "`" not in text:
            continue

        truncated = truncate_to_sentences(text, 100, 300)
        if not truncated:
            continue

        wc = get_word_count(truncated)
        anchors = get_anchor_count(truncated)
        if anchors >= 3:
            # Quick rule for troubleshooting vs explanation
            role = (
                "troubleshooting"
                if any(
                    x in truncated.lower() for x in ["error", "bug", "fail", "issue"]
                )
                else "instructional_explanation"
            )
            seed_id = f"v03_scale_tech_{count:03d}"
            yield Seed(
                id=seed_id,
                text=truncated,
                domain="instruction_technical",
                discourse_role=role,
                source_dataset="fineweb-edu",
                length_band=get_length_band(wc),
                word_count=wc,
                anchors_count=anchors,
                instruction="Share a short technical note from your own experience.",
            )
            count += 1
            if count >= target_count:
                break


def stream_blog_opinion(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming blog_opinion seeds from cnn_dailymail...")
    dataset = load_dataset("cnn_dailymail", "3.0.0", split="train", streaming=True)
    count = 0
    roles: list[DiscourseRole] = ["argument_opinion", "anecdote", "reflection"]

    for row in dataset:
        article = row.get("article", "").strip()
        # Strip Reuters / CNN datelines at the beginning
        article = re.sub(r"^[A-Z\s,]+(?:\(Reuters\)|\(CNN\))\s*--\s*", "", article)

        truncated = truncate_to_sentences(article, 100, 300)
        if not truncated:
            continue

        wc = get_word_count(truncated)
        anchors = get_anchor_count(truncated)
        if anchors >= 2:
            role = roles[count % len(roles)]
            seed_id = f"v03_scale_blog_{count:03d}"
            yield Seed(
                id=seed_id,
                text=truncated,
                domain="blog_opinion",
                discourse_role=role,
                source_dataset="cnn_dailymail",
                length_band=get_length_band(wc),
                word_count=wc,
                anchors_count=anchors,
                instruction="Write a short blog post or opinion piece on a current topic.",
            )
            count += 1
            if count >= target_count:
                break


def stream_academic(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming academic seeds from liamdugan/raid abstracts...")
    dataset = load_dataset("liamdugan/raid", split="train", streaming=True)
    count = 0
    roles: list[DiscourseRole] = [
        "methods",
        "literature_review",
        "discussion_limitations",
        "reported_summary",
    ]

    for row in dataset:
        if row.get("model") != "human" or row.get("domain") != "abstracts":
            continue

        text = row.get("generation", "").strip()
        wc = get_word_count(text)
        if 100 <= wc <= 300:
            anchors = get_anchor_count(text)
            if anchors >= 2:
                role = roles[count % len(roles)]
                seed_id = f"v03_scale_academic_{count:03d}"
                yield Seed(
                    id=seed_id,
                    text=text,
                    domain="academic",
                    discourse_role=role,
                    source_dataset="raid",
                    length_band=get_length_band(wc),
                    word_count=wc,
                    anchors_count=anchors,
                    instruction="Write an academic abstract summarizing a research study.",
                )
                count += 1
                if count >= target_count:
                    break


def stream_creative(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming creative seeds from euclaise/writingprompts...")
    dataset = load_dataset("euclaise/writingprompts", split="train", streaming=True)
    count = 0
    roles: list[DiscourseRole] = ["scene", "narrative_reflection"]

    for row in dataset:
        story = clean_creative(row.get("story", ""))
        truncated = truncate_to_sentences(story, 120, 350)
        if not truncated:
            continue

        wc = get_word_count(truncated)
        anchors = get_anchor_count(truncated)
        # Creative writing has fewer technical anchors, so we allow >= 1
        if anchors >= 1:
            role = roles[count % len(roles)]
            seed_id = f"v03_scale_creative_{count:03d}"
            yield Seed(
                id=seed_id,
                text=truncated,
                domain="creative",
                discourse_role=role,
                source_dataset="writingprompts",
                length_band=get_length_band(wc),
                word_count=wc,
                anchors_count=anchors,
                instruction="Write a short creative story or narrative passage.",
            )
            count += 1
            if count >= target_count:
                break


def stream_email_customer(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming customer email seeds from rtweera/customer_care_emails...")
    dataset = load_dataset(
        "rtweera/customer_care_emails", split="train", streaming=True
    )
    count = 0
    roles: list[DiscourseRole] = ["request", "troubleshooting", "decision_rationale"]

    for row in dataset:
        body = clean_email(row.get("message_body", ""))
        wc = get_word_count(body)
        if 50 <= wc <= 200:
            anchors = get_anchor_count(body)
            if anchors >= 2:
                role = roles[count % len(roles)]
                seed_id = f"v03_scale_email_cust_{count:03d}"
                yield Seed(
                    id=seed_id,
                    text=body,
                    domain="email",
                    discourse_role=role,
                    source_dataset="customer_care_emails",
                    length_band=get_length_band(wc),
                    word_count=wc,
                    anchors_count=anchors,
                    instruction="Write a customer support email addressing an issue.",
                )
                count += 1
                if count >= target_count:
                    break


def stream_creative_tiny(target_count: int) -> Generator[Seed, None, None]:
    print("Streaming creative seeds from roneneldan/TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    count = 0
    roles: list[DiscourseRole] = ["scene", "narrative_reflection", "anecdote"]

    for row in dataset:
        story = clean_creative(row.get("text", ""))
        if not story:
            continue

        wc = get_word_count(story)
        if 40 <= wc <= 300:
            anchors = get_anchor_count(story)
            if anchors >= 0:
                role = roles[count % len(roles)]
                seed_id = f"v03_scale_creative_tiny_{count:03d}"
                yield Seed(
                    id=seed_id,
                    text=story,
                    domain="creative",
                    discourse_role=role,
                    source_dataset="TinyStories",
                    length_band=get_length_band(wc),
                    word_count=wc,
                    anchors_count=anchors,
                    instruction="Write a short narrative story about a specific situation and its outcome.",
                )
                count += 1
                if count >= target_count:
                    break


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scale up corpus seeds via HF streaming."
    )
    parser.add_argument(
        "--num-seeds", type=int, default=1000, help="Total number of seeds to compile."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="seeds/v03/corpus_seeds.jsonl",
        help="Output file path.",
    )
    args = parser.parse_args()

    total = args.num_seeds
    # Calculate target counts proportionally
    targets = {
        "email": int(total * 0.25),
        "instruction_technical": int(total * 0.25),
        "blog_opinion": int(total * 0.25),
        "academic": int(total * 0.15),
        "creative": total - int(total * 0.25) * 3 - int(total * 0.15),
    }

    print(f"Compiling {total} seeds with distribution: {targets}")

    seeds: list[Seed] = []

    # Run streams
    seeds.extend(stream_email(targets["email"]))
    seeds.extend(stream_instruction_technical(targets["instruction_technical"]))
    seeds.extend(stream_blog_opinion(targets["blog_opinion"]))
    seeds.extend(stream_academic(targets["academic"]))
    seeds.extend(stream_creative(targets["creative"]))

    # Shuffle compiled seeds so domains are interleaved
    import random

    random.seed(42)
    random.shuffle(seeds)

    # Ensure stable sequential IDs after mixing
    for idx, seed in enumerate(seeds):
        # Format id as v03_scale_XXXX
        seed.id = f"v03_scale_{idx:04d}"

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as f:
        for seed in seeds:
            f.write(json.dumps(to_arka_seed_row(seed)) + "\n")

    print(f"Success! Wrote {len(seeds)} seeds to {out_path}")


if __name__ == "__main__":
    main()
