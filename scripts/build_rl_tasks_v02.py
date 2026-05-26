"""Build humanize_tasks_v02 — larger task set from real source texts.

Sources:
- data/raw/v04_stream_b_seeds.jsonl  (1915 email/creative/general, ~133 words avg)
- data/processed/v04_sft_final.jsonl  (4835 instruction/response pairs)

Targets: ~512 tasks minimum, covering all 5 reward profiles,
with input texts ranging from 50-220 words.

Usage:
    uv run scripts/build_rl_tasks_v02.py
    uv run scripts/build_rl_tasks_v02.py --max-tasks 1024 --split-seed 99
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

import click

from humanize_rl.reward.tasks import (
    RLTask,
    TaskConstraints,
    summarize_tasks,
    write_tasks,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

NUMBER_RE = re.compile(r"\b(?:\d[\d,.:/-]*|\$[\d,]+)\b")
ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]{2,})*\b")
STOP = {"The", "This", "That", "They", "We", "You", "It", "A", "An", "In",
        "On", "At", "For", "To", "Of", "And", "But", "Or", "So", "If",
        "Be", "Is", "Are", "Was", "Were", "Has", "Have", "Had", "Do",
        "Does", "Did", "Will", "Would", "Could", "Should", "May", "Can",
        "Our", "Your", "Their", "His", "Her", "My", "All", "No", "Not",
        "Dear", "Hi", "Hello", "Subject", "Re", "Email", "Message",
        "Please", "Thank", "Thanks", "Best", "Regards", "Sincerely"}


def extract_numbers(text: str) -> list[str]:
    return list({m.group(0) for m in NUMBER_RE.finditer(text)})[:4]


def extract_entities(text: str) -> list[str]:
    seen = []
    for m in ENTITY_RE.finditer(text):
        e = m.group(0)
        if e not in STOP and e not in seen:
            seen.append(e)
    return seen[:4]


def word_count(text: str) -> int:
    return len(text.split())


def stable_id(index: int) -> str:
    return f"rl_v01_{index:06d}"


def _first_sentence_nouns(text: str) -> list[str]:
    """Rough required_facts: first 2 numbers + first 2 entities from text."""
    facts = []
    for n in extract_numbers(text):
        facts.append(n)
        if len(facts) >= 2:
            break
    for e in extract_entities(text):
        if e not in facts:
            facts.append(e)
        if len(facts) >= 4:
            break
    return facts


# ---------------------------------------------------------------------------
# task builders per family
# ---------------------------------------------------------------------------

def make_rewrite_repair(text: str, domain: str, idx: int) -> dict:
    wc = word_count(text)
    max_words = min(max(wc // 3, 30), 80)
    return dict(
        family="rewrite_repair",
        domain=domain,
        mode="rewrite",
        register="casual",
        instruction=(
            f"Rewrite this {domain} message to sound natural and human. "
            f"Keep it under {max_words} words. Return only the message."
        ),
        input_text=text,
        constraints=dict(
            max_words=max_words,
            preserve_numbers=True,
            preserve_entities=True,
            no_subject_line=True,
            no_signoff=True,
        ),
        reward_profile="rewrite_faithful_concise",
        trap_tags=["over_polish", "wrapper_phrase", "verbosity"],
        required_facts=_first_sentence_nouns(text),
        forbidden_facts=[],
    )


def make_compression(text: str, domain: str, idx: int) -> dict:
    wc = word_count(text)
    max_words = min(max(wc // 4, 25), 60)
    return dict(
        family="compression",
        domain=domain,
        mode="rewrite",
        register="terse",
        instruction=(
            f"Compress this into a single tight {domain} update. "
            f"Under {max_words} words. Facts only, no padding. Return only the message."
        ),
        input_text=text,
        constraints=dict(
            max_words=max_words,
            preserve_numbers=True,
            preserve_entities=True,
            no_subject_line=True,
            no_signoff=True,
            allow_markdown=False,
        ),
        reward_profile="compression_update",
        trap_tags=["verbosity", "padding", "over_polish"],
        required_facts=_first_sentence_nouns(text),
        forbidden_facts=[],
    )


def make_tone_shift(text: str, domain: str, idx: int) -> dict:
    wc = word_count(text)
    max_words = min(max(wc // 2, 40), 100)
    return dict(
        family="tone_shift",
        domain=domain,
        mode="rewrite",
        register="warm_professional",
        instruction=(
            "Rewrite this to sound warm and direct — like a real colleague, "
            f"not a corporate template. Under {max_words} words. Return only the message."
        ),
        input_text=text,
        constraints=dict(
            max_words=max_words,
            preserve_numbers=True,
            preserve_entities=True,
            no_subject_line=True,
            no_signoff=True,
        ),
        reward_profile="sensitive_comms",
        trap_tags=["corporate_filler", "register_mismatch", "over_polish"],
        required_facts=_first_sentence_nouns(text),
        forbidden_facts=[],
    )


def make_ai_tell_removal(text: str, domain: str, idx: int) -> dict:
    wc = word_count(text)
    max_words = min(max(wc, 40), 150)
    return dict(
        family="adversarial_ai_tell_removal",
        domain=domain,
        mode="rewrite",
        register="natural",
        instruction=(
            "Remove all AI-sounding phrases from this text. "
            "No 'furthermore', 'it is worth noting', 'in conclusion', or similar. "
            f"Keep all facts. Under {max_words} words. Return only the rewritten text."
        ),
        input_text=text,
        constraints=dict(
            max_words=max_words,
            preserve_numbers=True,
            preserve_entities=True,
        ),
        reward_profile="rewrite_faithful_concise",
        trap_tags=["ai_tell", "over_polish", "wrapper_phrase"],
        required_facts=_first_sentence_nouns(text),
        forbidden_facts=["furthermore", "it is worth noting", "in conclusion",
                         "moreover", "additionally"],
    )


def make_technical_explain(text: str, domain: str, idx: int) -> dict:
    wc = word_count(text)
    max_words = min(max(wc, 60), 200)
    return dict(
        family="technical_explain",
        domain=domain,
        mode="rewrite",
        register="plain_english",
        instruction=(
            "Rewrite this in plain English for a non-technical audience. "
            "No jargon, no bullet lists. Conversational tone. "
            f"Under {max_words} words. Return only the explanation."
        ),
        input_text=text,
        constraints=dict(
            max_words=max_words,
            preserve_numbers=True,
            allow_markdown=False,
        ),
        reward_profile="technical_explain_natural",
        trap_tags=["jargon", "list_overuse", "over_formality"],
        required_facts=_first_sentence_nouns(text),
        forbidden_facts=[],
    )


BUILDERS = [
    make_rewrite_repair,
    make_compression,
    make_tone_shift,
    make_ai_tell_removal,
    make_technical_explain,
]

DOMAIN_MAP = {
    "email": "email",
    "creative": "creative_general",
    "general": "product",
    "chat": "slack",
    "slack": "slack",
    "blog": "creative_general",
    "article": "creative_general",
    "technical": "technical",
}


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_stream_b(path: Path, min_words: int = 40) -> list[tuple[str, str]]:
    """Load (text, domain) pairs from v04 stream b seeds."""
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        text = row.get("instruction", "").strip()
        domain = DOMAIN_MAP.get(row.get("domain", "general"), "general")
        if word_count(text) >= min_words:
            out.append((text, domain))
    return out


def load_sft(path: Path, min_words: int = 60) -> list[tuple[str, str]]:
    """Load longer instruction texts from SFT dataset as rewrite sources."""
    out = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        text = row.get("instruction", "").strip()
        # strip leading imperative verb if short
        if word_count(text) >= min_words:
            # infer domain from content
            t_lower = text.lower()
            if "email" in t_lower or "dear" in t_lower:
                domain = "email"
            elif "slack" in t_lower:
                domain = "slack"
            elif "blog" in t_lower or "article" in t_lower:
                domain = "creative_general"
            elif any(w in t_lower for w in ["code", "api", "database", "server", "deploy"]):
                domain = "technical"
            else:
                domain = "product"
            out.append((text, domain))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--max-tasks", default=512, help="Target number of tasks to generate")
@click.option("--min-words", default=40, help="Minimum source text word count")
@click.option("--split-seed", default=42, help="Random seed for train/eval/test split")
@click.option("--out", default="data/rl/humanize_tasks_v02.jsonl", help="Output path")
@click.option("--smoke-out", default="environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl",
              help="Bundled smoke subset path")
@click.option("--smoke-n", default=100, help="Number of tasks in smoke subset")
def main(max_tasks: int, min_words: int, split_seed: int, out: str, smoke_out: str, smoke_n: int) -> None:
    rng = random.Random(split_seed)

    # Load sources
    stream_b = load_stream_b(Path("data/raw/v04_stream_b_seeds.jsonl"), min_words=min_words)
    sft = load_sft(Path("data/processed/v04_sft_final.jsonl"), min_words=min_words)
    all_sources = stream_b + sft
    rng.shuffle(all_sources)

    click.echo(f"Sources: stream_b={len(stream_b)} sft={len(sft)} total={len(all_sources)}")

    # Generate tasks by cycling through builders
    tasks: list[RLTask] = []
    id_counter = 1

    for i, (text, domain) in enumerate(all_sources):
        if len(tasks) >= max_tasks:
            break
        builder = BUILDERS[i % len(BUILDERS)]
        try:
            raw = builder(text, domain, i)
            # Assign split: 80% train, 10% eval, 10% test
            r = rng.random()
            split = "train" if r < 0.80 else ("validation" if r < 0.90 else "test")
            task = RLTask.model_validate({
                **raw,
                "id": stable_id(id_counter),
                "source": "v04_stream_b_and_sft",
                "split": split,
                "release_eligible": True,
                "source_group": f"{raw['family']}_{domain}_{i:04d}",
                "license": "project_synthetic",
            })
            tasks.append(task)
            id_counter += 1
        except Exception as e:
            click.echo(f"  skip row {i}: {e}", err=True)
            continue

    click.echo(f"\nGenerated {len(tasks)} tasks")
    click.echo(summarize_tasks(tasks))

    # Write full set
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(out_path, tasks)
    click.echo(f"Written: {out_path}")

    # Write smoke subset (stratified by family)
    smoke_tasks = _stratified_sample(tasks, smoke_n, rng)
    smoke_path = Path(smoke_out)
    smoke_path.parent.mkdir(parents=True, exist_ok=True)
    write_tasks(smoke_path, smoke_tasks)
    click.echo(f"Smoke subset ({len(smoke_tasks)}): {smoke_path}")


def _stratified_sample(tasks: list[RLTask], n: int, rng: random.Random) -> list[RLTask]:
    """Sample n tasks preserving family distribution."""
    from collections import defaultdict
    by_family: dict[str, list[RLTask]] = defaultdict(list)
    for t in tasks:
        by_family[t.family].append(t)
    n_per = max(1, n // len(by_family))
    sampled = []
    for family_tasks in by_family.values():
        sampled.extend(rng.sample(family_tasks, min(n_per, len(family_tasks))))
    rng.shuffle(sampled)
    return sampled[:n]


if __name__ == "__main__":
    main()
