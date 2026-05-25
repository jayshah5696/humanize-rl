from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

AI_TELL_PHRASES: tuple[str, ...] = (
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
)

META_INSTRUCTION_PHRASES: tuple[str, ...] = (
    "chatgpt",
    "ai-generated",
    "ai generated",
    "ai-written",
    "ai written",
    "humanize this",
    "human wrote it",
    "bypass detector",
    "avoid ai detector",
)

COMMON_FAKE_NAMES: tuple[str, ...] = (
    "Sarah",
    "Marcus",
    "John",
    "Jane",
    "Alice",
    "Bob",
    "Charlie",
)

PLACEHOLDER_RE = re.compile(r"\[[A-Za-z][A-Za-z ]{0,40}\]")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}")
WHITESPACE_RE = re.compile(r"\s+")

SOURCE_LICENSES: dict[str, dict[str, str | bool]] = {
    "stream_b": {
        "license": "mixed-approved-public-sources",
        "release_eligible": True,
    },
    "safe_expand_raw": {
        "license": "project_synthetic",
        "release_eligible": True,
    },
    "safe_expand_3000_raw": {
        "license": "project_synthetic",
        "release_eligible": True,
    },
    "legacy_gold": {
        "license": "project_legacy_reviewed",
        "release_eligible": True,
    },
}


@dataclass(frozen=True)
class BuildConfig:
    input_path: Path
    output_dir: Path
    track_a_scorer_path: Path | None = None
    seed: int = 3407
    train_ratio: float = 0.9
    valid_ratio: float = 0.05
    smoke_train_size: int = 100
    smoke_valid_size: int = 20
    pilot_train_size: int = 500
    pilot_valid_size: int = 50
    max_ai_probability: float = 0.85
    min_response_chars: int = 20
    max_response_chars: int = 2500
    min_instruction_chars: int = 10
    max_instruction_chars: int = 3000
    write_rejected: bool = True


@dataclass
class Rejection:
    index: int
    reasons: list[str]
    row: dict[str, Any]


@dataclass
class BuildResult:
    accepted_rows: list[dict[str, Any]]
    rejected_rows: list[Rejection]
    split_counts: dict[str, int]
    report: dict[str, Any]


@dataclass(frozen=True)
class NormalizedPair:
    instruction: str
    response: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        raise FileNotFoundError(f"Input JSONL does not exist: {path}")
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, instruction: str, response: str) -> str:
    payload = f"{instruction}\n---\n{response}".encode()
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:16]}"


def normalize_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def extract_assistant_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        role = str(message.get("role", ""))
        if role in {"assistant", "model"}:
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "\n".join(parts)
    return ""


def extract_user_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        role = str(message.get("role", ""))
        if role == "user":
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: list[str] = []
                for item in content:
                    if isinstance(item, dict) and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                return "\n".join(parts)
    return ""


def normalize_row(row: dict[str, Any]) -> NormalizedPair:
    if "messages" in row and isinstance(row["messages"], list):
        instruction = extract_user_from_messages(row["messages"])
        response = extract_assistant_from_messages(row["messages"])
    else:
        instruction = str(
            row.get("instruction") or row.get("prompt") or row.get("input") or ""
        )
        response = str(
            row.get("response") or row.get("output") or row.get("completion") or ""
        )

    metadata = {key: value for key, value in row.items() if key not in {"messages"}}
    return NormalizedPair(
        instruction=instruction.strip(),
        response=response.strip(),
        metadata=metadata,
    )


def infer_domain(instruction: str, metadata: dict[str, Any]) -> str:
    domain = metadata.get("domain")
    if isinstance(domain, str) and domain:
        return domain
    text = instruction.lower()
    if "slack" in text or "dm" in text or "channel" in text:
        return "chat"
    if "email" in text:
        return "email"
    if "technical" in text or "api" in text or "database" in text or "docker" in text:
        return "technical"
    if "linkedin" in text or "post" in text:
        return "social_media"
    return "general"


def infer_task_type(instruction: str, metadata: dict[str, Any]) -> str:
    task_type = metadata.get("task_type")
    if isinstance(task_type, str) and task_type:
        return task_type
    lower = instruction.lower()
    if "slack" in lower:
        return "slack_chat"
    if "email" in lower:
        return "email"
    if lower.startswith(
        ("rewrite", "clean up", "fix", "tighten", "shorten", "condense")
    ):
        return "rewrite_or_edit"
    return "direct_generation"


def infer_mode(instruction: str, metadata: dict[str, Any]) -> str:
    mode = metadata.get("mode")
    if isinstance(mode, str) and mode:
        return mode
    lower = instruction.lower().strip()
    if lower.startswith(
        ("rewrite", "clean up", "fix", "tighten", "shorten", "condense", "make this")
    ):
        return "rewrite_humanize"
    return "direct_generation"


def source_metadata(source: str | None) -> dict[str, str | bool]:
    if source and source in SOURCE_LICENSES:
        return SOURCE_LICENSES[source]
    return {"license": "unknown", "release_eligible": False}


def contains_name(text: str, name: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(name)}\b", flags=re.IGNORECASE)
    return bool(pattern.search(text))


def has_fake_name_issue(instruction: str, response: str) -> list[str]:
    flags: list[str] = []
    lower_instruction = instruction.lower()
    asks_for_missing_cover = (
        "who will cover" in lower_instruction
        or "cover my projects" in lower_instruction
    )
    rewrite_like = lower_instruction.startswith(
        ("rewrite", "clean up", "shorten", "make this", "fix", "condense", "tighten")
    )
    quoted = "'" in instruction or '"' in instruction

    for name in COMMON_FAKE_NAMES:
        name_in_response = contains_name(response, name)
        name_in_instruction = contains_name(instruction, name)
        if name_in_response and not name_in_instruction and (rewrite_like or quoted):
            flags.append(f"possible_fake_name:{name}")
        if (
            asks_for_missing_cover
            and name_in_response
            and not name_in_instruction
            and not PLACEHOLDER_RE.search(response)
        ):
            flags.append(f"missing_placeholder_for_unspecified_cover:{name}")
    return flags


def rejection_reasons(pair: NormalizedPair, config: BuildConfig) -> list[str]:
    reasons: list[str] = []
    instruction = pair.instruction
    response = pair.response
    lower_instruction = instruction.lower()
    lower_response = response.lower()

    if len(instruction) < config.min_instruction_chars:
        reasons.append("instruction_too_short")
    if len(instruction) > config.max_instruction_chars:
        reasons.append("instruction_too_long")
    if len(response) < config.min_response_chars:
        reasons.append("response_too_short")
    if len(response) > config.max_response_chars:
        reasons.append("response_too_long")
    if not instruction:
        reasons.append("missing_instruction")
    if not response:
        reasons.append("missing_response")

    for phrase in AI_TELL_PHRASES:
        if phrase in lower_response:
            reasons.append(f"response_ai_tell:{phrase}")

    for phrase in META_INSTRUCTION_PHRASES:
        if phrase in lower_instruction:
            reasons.append(f"meta_instruction:{phrase}")

    if EMAIL_RE.search(instruction) or EMAIL_RE.search(response):
        reasons.append("possible_email_pii")
    if PHONE_RE.search(instruction) or PHONE_RE.search(response):
        reasons.append("possible_phone_pii")

    quality_judge = pair.metadata.get("quality_judge")
    if isinstance(quality_judge, dict):
        if quality_judge.get("keep") is False:
            reasons.append("quality_judge_reject")
        for key in (
            "fact_preservation",
            "instruction_following",
            "unsupported_details",
            "ai_tells",
        ):
            if quality_judge.get(key) == "fail":
                reasons.append(f"quality_judge_{key}_fail")
        naturalness = quality_judge.get("naturalness")
        if isinstance(naturalness, int | float) and naturalness < 4:
            reasons.append("quality_judge_naturalness_lt_4")

    reasons.extend(has_fake_name_issue(instruction, response))
    return reasons


def load_track_a_scorer(path: Path | None) -> Any | None:
    if path is None or not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def add_scorer_fields(
    rows: list[dict[str, Any]], scorer: Any | None, max_ai_probability: float
) -> list[Rejection]:
    if scorer is None or not rows:
        return []
    texts = [str(row["messages"][-1]["content"]) for row in rows]
    probabilities = scorer.predict_binary(texts)
    rejected: list[Rejection] = []
    for index, (row, probability) in enumerate(zip(rows, probabilities, strict=True)):
        quality = row.setdefault("quality", {})
        quality["track_a_ai_probability"] = float(probability)
        if float(probability) > max_ai_probability:
            rejected.append(
                Rejection(
                    index=index, reasons=["track_a_ai_probability_too_high"], row=row
                )
            )
    return rejected


def split_rows(
    rows: list[dict[str, Any]], config: BuildConfig
) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        metadata = row.get("metadata", {})
        source = str(metadata.get("source", "unknown"))
        instruction = normalize_space(str(row["messages"][0]["content"])).lower()
        group_digest = hashlib.sha256(f"{source}:{instruction}".encode()).hexdigest()[
            :16
        ]
        grouped[group_digest].append(row)

    groups = list(grouped.values())
    rng = random.Random(config.seed)
    rng.shuffle(groups)

    total = sum(len(group) for group in groups)
    train_target = int(total * config.train_ratio)
    valid_target = int(total * config.valid_ratio)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "valid": [], "test": []}
    for group in groups:
        if len(splits["train"]) < train_target:
            target = "train"
        elif len(splits["valid"]) < valid_target:
            target = "valid"
        else:
            target = "test"
        for row in group:
            row["split"] = target
        splits[target].extend(group)

    return splits


def build_dataset(config: BuildConfig) -> BuildResult:
    raw_rows = load_jsonl(config.input_path)
    accepted: list[dict[str, Any]] = []
    rejected: list[Rejection] = []
    seen: set[tuple[str, str]] = set()

    for index, row in enumerate(raw_rows):
        pair = normalize_row(row)
        reasons = rejection_reasons(pair, config)
        key = (
            normalize_space(pair.instruction).lower(),
            normalize_space(pair.response).lower(),
        )
        if key in seen:
            reasons.append("exact_duplicate")
        if reasons:
            rejected.append(Rejection(index=index, reasons=reasons, row=row))
            continue
        seen.add(key)

        source = pair.metadata.get("source")
        source = source if isinstance(source, str) else None
        license_info = source_metadata(source)
        domain = infer_domain(pair.instruction, pair.metadata)
        task_type = infer_task_type(pair.instruction, pair.metadata)
        mode = infer_mode(pair.instruction, pair.metadata)
        quality_judge = pair.metadata.get("quality_judge")
        quality: dict[str, Any] = {}
        if isinstance(quality_judge, dict):
            quality["judge_keep"] = quality_judge.get("keep")
            quality["judge_naturalness"] = quality_judge.get("naturalness")
            quality["judge_reason"] = quality_judge.get("reason")

        record = {
            "id": stable_id("sft_v04", pair.instruction, pair.response),
            "messages": [
                {"role": "user", "content": pair.instruction},
                {"role": "assistant", "content": pair.response},
            ],
            "domain": domain,
            "task_type": task_type,
            "mode": mode,
            "source": source or "unknown",
            "license": license_info["license"],
            "release_eligible": bool(license_info["release_eligible"]),
            "quality": quality,
            "metadata": {
                "input_index": index,
                "source": source or "unknown",
            },
        }
        accepted.append(record)

    scorer = load_track_a_scorer(config.track_a_scorer_path)
    scorer_rejections = add_scorer_fields(accepted, scorer, config.max_ai_probability)
    if scorer_rejections:
        rejected.extend(scorer_rejections)
        rejected_ids = {item.row["id"] for item in scorer_rejections}
        accepted = [row for row in accepted if row["id"] not in rejected_ids]

    splits = split_rows(accepted, config)
    split_counts = {key: len(value) for key, value in splits.items()}

    train = splits["train"]
    valid = splits["valid"]
    smoke_train = train[: config.smoke_train_size]
    smoke_valid = valid[: config.smoke_valid_size]
    pilot_train = train[: config.pilot_train_size]
    pilot_valid = valid[: config.pilot_valid_size]

    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(config.output_dir / "train.jsonl", train)
    write_jsonl(config.output_dir / "valid.jsonl", valid)
    write_jsonl(config.output_dir / "test.jsonl", splits["test"])
    write_jsonl(config.output_dir / "smoke_train.jsonl", smoke_train)
    write_jsonl(config.output_dir / "smoke_valid.jsonl", smoke_valid)
    write_jsonl(config.output_dir / "pilot_train.jsonl", pilot_train)
    write_jsonl(config.output_dir / "pilot_valid.jsonl", pilot_valid)

    if config.write_rejected:
        write_jsonl(
            config.output_dir / "rejected.jsonl",
            [
                {"input_index": item.index, "reasons": item.reasons, "row": item.row}
                for item in rejected
            ],
        )

    report = make_report(raw_rows, accepted, rejected, split_counts, config)
    (config.output_dir / "manifest.json").write_text(
        json.dumps(report["manifest"], indent=2, ensure_ascii=False)
    )
    (config.output_dir / "quality_report.md").write_text(render_quality_report(report))

    return BuildResult(
        accepted_rows=accepted,
        rejected_rows=rejected,
        split_counts=split_counts,
        report=report,
    )


def phrase_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for phrase in AI_TELL_PHRASES:
        count = 0
        for row in rows:
            response = str(row["messages"][-1]["content"]).lower()
            if phrase in response:
                count += 1
        if count:
            counts[phrase] = count
    return counts


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    sorted_values = sorted(values)
    index = min(len(sorted_values) - 1, round((pct / 100) * (len(sorted_values) - 1)))
    return sorted_values[index]


def make_report(
    raw_rows: list[dict[str, Any]],
    accepted: list[dict[str, Any]],
    rejected: list[Rejection],
    split_counts: dict[str, int],
    config: BuildConfig,
) -> dict[str, Any]:
    responses = [str(row["messages"][-1]["content"]) for row in accepted]
    response_lengths = [len(response) for response in responses]
    rejection_counter = Counter(reason for item in rejected for reason in item.reasons)
    ai_probabilities = [
        row.get("quality", {}).get("track_a_ai_probability") for row in accepted
    ]
    ai_prob_numbers = [
        float(value) for value in ai_probabilities if isinstance(value, int | float)
    ]

    manifest = {
        "input_path": str(config.input_path),
        "input_sha256": file_sha256(config.input_path),
        "track_a_scorer_path": str(config.track_a_scorer_path)
        if config.track_a_scorer_path
        else None,
        "track_a_scorer_sha256": file_sha256(config.track_a_scorer_path)
        if config.track_a_scorer_path and config.track_a_scorer_path.exists()
        else None,
        "seed": config.seed,
        "raw_rows": len(raw_rows),
        "accepted_rows": len(accepted),
        "rejected_rows": len(rejected),
        "split_counts": split_counts,
        "smoke_train_size": min(config.smoke_train_size, split_counts.get("train", 0)),
        "smoke_valid_size": min(config.smoke_valid_size, split_counts.get("valid", 0)),
        "pilot_train_size": min(config.pilot_train_size, split_counts.get("train", 0)),
        "pilot_valid_size": min(config.pilot_valid_size, split_counts.get("valid", 0)),
    }

    return {
        "manifest": manifest,
        "domain_counts": Counter(row["domain"] for row in accepted),
        "task_type_counts": Counter(row["task_type"] for row in accepted),
        "mode_counts": Counter(row["mode"] for row in accepted),
        "source_counts": Counter(row["source"] for row in accepted),
        "license_counts": Counter(row["license"] for row in accepted),
        "release_counts": Counter(str(row["release_eligible"]) for row in accepted),
        "rejection_counts": rejection_counter,
        "phrase_counts": phrase_counts(accepted),
        "response_length": {
            "min": min(response_lengths) if response_lengths else 0,
            "p50": percentile(response_lengths, 50),
            "p90": percentile(response_lengths, 90),
            "p95": percentile(response_lengths, 95),
            "max": max(response_lengths) if response_lengths else 0,
        },
        "track_a_ai_probability": {
            "count": len(ai_prob_numbers),
            "min": min(ai_prob_numbers) if ai_prob_numbers else None,
            "p50": percentile([round(value * 10000) for value in ai_prob_numbers], 50)
            / 10000
            if ai_prob_numbers
            else None,
            "p95": percentile([round(value * 10000) for value in ai_prob_numbers], 95)
            / 10000
            if ai_prob_numbers
            else None,
            "max": max(ai_prob_numbers) if ai_prob_numbers else None,
        },
        "samples": accepted[:10],
    }


def counter_table(counter: Counter[str]) -> str:
    if not counter:
        return "_None_\n"
    lines = ["| value | count |", "|---|---:|"]
    for key, value in counter.most_common():
        lines.append(f"| {key} | {value} |")
    return "\n".join(lines) + "\n"


def render_quality_report(report: dict[str, Any]) -> str:
    manifest = report["manifest"]
    lines = [
        "# Gemma 4 E2B SFT Dataset Quality Report",
        "",
        "## Summary",
        "",
        f"- Raw rows: {manifest['raw_rows']}",
        f"- Accepted rows: {manifest['accepted_rows']}",
        f"- Rejected rows: {manifest['rejected_rows']}",
        f"- Split counts: {manifest['split_counts']}",
        f"- Input SHA256: `{manifest['input_sha256']}`",
        f"- Track A scorer SHA256: `{manifest['track_a_scorer_sha256']}`",
        "",
        "## Response lengths",
        "",
        "```json",
        json.dumps(report["response_length"], indent=2),
        "```",
        "",
        "## Track A AI probability",
        "",
        "```json",
        json.dumps(report["track_a_ai_probability"], indent=2),
        "```",
        "",
        "## Domains",
        "",
        counter_table(report["domain_counts"]),
        "## Task types",
        "",
        counter_table(report["task_type_counts"]),
        "## Modes",
        "",
        counter_table(report["mode_counts"]),
        "## Sources",
        "",
        counter_table(report["source_counts"]),
        "## Licenses",
        "",
        counter_table(report["license_counts"]),
        "## Release eligibility",
        "",
        counter_table(report["release_counts"]),
        "## Rejection reasons",
        "",
        counter_table(report["rejection_counts"]),
        "## Remaining AI-tell phrase counts",
        "",
        json.dumps(report["phrase_counts"], indent=2),
        "",
        "## Accepted samples",
        "",
    ]
    for sample in report["samples"]:
        messages = sample["messages"]
        lines.extend(
            [
                f"### {sample['id']}",
                "",
                f"**Domain:** {sample['domain']}  ",
                f"**Mode:** {sample['mode']}  ",
                f"**Source:** {sample['source']}",
                "",
                "**User**",
                "",
                str(messages[0]["content"]),
                "",
                "**Assistant**",
                "",
                str(messages[1]["content"]),
                "",
            ]
        )
    return "\n".join(lines)


def convert_to_mlx_smoke(output_dir: Path, mlx_dir: Path) -> None:
    mlx_dir.mkdir(parents=True, exist_ok=True)
    mapping = {
        "smoke_train.jsonl": "train.jsonl",
        "smoke_valid.jsonl": "valid.jsonl",
        "test.jsonl": "test.jsonl",
    }
    for source_name, target_name in mapping.items():
        source_path = output_dir / source_name
        if source_path.exists():
            rows = load_jsonl(source_path)
            write_jsonl(
                mlx_dir / target_name, [{"messages": row["messages"]} for row in rows]
            )
