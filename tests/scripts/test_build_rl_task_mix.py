"""Tests for the Slice 3 RL task mixer.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 3.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.rl.build_rl_task_mix import (
    DEFAULT_RATIO_POST_V03,
    LENGTH_BUCKETS,
    DatasetIngest,
    _length_bucket,
    _text_fingerprint,
    _word_count,
    build_mix,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _task(
    tid: str,
    *,
    family: str = "rewrite_repair",
    reward_profile: str = "rewrite_faithful_concise",
    input_text: str = "Staging recovered at 3 pm.",
    split: str = "train",
    mode: str = "rewrite",
    register: str = "casual",
) -> dict:
    return {
        "id": tid,
        "family": family,
        "domain": "slack",
        "mode": mode,
        "register_": register,
        "instruction": "Clean up this Slack update.",
        "input_text": input_text,
        "constraints": {
            "max_words": 20,
            "preserve_numbers": True,
            "preserve_entities": True,
        },
        "reward_profile": reward_profile,
        "trap_tags": ["wrapper_phrase"],
        "split": split,
        "required_facts": ["Staging recovered", "3 pm"],
    }


def _write(path: Path, rows: list[dict]) -> Path:
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return path


# ---------------------------------------------------------------------------
# Bucket helpers
# ---------------------------------------------------------------------------


def test_word_count_handles_punctuation_and_empties() -> None:
    assert _word_count("") == 0
    assert _word_count("hello world!") == 2
    assert _word_count("hi, there. it's me.") == 4


def test_length_bucket_endpoints() -> None:
    assert _length_bucket(0) == "short"
    assert _length_bucket(39) == "short"
    assert _length_bucket(40) == "medium"
    assert _length_bucket(119) == "medium"
    assert _length_bucket(120) == "long"
    assert _length_bucket(319) == "long"
    assert _length_bucket(320) == "xlong"
    assert _length_bucket(100_000) == "xlong"
    # Buckets cover the full positive range with no gap.
    for lo, hi in zip(
        [b[2] for b in LENGTH_BUCKETS[:-1]],
        [b[1] for b in LENGTH_BUCKETS[1:]],
        strict=True,
    ):
        assert lo == hi


def test_text_fingerprint_is_whitespace_and_case_insensitive() -> None:
    assert _text_fingerprint("Hello World") == _text_fingerprint("  hello   world  ")
    assert _text_fingerprint("a") != _text_fingerprint("b")


# ---------------------------------------------------------------------------
# build_mix: schema + enrichment
# ---------------------------------------------------------------------------


def test_build_mix_attaches_slice3_fields(tmp_path: Path) -> None:
    p01 = _write(tmp_path / "v01.jsonl", [_task("rl_v01_000001")])
    p02 = _write(tmp_path / "v02.jsonl", [_task("rl_v01_000099")])

    result = build_mix(
        [
            DatasetIngest(version="v01", path=p01),
            DatasetIngest(version="v02", path=p02),
        ],
        ratio={"v01": 0.3, "v02": 0.7},
    )
    assert len(result.rows) == 2
    for row in result.rows:
        assert row["dataset_version"] in {"v01", "v02"}
        assert row["length_bucket"] in {b[0] for b in LENGTH_BUCKETS}
        assert row["difficulty_bucket"] == "unknown"
        assert row["source_group"]
        assert isinstance(row["mix_weight"], float)


# ---------------------------------------------------------------------------
# build_mix: dedupe
# ---------------------------------------------------------------------------


def test_build_mix_drops_duplicate_ids(tmp_path: Path) -> None:
    p01 = _write(
        tmp_path / "v01.jsonl",
        [_task("rl_v01_000001"), _task("rl_v01_000001", input_text="different text here")],
    )
    p02 = _write(tmp_path / "v02.jsonl", [_task("rl_v01_000099")])

    ingests = [
        DatasetIngest(version="v01", path=p01),
        DatasetIngest(version="v02", path=p02),
    ]
    result = build_mix(ingests, ratio={"v01": 0.3, "v02": 0.7})

    assert len(result.rows) == 2
    v01_ingest = next(i for i in result.ingests if i.version == "v01")
    assert v01_ingest.n_dropped_dup_id == 1


def test_build_mix_drops_text_duplicates_within_dataset(tmp_path: Path) -> None:
    same_text = "Staging recovered at 3 pm after the STRIPE_WEBHOOK_SECRET fix."
    p01 = _write(
        tmp_path / "v01.jsonl",
        [
            _task("rl_v01_000001", input_text=same_text),
            _task("rl_v01_000002", input_text="  STAGING recovered AT 3 PM after the STRIPE_WEBHOOK_SECRET fix.   "),
        ],
    )
    p02 = _write(tmp_path / "v02.jsonl", [_task("rl_v01_000099", input_text=same_text)])

    ingests = [
        DatasetIngest(version="v01", path=p01),
        DatasetIngest(version="v02", path=p02),
    ]
    result = build_mix(ingests, ratio={"v01": 0.3, "v02": 0.7})

    # v01 second row drops (text dup within v01).
    # v02 keeps its copy: cross-dataset duplicates are allowed (acceptance §Slice 3
    # only requires no duplicate task IDs; identical sources across versions
    # legitimately appear).
    assert len(result.rows) == 2
    v01_ingest = next(i for i in result.ingests if i.version == "v01")
    assert v01_ingest.n_dropped_dup_text == 1


# ---------------------------------------------------------------------------
# build_mix: mix_weight math
# ---------------------------------------------------------------------------


def test_mix_weight_matches_target_ratio(tmp_path: Path) -> None:
    # 4 v01 rows + 16 v02 rows, asking for 50/50 \u2192 v01 rows should weigh 4\u00d7 v02 rows.
    v01_rows = [_task(f"rl_v01_{i:06d}", input_text=f"v01 row {i} text") for i in range(4)]
    v02_rows = [_task(f"rl_v01_{i+1000:06d}", input_text=f"v02 row {i} text") for i in range(16)]
    p01 = _write(tmp_path / "v01.jsonl", v01_rows)
    p02 = _write(tmp_path / "v02.jsonl", v02_rows)

    result = build_mix(
        [
            DatasetIngest(version="v01", path=p01),
            DatasetIngest(version="v02", path=p02),
        ],
        ratio={"v01": 0.5, "v02": 0.5},
    )
    w01 = next(r["mix_weight"] for r in result.rows if r["dataset_version"] == "v01")
    w02 = next(r["mix_weight"] for r in result.rows if r["dataset_version"] == "v02")
    assert w01 / w02 == pytest.approx(4.0)
    # Per-row weights average to ~1.0 (normalization).
    total = sum(r["mix_weight"] for r in result.rows)
    assert total == pytest.approx(len(result.rows))


def test_mix_weight_respects_pre_v03_default(tmp_path: Path) -> None:
    """When no v03 supplied, default ratio 30/70 means v02 weight \u2248 (70/30) * (v01_count/v02_count) * v01 weight."""
    p01 = _write(tmp_path / "v01.jsonl", [_task(f"rl_v01_{i:06d}", input_text=f"v01 row {i}") for i in range(10)])
    p02 = _write(tmp_path / "v02.jsonl", [_task(f"rl_v01_{i+1000:06d}", input_text=f"v02 row {i}") for i in range(10)])

    result = build_mix(
        [
            DatasetIngest(version="v01", path=p01),
            DatasetIngest(version="v02", path=p02),
        ],
        ratio={"v01": 0.30, "v02": 0.70},
    )
    w01 = next(r["mix_weight"] for r in result.rows if r["dataset_version"] == "v01")
    w02 = next(r["mix_weight"] for r in result.rows if r["dataset_version"] == "v02")
    # Equal counts \u2192 w02 / w01 \u2248 0.7 / 0.3
    assert w02 / w01 == pytest.approx(0.7 / 0.3, rel=1e-6)


# ---------------------------------------------------------------------------
# build_mix: v03 optional path
# ---------------------------------------------------------------------------


def test_v03_can_be_added_without_changing_downstream_schema(tmp_path: Path) -> None:
    v03_task = _task(
        "rl_v03_000001",
        mode="long_form_generate",
        family="technical_explain",
        reward_profile="technical_explain_natural",
        input_text="A v03 long-form task with different content.",
    )
    v03_task["task_author_model"] = "google/gemini-3.1-pro-preview"

    p01 = _write(tmp_path / "v01.jsonl", [_task("rl_v01_000001")])
    p02 = _write(tmp_path / "v02.jsonl", [_task("rl_v01_000099")])
    p03 = _write(tmp_path / "v03.jsonl", [v03_task])

    result = build_mix(
        [
            DatasetIngest(version="v01", path=p01),
            DatasetIngest(version="v02", path=p02),
            DatasetIngest(version="v03", path=p03),
        ],
        ratio=DEFAULT_RATIO_POST_V03,
    )
    versions = {r["dataset_version"] for r in result.rows}
    assert versions == {"v01", "v02", "v03"}
    # Common Slice 3 fields exist on every row, including v03.
    for row in result.rows:
        for field in (
            "dataset_version",
            "length_bucket",
            "difficulty_bucket",
            "source_group",
            "mix_weight",
        ):
            assert field in row, f"missing {field} on {row['id']}"


# ---------------------------------------------------------------------------
# build_mix: rejects bad schema
# ---------------------------------------------------------------------------


def test_invalid_rows_are_dropped(tmp_path: Path) -> None:
    bad = _task("rl_v01_000002")
    bad.pop("instruction")  # required field
    p01 = _write(tmp_path / "v01.jsonl", [_task("rl_v01_000001"), bad])
    p02 = _write(tmp_path / "v02.jsonl", [_task("rl_v01_000099")])

    result = build_mix(
        [
            DatasetIngest(version="v01", path=p01),
            DatasetIngest(version="v02", path=p02),
        ],
        ratio={"v01": 0.3, "v02": 0.7},
    )
    v01_ingest = next(i for i in result.ingests if i.version == "v01")
    assert v01_ingest.n_raw == 2
    assert v01_ingest.n_valid == 1
    assert v01_ingest.n_dropped_invalid == 1
