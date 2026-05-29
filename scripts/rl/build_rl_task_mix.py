#!/usr/bin/env python3
"""Build a normalized v01 + v02 (+ v03 draft) RL task mix.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 3.

For every input task we:
  * validate against ``RLTask`` (drops malformed rows with a warning),
  * tag ``dataset_version`` (v01/v02/v03),
  * tag ``length_bucket`` from ``input_text`` word count,
  * default ``difficulty_bucket = "unknown"`` (Slice 4 fills this in),
  * preserve ``source_group`` (or fill from input filename),
  * compute ``mix_weight`` per the configured ratio so a sampler with
    weighted sampling will hit the requested per-dataset share.

Dedupes by ``id`` first, then by normalized ``input_text`` hash within
each ``dataset_version``. Writes:

  * the merged ``.jsonl`` (one row per task with the new fields), and
  * a ``_summary.json`` with counts, family/profile/version mix, length
    bucket distribution, dedupe stats, and the ratio actually achieved.

Run::

  rtk uv run python scripts/rl/build_rl_task_mix.py \\
    --v01 data/rl/humanize_tasks_v01_smoke.jsonl \\
    --v02 data/rl/humanize_tasks_v02.jsonl \\
    --output data/rl/humanize_tasks_rl_mix_v1.jsonl \\
    --summary outputs/rl_mix/humanize_tasks_rl_mix_v1_summary.json
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import click
from pydantic import ValidationError

from humanize_rl.reward.tasks import RLTask

LENGTH_BUCKETS: tuple[tuple[str, int, int], ...] = (
    # (name, lo_inclusive, hi_exclusive)  \u2014 measured on input_text word count
    ("short", 0, 40),
    ("medium", 40, 120),
    ("long", 120, 320),
    ("xlong", 320, 10_000),
)

DEFAULT_RATIO_PRE_V03 = {"v01": 0.15, "v02": 0.35, "v03": 0.50}
DEFAULT_RATIO_POST_V03 = {"v01": 0.10, "v02": 0.20, "v03": 0.70}

_WORD_RE = re.compile(r"\b[\w'-]+\b")
_WS_RE = re.compile(r"\s+")


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text or ""))


def _length_bucket(word_count: int) -> str:
    for name, lo, hi in LENGTH_BUCKETS:
        if lo <= word_count < hi:
            return name
    return LENGTH_BUCKETS[-1][0]


def _text_fingerprint(text: str) -> str:
    norm = _WS_RE.sub(" ", (text or "").strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Mixer
# ---------------------------------------------------------------------------


@dataclass
class DatasetIngest:
    """One input dataset slice."""

    version: str
    path: Path
    n_raw: int = 0
    n_valid: int = 0
    n_dropped_invalid: int = 0
    n_dropped_dup_id: int = 0
    n_dropped_dup_text: int = 0


@dataclass
class MixResult:
    rows: list[dict] = field(default_factory=list)
    ingests: list[DatasetIngest] = field(default_factory=list)


def _iter_jsonl(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _enrich(
    raw: dict, dataset_version: str, default_source_group: str
) -> dict:
    """Validate + add the Slice 3 fields. Raises ValidationError on bad rows."""
    task = RLTask.model_validate(raw)
    out = task.model_dump(by_alias=True, exclude_none=False)
    out["dataset_version"] = dataset_version
    out["length_bucket"] = _length_bucket(_word_count(task.input_text))
    out["difficulty_bucket"] = "unknown"  # Slice 4 will fill this in.
    if not out.get("source_group"):
        out["source_group"] = default_source_group
    # mix_weight is added in a second pass once we know counts.
    out["mix_weight"] = 1.0
    return out


def _ingest_one(
    ingest: DatasetIngest, seen_ids: set[str], seen_texts: set[str]
) -> list[dict]:
    rows: list[dict] = []
    default_source_group = f"{ingest.version}:{ingest.path.stem}"
    for raw in _iter_jsonl(ingest.path):
        ingest.n_raw += 1
        try:
            enriched = _enrich(raw, ingest.version, default_source_group)
        except ValidationError as e:
            ingest.n_dropped_invalid += 1
            click.echo(
                f"[warn] {ingest.path.name}: dropping row id={raw.get('id', '?')}: "
                f"{e.errors()[0]['msg'] if e.errors() else e}",
                err=True,
            )
            continue
        tid = enriched["id"]
        if tid in seen_ids:
            ingest.n_dropped_dup_id += 1
            continue
        fp = _text_fingerprint(enriched.get("input_text", ""))
        text_key = f"{ingest.version}:{fp}"
        if text_key in seen_texts:
            ingest.n_dropped_dup_text += 1
            continue
        seen_ids.add(tid)
        seen_texts.add(text_key)
        ingest.n_valid += 1
        rows.append(enriched)
    return rows


def build_mix(ingests: list[DatasetIngest], ratio: dict[str, float]) -> MixResult:
    """Ingest, dedupe, attach ``mix_weight`` based on ``ratio``."""
    seen_ids: set[str] = set()
    seen_texts: set[str] = set()
    all_rows: list[dict] = []
    for ingest in ingests:
        all_rows.extend(_ingest_one(ingest, seen_ids, seen_texts))

    # Compute mix_weight so that, under proportional weighted sampling,
    # each version's expected share equals ratio[version].
    counts = Counter(r["dataset_version"] for r in all_rows)
    weights: dict[str, float] = {}
    for version, count in counts.items():
        target = ratio.get(version, 0.0)
        if count == 0 or target <= 0:
            weights[version] = 0.0
        else:
            weights[version] = target / count
    # Normalize so the per-row weights average to 1.0 (numerical hygiene).
    total = sum(weights[v] * counts[v] for v in counts)
    if total > 0:
        scale = sum(counts.values()) / total
        weights = {v: w * scale for v, w in weights.items()}
    for row in all_rows:
        row["mix_weight"] = weights.get(row["dataset_version"], 0.0)

    return MixResult(rows=all_rows, ingests=ingests)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _summarize(result: MixResult, ratio: dict[str, float]) -> dict:
    by_version = Counter(r["dataset_version"] for r in result.rows)
    by_family = Counter(r["family"] for r in result.rows)
    by_profile = Counter(r["reward_profile"] for r in result.rows)
    by_length = Counter(r["length_bucket"] for r in result.rows)
    by_split = Counter(r["split"] for r in result.rows)
    by_mode = Counter(r["mode"] for r in result.rows)
    achieved_ratio = {
        v: by_version[v] / len(result.rows) for v in by_version
    } if result.rows else {}

    return {
        "n_rows": len(result.rows),
        "target_ratio": ratio,
        "achieved_ratio": achieved_ratio,
        "by_dataset_version": dict(by_version),
        "by_family": dict(by_family),
        "by_reward_profile": dict(by_profile),
        "by_length_bucket": dict(by_length),
        "by_split": dict(by_split),
        "by_mode": dict(by_mode),
        "ingest": [
            {
                "version": i.version,
                "path": str(i.path),
                "n_raw": i.n_raw,
                "n_valid": i.n_valid,
                "n_dropped_invalid": i.n_dropped_invalid,
                "n_dropped_dup_id": i.n_dropped_dup_id,
                "n_dropped_dup_text": i.n_dropped_dup_text,
            }
            for i in result.ingests
        ],
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_ratio(ctx, param, value: str | None) -> dict[str, float] | None:
    if value is None:
        return None
    parts = {}
    for chunk in value.split(","):
        k, _, v = chunk.partition("=")
        if not _:
            raise click.BadParameter("expected v01=0.15,v02=0.35,v03=0.50")
        parts[k.strip()] = float(v)
    return parts


@click.command()
@click.option("--v01", "v01_path", type=click.Path(exists=True, path_type=Path),
              default="data/rl/humanize_tasks_v01_smoke.jsonl", show_default=True)
@click.option("--v02", "v02_path", type=click.Path(exists=True, path_type=Path),
              default="data/rl/humanize_tasks_v02.jsonl", show_default=True)
@click.option("--v03", "v03_path", type=click.Path(exists=True, path_type=Path),
              default=None, help="Optional v03 draft file (e.g. v03_slice3_tasks.jsonl)")
@click.option("--output", "output_path", type=click.Path(path_type=Path), required=True)
@click.option("--summary", "summary_path", type=click.Path(path_type=Path), required=True)
@click.option(
    "--ratio",
    callback=_parse_ratio,
    default=None,
    help="Per-version mix ratio, e.g. 'v01=0.15,v02=0.35,v03=0.50'. "
         "Defaults adapt to whether --v03 is supplied.",
)
def main(
    v01_path: Path,
    v02_path: Path,
    v03_path: Path | None,
    output_path: Path,
    summary_path: Path,
    ratio: dict[str, float] | None,
) -> None:
    """Build the normalized v01/v02(/v03) RL task mix."""
    ingests = [
        DatasetIngest(version="v01", path=v01_path),
        DatasetIngest(version="v02", path=v02_path),
    ]
    if v03_path is not None:
        ingests.append(DatasetIngest(version="v03", path=v03_path))

    if ratio is None:
        ratio = (
            DEFAULT_RATIO_POST_V03 if v03_path is not None else
            # Pre-v03: re-normalize 15/35 to 30/70 over only v01+v02.
            {"v01": 0.30, "v02": 0.70}
        )
    # Validate ratio sums to ~1.0
    s = sum(ratio.values())
    if abs(s - 1.0) > 1e-3:
        raise click.BadParameter(f"ratio must sum to 1.0 (got {s:.4f})")

    result = build_mix(ingests, ratio)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        for row in result.rows:
            fh.write(json.dumps(row) + "\n")
    summary = _summarize(result, ratio)
    summary_path.write_text(json.dumps(summary, indent=2))

    click.echo(f"Wrote {output_path}  ({summary['n_rows']} rows)")
    click.echo(f"Wrote {summary_path}")
    click.echo(f"Achieved ratio: {summary['achieved_ratio']}")
    for ingest in result.ingests:
        click.echo(
            f"  {ingest.version}: raw={ingest.n_raw} valid={ingest.n_valid} "
            f"dup_id={ingest.n_dropped_dup_id} dup_text={ingest.n_dropped_dup_text} "
            f"invalid={ingest.n_dropped_invalid}"
        )


if __name__ == "__main__":
    main()
