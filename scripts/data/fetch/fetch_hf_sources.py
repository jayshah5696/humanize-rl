"""Generic HF dataset fetcher driven by a YAML manifest.

Replaces the per-stream `fetch_v04_*.py` scripts (which move to
`scripts/archive/` once v03 lands). See plan §7.1.

Manifest schema (configs/v03/sources.yaml):

    defaults:
      out_schema: rl_seed
      dedupe_on: input_text     # or "instruction"
      min_words: 20

    sources:
      - hf: <repo>
        split: train
        limit: 500
        config: <optional config>
        sample_seed: <optional int>
        license: <free text>
        filter: { <col>: <val> }   # equality filter, ANDed
        map:
          instruction: <col>      # output column ← source column
          input_text: <col>
          domain: <col>
        tags: [seed, source, <mode>, ...]

Output row schema (rl_seed) — arka seed_source requires `instruction` AND
`response`, so we always populate both. The semantic mapping is:

    {
      "instruction": str,        # the seed instruction (or a placeholder)
      "response":    str,        # the source text (what humans wrote)
      "input_text":  str | None, # alias of response, kept for clarity
      "domain": str | None,
      "source_dataset": str,
      "source_row_id": str,
      "tags": [str],
      "license": str
    }

Mode prompts read `payload.response` (the source text) as `{input_text}`.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import click
import yaml


def _word_count(text: str | None) -> int:
    return len((text or "").split())


def _row_id(source: str, payload: dict[str, Any]) -> str:
    h = hashlib.sha1(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:12]
    return f"{source}#{h}"


def _resolve_value(row: dict[str, Any], spec_value: str) -> Any:
    """Resolve a column spec.

    Supported forms:
      * "col_name"                         — plain lookup
      * "first_user_message:col_name"      — col is a list of {role, content}
        dicts; return content of first role=user message.
      * "first_assistant_message:col_name" — same but role=assistant
      * "join_messages:col_name"           — concat all message contents
    """
    if ":" in spec_value:
        op, _, col = spec_value.partition(":")
    else:
        op, col = "", spec_value
    val = row.get(col)
    if op == "first_user_message" and isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get("role") == "user":
                return m.get("content")
        return None
    if op == "first_assistant_message" and isinstance(val, list):
        for m in val:
            if isinstance(m, dict) and m.get("role") == "assistant":
                return m.get("content")
        return None
    if op == "join_messages" and isinstance(val, list):
        parts = []
        for m in val:
            if isinstance(m, dict) and isinstance(m.get("content"), str):
                parts.append(m["content"])
        return "\n\n".join(parts) if parts else None
    return val


def _project(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    mapping = spec.get("map", {})
    projected: dict[str, Any] = {}
    for out_key, src_key in mapping.items():
        val = _resolve_value(row, src_key)
        if isinstance(val, str):
            val = val.strip()
        projected[out_key] = val
    projected.setdefault("instruction", None)
    projected.setdefault("input_text", None)
    projected.setdefault("domain", None)
    # arka seed_source REQUIRES both `instruction` and `response` strings.
    # Map our semantic fields onto them: instruction stays, response holds
    # the source text (= input_text for downstream mode prompts).
    inst = (projected["instruction"] or "").strip()
    text = (projected["input_text"] or "").strip()
    # Important: arka's near-dedup tokenizes payload.instruction. If we let
    # every seed share the same stub here, dedup collapses the whole set.
    # Put a unique snippet (first 200 chars of the source) into instruction
    # so dedup sees per-row variation.
    canonical = text or inst
    projected["instruction"] = canonical[:200] if canonical else "(no seed)"
    projected["response"] = canonical
    projected["input_text"] = canonical
    projected["source_dataset"] = spec["hf"]
    projected["source_row_id"] = _row_id(spec["hf"], row)
    projected["tags"] = list(spec.get("tags", []))
    projected["license"] = spec.get("license", "unknown")
    return projected


def _matches_filter(row: dict[str, Any], filter_spec: dict[str, Any]) -> bool:
    for col, expected in filter_spec.items():
        if row.get(col) != expected:
            return False
    return True


def _stream_source(
    spec: dict[str, Any], min_words: int, scan_cap: int,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kw: dict[str, Any] = {"split": spec.get("split", "train"), "streaming": True}
    if "config" in spec:
        kw["name"] = spec["config"]

    rng = random.Random(spec.get("sample_seed", 42))
    limit = int(spec.get("limit", 500))
    filter_spec = spec.get("filter") or {}
    collected: list[dict[str, Any]] = []
    seen = 0

    click.echo(f"  → loading {spec['hf']} (limit={limit})")
    ds = load_dataset(spec["hf"], **kw)

    hard_cap = min(scan_cap, max(limit * 20, 5000))
    for row in ds:
        seen += 1
        if filter_spec and not _matches_filter(row, filter_spec):
            if seen >= hard_cap:
                break
            continue
        projected = _project(row, spec)
        text_for_len = projected.get("input_text") or projected.get("instruction") or ""
        if _word_count(text_for_len) < min_words:
            if seen >= hard_cap:
                break
            continue
        collected.append(projected)
        if len(collected) >= limit:
            break
        if seen >= hard_cap:
            click.echo(f"    hit scan_cap={hard_cap}")
            break

    if "sample_seed" in spec:
        rng.shuffle(collected)
    click.echo(f"    kept {len(collected)} / scanned {seen}")
    return collected


def _dedupe(rows: list[dict[str, Any]], on: str) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (row.get(on) or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


@click.command(context_settings={"show_default": True})
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True, help="Sources YAML manifest."
)
@click.option(
    "--out", type=click.Path(dir_okay=False, path_type=Path),
    required=True, help="Output JSONL path."
)
@click.option(
    "--dedupe-on", type=click.Choice(["instruction", "input_text", "none"]),
    default=None, help="Override dedupe field. 'none' disables dedupe."
)
@click.option(
    "--min-words", type=int, default=None,
    help="Override default min_words filter."
)
@click.option(
    "--normalize-schema", type=click.Choice(["rl_seed"]), default="rl_seed",
    help="Output schema (only rl_seed today)."
)
@click.option(
    "--limit", type=int, default=None,
    help="Override per-source `limit` (smoke runs)."
)
@click.option(
    "--scan-cap", type=int, default=20000,
    help="Max rows to stream per source (defends against suri-style hangs)."
)
def main(
    config: Path, out: Path, dedupe_on: str | None,
    min_words: int | None, normalize_schema: str, limit: int | None,
    scan_cap: int,
) -> None:
    """Fetch HF datasets per manifest, normalize, dedupe, write JSONL."""
    manifest = yaml.safe_load(config.read_text())
    defaults = manifest.get("defaults", {})
    effective_dedupe = dedupe_on or defaults.get("dedupe_on", "input_text")
    effective_min_words = min_words if min_words is not None else int(
        defaults.get("min_words", 20)
    )

    all_rows: list[dict[str, Any]] = []
    for spec in manifest["sources"]:
        if limit is not None:
            spec = {**spec, "limit": limit}
        try:
            all_rows.extend(
                _stream_source(spec, effective_min_words, scan_cap)
            )
        except Exception as exc:  # noqa: BLE001 — surface and continue
            click.secho(f"  ! {spec['hf']} failed: {exc}", fg="red")

    click.echo(f"raw rows: {len(all_rows)}")
    if effective_dedupe != "none":
        all_rows = _dedupe(all_rows, effective_dedupe)
        click.echo(f"after dedupe on `{effective_dedupe}`: {len(all_rows)}")

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in all_rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    click.secho(f"wrote {len(all_rows)} rows → {out}", fg="green")


if __name__ == "__main__":
    main()
