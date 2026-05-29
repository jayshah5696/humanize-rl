"""Split a seed JSONL into per-author (optionally per-mode) shards.

Used by slice 4 to give each (author, mode) pair a deterministic,
non-overlapping subset of seeds. Per-author shards are picked up by
`run_arka_pipeline.py --seed-path`.

Usage (author only):

    uv run scripts/data/build/split_seeds_by_author.py \
        --input data/raw/v03_writing_seeds_full.jsonl \
        --out-dir data/raw/v03_seeds_by_author \
        --total 1500 \
        --share google/gemini-3.1-pro-preview=0.05 \
        --share openai/gpt-5.4-mini=0.50 \
        --share google/gemini-3.1-flash-lite-preview=0.45 \
        --seed 99

Usage (author x mode):

    uv run scripts/data/build/split_seeds_by_author.py \
        --input data/raw/v03_writing_seeds_full.jsonl \
        --out-dir data/raw/v03_seeds_by_author \
        --total 1500 \
        --share google/gemini-3.1-pro-preview=0.05 \
        --share openai/gpt-5.4-mini=0.50 \
        --share google/gemini-3.1-flash-lite-preview=0.45 \
        --modes rewrite_humanize --modes compression --modes tone_shift \
        --modes expansion --modes long_form_generate --modes multi_constraint_compose
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import click


def _slug(author: str) -> str:
    return author.replace("/", "_").replace(":", "_")


@click.command(context_settings={"show_default": True})
@click.option("--input", "input_path",
              type=click.Path(exists=True, dir_okay=False, path_type=Path),
              required=True)
@click.option("--out-dir", type=click.Path(file_okay=False, path_type=Path),
              required=True)
@click.option("--total", type=int, default=None,
              help="Total seeds to use (cap before splitting). Default: all.")
@click.option("--share", multiple=True, required=True,
              help="`author=fraction`, repeatable. Must sum to ~1.0.")
@click.option("--modes", multiple=True, default=(),
              help="If set, further partition each author shard across N modes. "
                   "Output files become {author}__{mode}.jsonl.")
@click.option("--seed", type=int, default=99)
def main(
    input_path: Path, out_dir: Path, total: int | None,
    share: tuple[str, ...], modes: tuple[str, ...], seed: int,
) -> None:
    """Split seeds into per-author (and optional per-mode) shards."""
    shares: dict[str, float] = {}
    for s in share:
        if "=" not in s:
            raise click.ClickException(f"bad --share: {s}")
        k, _, v = s.partition("=")
        shares[k.strip()] = float(v)
    total_share = sum(shares.values())
    if abs(total_share - 1.0) > 0.01:
        raise click.ClickException(
            f"shares must sum to ~1.0, got {total_share}"
        )

    rows: list[str] = []
    with input_path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(line)
    rng = random.Random(seed)
    rng.shuffle(rows)
    if total:
        rows = rows[:total]
    click.echo(f"loaded {len(rows)} rows")

    out_dir.mkdir(parents=True, exist_ok=True)
    cursor = 0
    summary: dict[str, dict[str, int]] = {}
    for author, frac in shares.items():
        count = int(round(frac * len(rows)))
        shard = rows[cursor:cursor + count]
        cursor += count
        slug = _slug(author)
        if not modes:
            out_path = out_dir / f"{slug}.jsonl"
            with out_path.open("w") as fh:
                for r in shard:
                    fh.write(r + "\n")
            summary[author] = {"_total": len(shard)}
            click.echo(f"  {author}: {len(shard)} -> {out_path}")
        else:
            per_mode = len(shard) // len(modes)
            summary[author] = {}
            sub_cursor = 0
            for mode in modes:
                sub_shard = shard[sub_cursor:sub_cursor + per_mode]
                sub_cursor += per_mode
                out_path = out_dir / f"{slug}__{mode}.jsonl"
                with out_path.open("w") as fh:
                    for r in sub_shard:
                        fh.write(r + "\n")
                summary[author][mode] = len(sub_shard)
            click.echo(
                f"  {author}: {len(shard)} total -> {len(modes)} modes "
                f"x {per_mode} each"
            )
    leftover = len(rows) - cursor
    if leftover:
        click.echo(f"  (leftover unassigned: {leftover})")

    summary_path = out_dir / "_split_summary.json"
    summary_path.write_text(json.dumps({
        "total_input": len(rows),
        "seed": seed,
        "by_author": summary,
    }, indent=2))
    click.echo(f"summary -> {summary_path}")


if __name__ == "__main__":
    main()
