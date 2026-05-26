#!/usr/bin/env python3
"""Reusable JSONL utility for common data operations."""

from __future__ import annotations

import json
import random
from pathlib import Path

import click


@click.group()
def cli() -> None:
    """JSONL data manipulation tool."""
    pass


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
def count(input_file: str) -> None:
    """Print the number of rows in the JSONL file."""
    p = Path(input_file)
    lines = p.read_text().splitlines()
    rows = [line for line in lines if line.strip()]
    click.echo(f"{len(rows)}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file.",
)
@click.option("--output", required=True, type=click.Path(), help="Output JSONL file.")
@click.option("--n", required=True, type=int, help="Number of rows to sample or slice.")
@click.option("--seed", type=int, help="Random seed for sampling.")
@click.option(
    "--slice",
    "use_slice",
    is_flag=True,
    help="Slice the first N rows instead of random sampling.",
)
def sample(
    input_file: str, output: str, n: int, seed: int | None, use_slice: bool
) -> None:
    """Sample or slice N rows from a JSONL file."""
    p = Path(input_file)
    lines = [line for line in p.read_text().splitlines() if line.strip()]

    if n > len(lines):
        click.echo(
            f"Warning: requested N={n} is greater than total rows={len(lines)}. Keeping all rows.",
            err=True,
        )
        selected = lines
    elif use_slice:
        selected = lines[:n]
    else:
        if seed is not None:
            random.seed(seed)
        selected = random.sample(lines, n)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(selected) + "\n")
    click.echo(f"Wrote {len(selected)} rows to {output}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file.",
)
@click.option("--output", required=True, type=click.Path(), help="Output JSONL file.")
@click.option(
    "--key",
    required=True,
    help="Deduplication key field. Can be comma-separated for composite keys (e.g. instruction,response).",
)
def dedupe(input_file: str, output: str, key: str) -> None:
    """Remove duplicate rows based on unique key values."""
    p = Path(input_file)
    lines = [line for line in p.read_text().splitlines() if line.strip()]

    keys = [k.strip() for k in key.split(",") if k.strip()]

    seen = set()
    unique_rows = []

    for line in lines:
        row = json.loads(line)
        if len(keys) == 1:
            val = row.get(keys[0])
        else:
            val = tuple(row.get(k) for k in keys)

        if val in seen:
            continue
        seen.add(val)
        unique_rows.append(line)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(unique_rows) + "\n")
    click.echo(
        f"Deduped {input_file}: {len(lines)} -> {len(unique_rows)} rows saved to {output}"
    )


@cli.command()
@click.option(
    "--output", required=True, type=click.Path(), help="Output merged JSONL file."
)
@click.argument("inputs", nargs=-1, type=click.Path(exists=True), required=True)
def merge(output: str, inputs: tuple[str, ...]) -> None:
    """Sequentially merge multiple JSONL files into one."""
    merged_lines = []
    for inp in inputs:
        p = Path(inp)
        lines = [line for line in p.read_text().splitlines() if line.strip()]
        merged_lines.extend(lines)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(merged_lines) + "\n")
    click.echo(
        f"Merged {len(inputs)} files ({len(merged_lines)} total rows) into {output}"
    )


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file.",
)
@click.option(
    "--out-dir",
    required=True,
    type=click.Path(),
    help="Output directory to write splits.",
)
@click.option("--train", default=0.8, type=float, help="Train split ratio.")
@click.option("--valid", default=0.1, type=float, help="Validation split ratio.")
@click.option("--test", default=0.1, type=float, help="Test split ratio.")
@click.option("--seed", type=int, default=42, help="Random seed for splitting.")
def split(
    input_file: str, out_dir: str, train: float, valid: float, test: float, seed: int
) -> None:
    """Split a JSONL file into train/validation/test datasets."""
    total_ratio = train + valid + test
    if not (0.99 <= total_ratio <= 1.01):
        raise click.ClickException(f"Split ratios must sum to 1.0 (got {total_ratio})")

    p = Path(input_file)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    random.seed(seed)
    random.shuffle(rows)

    n_total = len(rows)
    n_train = int(n_total * train)
    n_valid = int(n_total * valid)

    train_rows = rows[:n_train]
    valid_rows = rows[n_train : n_train + n_valid]
    test_rows = rows[n_train + n_valid :]

    out_p = Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    for name, subset in [
        ("train", train_rows),
        ("validation", valid_rows),
        ("test", test_rows),
    ]:
        if subset:
            file_path = out_p / f"{name}.jsonl"
            file_path.write_text("\n".join(json.dumps(r) for r in subset) + "\n")
            click.echo(f"  Wrote {len(subset)} rows to {file_path}")


@cli.command()
@click.option(
    "--input",
    "input_file",
    required=True,
    type=click.Path(exists=True),
    help="Input JSONL file.",
)
@click.option("--output", required=True, type=click.Path(), help="Output JSONL file.")
@click.option(
    "--copies", default=2, type=int, help="Number of copies of each row to make."
)
def duplicate(input_file: str, output: str, copies: int) -> None:
    """Duplicate JSONL rows N times and append copy suffix to IDs."""
    p = Path(input_file)
    rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]

    duplicated = []
    for r in rows:
        for i in range(copies):
            copy = dict(r)
            if "id" in copy:
                copy["id"] = f"{copy['id']}__c{i}"
            duplicated.append(json.dumps(copy))

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(duplicated) + "\n")
    click.echo(f"Wrote {len(duplicated)} rows ({copies}x of {len(rows)}) to {output}")


if __name__ == "__main__":
    cli()
