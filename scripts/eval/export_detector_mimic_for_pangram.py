# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "pydantic>=2",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Literal

import click
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from humanize_rl.scoring.detector_mimic import DetectorMimicRow

DEFAULT_INPUT = Path("data/eval/detector_mimic_v01.jsonl")
DEFAULT_OUTPUT = Path("runs/detector_mimic/pangram_bulk_items.json")


class PangramBulkItem(BaseModel):
    """One SDK-ready Pangram bulk item."""

    id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)


class PangramBulkItemsPayload(BaseModel):
    """Offline handoff payload for Pangram bulk detection."""

    artifact: Literal["pangram_bulk_items"] = "pangram_bulk_items"
    source_path: str
    source_sha256: str
    item_count: int
    sdk_method: str = "Pangram.submit_bulk(items=payload['items'])"
    docs: list[str] = [
        "https://docs.pangram.com/quickstart",
        "https://docs.pangram.com/sdk/python",
    ]
    items: list[PangramBulkItem]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[DetectorMimicRow]:
    rows: list[DetectorMimicRow] = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(DetectorMimicRow.model_validate_json(line))
        except ValueError as exc:
            raise click.ClickException(f"{path}:{line_number}: {exc}") from exc
    return rows


def build_pangram_bulk_items_payload(
    rows: list[DetectorMimicRow], *, source_path: Path
) -> PangramBulkItemsPayload:
    seen_ids: set[str] = set()
    items: list[PangramBulkItem] = []
    for row in rows:
        if row.id in seen_ids:
            raise click.ClickException(f"duplicate detector row id: {row.id}")
        seen_ids.add(row.id)
        items.append(PangramBulkItem(id=row.id, text=row.text))

    return PangramBulkItemsPayload(
        source_path=str(source_path),
        source_sha256=_sha256(source_path),
        item_count=len(items),
        items=items,
    )


@click.command(context_settings={"show_default": True})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_INPUT,
    help="Frozen detector-mimic JSONL rows to send to Pangram.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT,
    help="SDK-ready Pangram bulk items payload.",
)
def cli(input_path: Path, output_path: Path) -> None:
    """Export frozen detector-mimic rows as Pangram SDK bulk items."""
    payload = build_pangram_bulk_items_payload(
        _load_rows(input_path),
        source_path=input_path,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )
    click.echo(f"pangram_bulk_items={output_path} items={payload.item_count}")


if __name__ == "__main__":
    cli()
