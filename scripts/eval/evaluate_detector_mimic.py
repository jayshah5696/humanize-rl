# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "pydantic>=2",
# ]
# ///
from __future__ import annotations

import json
import sys
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from humanize_rl.scoring.detector_mimic import (
    DetectorMimicRow,
    evaluate_detector_mimic_rows,
)


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


@click.command(context_settings={"show_default": True})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/eval/detector_mimic_v01.jsonl"),
    help="Frozen detector-mimic JSONL input.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("runs/detector_mimic/detector_mimic_v01_report.json"),
    help="JSON summary/report output.",
)
@click.option(
    "--scored-output",
    "scored_output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("runs/detector_mimic/detector_mimic_v01_scored.jsonl"),
    help="Per-row scored JSONL output.",
)
@click.option(
    "--no-fail-on-gate",
    is_flag=True,
    help="Write reports but return success even when the gate fails.",
)
def cli(
    input_path: Path,
    output_path: Path,
    scored_output_path: Path,
    no_fail_on_gate: bool,
) -> None:
    """Run the frozen Pangram-style detector-mimic gate."""
    rows = _load_rows(input_path)
    report = evaluate_detector_mimic_rows(rows, source=str(input_path))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored_output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n")
    scored_output_path.write_text(
        "\n".join(row.model_dump_json() for row in report.rows) + "\n"
    )

    summary = report.summary
    click.echo(
        "rows={rows} false_positive={fp} false_negative={fn} gate={gate}".format(
            rows=summary.total_rows,
            fp=summary.false_positive_rows,
            fn=summary.false_negative_rows,
            gate="pass" if summary.gate_passed else "fail",
        )
    )
    if not summary.gate_passed and not no_fail_on_gate:
        raise click.ClickException("Detector mimic gate failed")


if __name__ == "__main__":
    cli()
