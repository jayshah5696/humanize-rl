# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "pydantic>=2",
# ]
# ///
from __future__ import annotations

import json
from pathlib import Path

import click

from humanize_rl.scoring.detector_mimic import DetectorMimicRow
from humanize_rl.scoring.pangram_alignment import (
    build_pangram_alignment_report,
    load_pangram_export_path,
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
    help="Frozen detector-mimic JSONL input sent to Pangram.",
)
@click.option(
    "--pangram-output",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Pangram JSON or JSONL export for the same rows.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("runs/detector_mimic/pangram_alignment_report.json"),
    help="Alignment report JSON output.",
)
@click.option(
    "--min-coverage",
    type=click.FloatRange(0.0, 1.0),
    default=1.0,
    help="Required fraction of detector rows matched by Pangram output.",
)
@click.option(
    "--max-label-disagreements",
    type=int,
    default=0,
    help="Allowed Pangram-vs-mimic human/nonhuman label disagreements.",
)
@click.option(
    "--max-mean-abs-fraction-delta",
    type=click.FloatRange(0.0, 1.0),
    default=0.30,
    help="Allowed mean absolute delta between Pangram and mimic AI fractions.",
)
@click.option(
    "--no-fail-on-gate",
    is_flag=True,
    help="Write the report but return success even when the gate fails.",
)
def cli(
    input_path: Path,
    pangram_output: Path,
    output_path: Path,
    min_coverage: float,
    max_label_disagreements: int,
    max_mean_abs_fraction_delta: float,
    no_fail_on_gate: bool,
) -> None:
    """Compare a saved Pangram detector export against the local mimic."""
    report = build_pangram_alignment_report(
        _load_rows(input_path),
        load_pangram_export_path(pangram_output),
        source=str(pangram_output),
        min_coverage=min_coverage,
        max_label_disagreements=max_label_disagreements,
        max_mean_abs_fraction_delta=max_mean_abs_fraction_delta,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"
    )

    summary = report.summary
    click.echo(
        "rows={rows} matched={matched} disagreements={disagreements} "
        "mean_abs_delta={delta:.6f} gate={gate}".format(
            rows=summary.total_rows,
            matched=summary.matched_rows,
            disagreements=summary.label_disagreement_rows,
            delta=summary.mean_abs_fraction_delta,
            gate="pass" if summary.gate_passed else "fail",
        )
    )
    if not summary.gate_passed and not no_fail_on_gate:
        raise click.ClickException("Pangram alignment gate failed")


if __name__ == "__main__":
    cli()
