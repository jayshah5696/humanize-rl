from __future__ import annotations

import json
from pathlib import Path

import click


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")
    return json.loads(path.read_text())


@click.command()
@click.option(
    "--dataset-dir",
    type=click.Path(path_type=Path, exists=True, file_okay=False),
    default=Path("data/processed/sft/gemma4_e2b_v04"),
    show_default=True,
)
def main(dataset_dir: Path) -> None:
    """Print the frozen Gemma 4 SFT dataset manifest summary."""
    manifest = load_json(dataset_dir / "manifest.json")
    report_path = dataset_dir / "quality_report.md"
    click.echo(json.dumps(manifest, indent=2))
    if report_path.exists():
        click.echo(f"quality_report={report_path}")
    else:
        click.echo(f"quality_report_missing={report_path}")


if __name__ == "__main__":
    main()
