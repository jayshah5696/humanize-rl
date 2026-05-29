from __future__ import annotations

from pathlib import Path

import click

from humanize_rl.training.gemma4_sft_dataset import (
    BuildConfig,
    build_dataset,
    convert_to_mlx_smoke,
)


@click.command()
@click.option(
    "--input-path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=Path("data/processed/v04_sft_final.jsonl"),
    show_default=True,
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/processed/sft/gemma4_e2b_v04"),
    show_default=True,
)
@click.option(
    "--mlx-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=Path("data/processed/sft/gemma4_e2b_v04_mlx_smoke"),
    show_default=True,
)
@click.option(
    "--track-a-scorer-path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path("models/track_a_10k/ridge.pkl"),
    show_default=True,
)
@click.option("--seed", type=int, default=3407, show_default=True)
@click.option("--smoke-train-size", type=int, default=100, show_default=True)
@click.option("--smoke-valid-size", type=int, default=20, show_default=True)
@click.option("--pilot-train-size", type=int, default=500, show_default=True)
@click.option("--pilot-valid-size", type=int, default=50, show_default=True)
@click.option("--max-ai-probability", type=float, default=0.85, show_default=True)
def main(
    input_path: Path,
    output_dir: Path,
    mlx_dir: Path,
    track_a_scorer_path: Path,
    seed: int,
    smoke_train_size: int,
    smoke_valid_size: int,
    pilot_train_size: int,
    pilot_valid_size: int,
    max_ai_probability: float,
) -> None:
    """Build frozen Gemma 4 E2B SFT train/valid/test and smoke splits."""
    scorer_path = track_a_scorer_path if track_a_scorer_path.exists() else None
    config = BuildConfig(
        input_path=input_path,
        output_dir=output_dir,
        track_a_scorer_path=scorer_path,
        seed=seed,
        smoke_train_size=smoke_train_size,
        smoke_valid_size=smoke_valid_size,
        pilot_train_size=pilot_train_size,
        pilot_valid_size=pilot_valid_size,
        max_ai_probability=max_ai_probability,
    )
    result = build_dataset(config)
    convert_to_mlx_smoke(output_dir, mlx_dir)

    manifest = result.report["manifest"]
    click.echo(f"raw_rows={manifest['raw_rows']}")
    click.echo(f"accepted_rows={manifest['accepted_rows']}")
    click.echo(f"rejected_rows={manifest['rejected_rows']}")
    click.echo(f"split_counts={manifest['split_counts']}")
    click.echo(f"output_dir={output_dir}")
    click.echo(f"mlx_dir={mlx_dir}")


if __name__ == "__main__":
    main()
