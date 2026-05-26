#!/usr/bin/env python3
"""Generic Hugging Face uploader script.

Supports uploading datasets, model weights folders, and arbitrary artifacts with flexible token resolution.
"""

from __future__ import annotations

import os
from pathlib import Path

import click
from datasets import Dataset
from huggingface_hub import HfApi, get_token


def get_hf_token() -> str:
    """Resolve HF token using specified priority order."""
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGING_FACE_HUB_TOKEN")
        or get_token()
    )
    if not token:
        raise click.ClickException(
            "No Hugging Face token found. Set HF_TOKEN, HUGGINGFACE_TOKEN, HUGGING_FACE_HUB_TOKEN, "
            "or run `huggingface-cli login` to authenticate."
        )
    return token


@click.group()
def cli() -> None:
    """Publish models, datasets, and reports to Hugging Face."""
    pass


@cli.command()
@click.option(
    "--repo-id",
    required=True,
    help="HF Hub dataset repository ID (e.g. jayshah5696/humanize-rl-sft-dataset).",
)
@click.option(
    "--path",
    type=click.Path(exists=True),
    help="Path to a prepared dataset file (e.g. jsonl, csv, parquet).",
)
@click.option("--config-name", help="Optional configuration/subset name.")
@click.option(
    "--readme", type=click.Path(exists=True), help="Path to local readme card file."
)
@click.option(
    "--artifact",
    "-a",
    "artifacts",
    multiple=True,
    help="Additional files to upload as 'local_path:remote_path'.",
)
@click.option(
    "--private", is_flag=True, default=False, help="Make the repository private."
)
def dataset(
    repo_id: str,
    path: str | None,
    config_name: str | None,
    readme: str | None,
    artifacts: list[str],
    private: bool,
) -> None:
    """Upload dataset files and artifacts to Hugging Face."""
    token = get_hf_token()
    api = HfApi(token=token)

    click.echo(f"Ensuring dataset repo exists: {repo_id}")
    api.create_repo(
        repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True
    )

    if path:
        click.echo(f"Uploading main dataset file: {path}")
        p = Path(path)
        try:
            if p.suffix == ".jsonl" or p.suffix == ".json":
                ds = Dataset.from_json(str(p))
            elif p.suffix == ".csv":
                ds = Dataset.from_csv(str(p))
            elif p.suffix == ".parquet":
                ds = Dataset.from_parquet(str(p))
            else:
                raise ValueError(f"Unsupported dataset file extension: {p.suffix}")

            click.echo("Pushing dataset representation...")
            ds.push_to_hub(
                repo_id, config_name=config_name, token=token, private=private
            )
        except Exception as e:
            click.echo(
                f"Could not load/push via datasets library ({e}). Uploading file directly.",
                err=True,
            )
            path_in_repo = f"data/{config_name}/{p.name}" if config_name else p.name
            api.upload_file(
                path_or_fileobj=str(p),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type="dataset",
            )

    if readme:
        click.echo(f"Uploading README: {readme}")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="dataset",
        )

    for art in artifacts:
        if ":" not in art:
            raise click.BadParameter(
                f"Artifact must be in 'local_path:remote_path' format: {art}"
            )
        local_path, remote_path = art.split(":", 1)
        click.echo(f"Uploading artifact: {local_path} -> {remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="dataset",
        )

    click.echo(f"Dataset upload complete: https://huggingface.co/datasets/{repo_id}")


@cli.command()
@click.option(
    "--repo-id",
    required=True,
    help="HF Hub model repository ID (e.g. jayshah5696/humanize-rl-track-a-ridge-scorer).",
)
@click.option(
    "--folder",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    help="Folder to upload containing model files.",
)
@click.option(
    "--readme", type=click.Path(exists=True), help="Path to local readme card file."
)
@click.option(
    "--artifact",
    "-a",
    "artifacts",
    multiple=True,
    help="Additional files to upload as 'local_path:remote_path'.",
)
@click.option(
    "--private", is_flag=True, default=False, help="Make the repository private."
)
def model(
    repo_id: str,
    folder: str | None,
    readme: str | None,
    artifacts: list[str],
    private: bool,
) -> None:
    """Upload model folders and artifacts to Hugging Face."""
    token = get_hf_token()
    api = HfApi(token=token)

    click.echo(f"Ensuring model repo exists: {repo_id}")
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)

    if folder:
        click.echo(f"Uploading model folder: {folder}")
        api.upload_folder(
            folder_path=folder,
            repo_id=repo_id,
            repo_type="model",
        )

    if readme:
        click.echo(f"Uploading README: {readme}")
        api.upload_file(
            path_or_fileobj=str(readme),
            path_in_repo="README.md",
            repo_id=repo_id,
            repo_type="model",
        )

    for art in artifacts:
        if ":" not in art:
            raise click.BadParameter(
                f"Artifact must be in 'local_path:remote_path' format: {art}"
            )
        local_path, remote_path = art.split(":", 1)
        click.echo(f"Uploading artifact: {local_path} -> {remote_path}")
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )

    click.echo(f"Model upload complete: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    cli()
