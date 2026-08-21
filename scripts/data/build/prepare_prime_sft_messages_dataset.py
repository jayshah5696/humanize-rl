# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import click

DEFAULT_INPUT_DIR = Path("data/processed/sft/gemma4_e2b_v04_prime_env0315_clean50")
DEFAULT_OUTPUT_DIR = Path(
    "runs/hf_datasets/humanize-rl-prime-sft-messages-env0315-clean50-primecompat"
)
DEFAULT_REPO_ID = (
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat"
)
SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "valid.jsonl",
    "test": "test.jsonl",
}
PRIME_SAFE_FIELDS = (
    "id",
    "messages",
    "domain",
    "task_type",
    "mode",
    "source",
    "license",
    "release_eligible",
    "split",
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_messages(row: dict[str, Any], *, split: str, index: int) -> None:
    messages = row.get("messages")
    if not isinstance(messages, list) or len(messages) != 2:
        raise click.ClickException(f"{split} row {index} must have two messages")
    roles = [message.get("role") for message in messages if isinstance(message, dict)]
    if roles != ["user", "assistant"]:
        raise click.ClickException(f"{split} row {index} roles must be user,assistant")
    for message in messages:
        if not isinstance(message, dict) or not str(message.get("content") or "").strip():
            raise click.ClickException(f"{split} row {index} has empty message content")


def _prime_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe = {field: row[field] for field in PRIME_SAFE_FIELDS if field in row}
    for field in ("id", "domain", "task_type", "mode", "source", "license", "split"):
        safe.setdefault(field, "")
    safe["release_eligible"] = bool(row.get("release_eligible", False))
    return safe


def _copy_if_exists(src: Path, dst: Path) -> str | None:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    return str(dst)


def _render_readme(
    *,
    repo_id: str,
    dataset_version: str,
    row_counts: dict[str, int],
    source_counts: Counter[str],
    mode_counts: Counter[str],
    repair_rows: int,
) -> str:
    total_rows = sum(row_counts.values())
    source_rows = "\n".join(
        f"| {source} | {count} |" for source, count in source_counts.most_common()
    )
    mode_rows = "\n".join(
        f"| {mode} | {count} |" for mode, count in mode_counts.most_common()
    )
    return f"""---
configs:
  - config_name: default
    data_files:
      - split: train
        path: data/train.jsonl
      - split: validation
        path: data/validation.jsonl
      - split: test
        path: data/test.jsonl
license: apache-2.0
task_categories:
  - text-generation
language:
  - en
size_categories:
  - 1K<n<10K
---

# {repo_id}

Prime `prime-rl` supervised fine-tuning dataset for Humanize-RL.

This is the `{dataset_version}` S2 repair-data candidate. It starts from the
env0314 Prime SFT corpus and adds cleaned env0315 repair references generated
from saved Prime rollout-audit failures.

## Splits

| split | rows |
|---|---:|
| train | {row_counts["train"]} |
| validation | {row_counts["validation"]} |
| test | {row_counts["test"]} |
| total | {total_rows} |

## Sources

| source | rows |
|---|---:|
{source_rows}

Accepted repair-reference rows: `{repair_rows}`.

## Modes

| mode | rows |
|---|---:|
{mode_rows}

## Schema

Training rows are restricted to Prime-compatible feature types. Each row includes
a `messages` column with exactly two turns:

```json
[
  {{"role": "user", "content": "..."}},
  {{"role": "assistant", "content": "..."}}
]
```

Nested audit fields such as `quality` and `metadata` are kept in the local
builder reports, not in the Hub training rows, because Prime's pinned
`datasets` stack does not accept Hub features exported as `_type: Json`.

## Intended Use

Use this dataset for the S2 Qwen 2B Prime dataset-SFT ablation. Do not treat it
as a promoted model result by itself. SFT output still has to beat base on the
frozen eval prompts, pass the detector-mimic gate, pass human read, and pass the
SFT promotion gate before any RL-after-SFT launch.
"""


def prepare_prime_sft_messages_dataset(
    *,
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    dataset_version: str,
    repair_report: Path | None = None,
) -> dict[str, Any]:
    """Prepare a three-split HF dataset folder for Prime prime-rl SFT."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    data_dir = output_dir / "data"
    reports_dir = output_dir / "reports"

    row_counts: dict[str, int] = {}
    split_sha256: dict[str, str] = {}
    all_rows: list[dict[str, Any]] = []
    observed_columns: set[str] = set()
    for split, file_name in SPLIT_FILES.items():
        src = input_dir / file_name
        if not src.exists():
            raise click.ClickException(f"missing split file: {src}")
        rows = _load_jsonl(src)
        for index, row in enumerate(rows, start=1):
            _validate_messages(row, split=split, index=index)
            observed_columns.update(row.keys())
        dst = data_dir / f"{split}.jsonl"
        _write_jsonl(dst, [_prime_safe_row(row) for row in rows])
        row_counts[split] = len(rows)
        split_sha256[split] = _sha256(dst)
        all_rows.extend(rows)

    source_counts = Counter(str(row.get("source") or "unknown") for row in all_rows)
    mode_counts = Counter(str(row.get("mode") or "unknown") for row in all_rows)
    repair_rows = source_counts.get("prime_failure_reference_generation", 0)

    copied_reports: dict[str, str] = {}
    for src, name in (
        (input_dir / "quality_report.md", "quality_report.md"),
        (input_dir / "manifest.json", "builder_manifest.json"),
    ):
        copied = _copy_if_exists(src, reports_dir / name)
        if copied:
            copied_reports[name] = copied
    if repair_report:
        copied = _copy_if_exists(repair_report, reports_dir / repair_report.name)
        if copied:
            copied_reports[repair_report.name] = copied

    readme = _render_readme(
        repo_id=repo_id,
        dataset_version=dataset_version,
        row_counts=row_counts,
        source_counts=source_counts,
        mode_counts=mode_counts,
        repair_rows=repair_rows,
    )
    (output_dir / "README.md").write_text(readme)

    report = {
        "artifact": "prime_sft_messages_hf_dataset",
        "repo_id": repo_id,
        "dataset_version": dataset_version,
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "row_counts": row_counts,
        "total_rows": sum(row_counts.values()),
        "split_sha256": split_sha256,
        "training_columns": list(PRIME_SAFE_FIELDS),
        "dropped_training_columns": sorted(
            observed_columns.difference(PRIME_SAFE_FIELDS)
        ),
        "source_counts": dict(sorted(source_counts.items())),
        "mode_counts": dict(sorted(mode_counts.items())),
        "repair_reference_rows": repair_rows,
        "reports": copied_reports,
    }
    (output_dir / "manifest.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option(
    "--input-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_INPUT_DIR,
    help="Built SFT dataset directory with train/valid/test JSONL files.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_OUTPUT_DIR,
    help="Hub-ready dataset folder to write.",
)
@click.option("--repo-id", default=DEFAULT_REPO_ID)
@click.option("--dataset-version", default="env0315_clean50", show_default=True)
@click.option(
    "--repair-report",
    type=click.Path(path_type=Path, dir_okay=False),
    default=Path(
        "data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50_report.json"
    ),
    help="Optional repair-reference clean report to include.",
)
def cli(
    input_dir: Path,
    output_dir: Path,
    repo_id: str,
    dataset_version: str,
    repair_report: Path | None,
) -> None:
    """Write a Hub-ready Prime SFT messages dataset folder."""
    report = prepare_prime_sft_messages_dataset(
        input_dir=input_dir,
        output_dir=output_dir,
        repo_id=repo_id,
        dataset_version=dataset_version,
        repair_report=repair_report,
    )
    click.echo(
        "prime_sft_messages_dataset={path} rows={rows} repair_rows={repair}".format(
            path=output_dir,
            rows=report["total_rows"],
            repair=report["repair_reference_rows"],
        )
    )


if __name__ == "__main__":
    cli()
