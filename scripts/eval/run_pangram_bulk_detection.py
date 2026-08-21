# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
#   "pangram-sdk>=0.3.1",
# ]
# ///
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import click

DEFAULT_INPUT = Path("runs/detector_mimic/pangram_bulk_items.json")
DEFAULT_OUTPUT = Path("runs/detector_mimic/pangram_export.json")
DEFAULT_SUBMIT_REPORT = Path("runs/detector_mimic/pangram_bulk_submit.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_items_payload(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    if data.get("artifact") != "pangram_bulk_items":
        raise click.ClickException("input artifact must be pangram_bulk_items")
    items = data.get("items")
    if not isinstance(items, list) or not items:
        raise click.ClickException("input must contain a non-empty items list")
    seen_ids: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise click.ClickException(f"items[{index}] must be an object")
        item_id = str(item.get("id") or "")
        text = str(item.get("text") or "")
        if not item_id:
            raise click.ClickException(f"items[{index}].id is required")
        if not text:
            raise click.ClickException(f"items[{index}].text is required")
        if item_id in seen_ids:
            raise click.ClickException(f"duplicate item id: {item_id}")
        seen_ids.add(item_id)
    return data


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _new_pangram_client(api_key: str | None = None) -> Any:
    try:
        from pangram import Pangram
    except ImportError as exc:
        raise click.ClickException(
            "pangram-sdk is required; run this script with uv run"
        ) from exc
    if api_key:
        return Pangram(api_key=api_key)
    return Pangram()


def _dry_run_report(*, input_path: Path, output_path: Path) -> dict[str, Any]:
    payload = _load_items_payload(input_path)
    return {
        "artifact": "pangram_bulk_dry_run",
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "output_path": str(output_path),
        "item_count": len(payload["items"]),
        "sdk_method": "Pangram.submit_bulk(items=payload['items'])",
    }


def run_pangram_bulk_detection(
    *,
    input_path: Path,
    output_path: Path,
    submit_report_path: Path,
    client: Any,
    timeout: float,
    poll_interval: float,
) -> dict[str, Any]:
    """Submit Pangram bulk items, wait for completion, and write raw results."""
    for method_name in ("submit_bulk", "wait_for_bulk", "get_bulk_results"):
        if not hasattr(client, method_name):
            raise click.ClickException(f"Pangram client missing {method_name}")

    payload = _load_items_payload(input_path)
    input_sha = _sha256(input_path)
    items = [{"id": item["id"], "text": item["text"]} for item in payload["items"]]

    submission = client.submit_bulk(items=items)
    bulk_id = submission.get("bulk_id")
    if not bulk_id:
        raise click.ClickException("Pangram submit_bulk response missing bulk_id")

    submit_report = {
        "artifact": "pangram_bulk_submit",
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "item_count": len(items),
        "submission": submission,
    }
    _write_json(submit_report_path, submit_report)

    status = client.wait_for_bulk(
        bulk_id,
        timeout=timeout,
        poll_interval=poll_interval,
    )
    results = client.get_bulk_results(bulk_id)
    report = {
        "artifact": "pangram_bulk_results",
        "input_path": str(input_path),
        "input_sha256": input_sha,
        "bulk_id": bulk_id,
        "item_count": len(items),
        "submission": submission,
        "status": status,
        "items": results.get("items", []),
        "failed_items": results.get("failed_items", []),
        "raw_results": results,
    }
    _write_json(output_path, report)
    return report


@click.command(context_settings={"show_default": True})
@click.option(
    "--input",
    "input_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=DEFAULT_INPUT,
    help="SDK-ready Pangram bulk items payload.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT,
    help="Raw Pangram bulk results JSON for compare_detector_mimic_to_pangram.py.",
)
@click.option(
    "--submit-report",
    "submit_report_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_SUBMIT_REPORT,
    help="Submission metadata JSON.",
)
@click.option("--timeout", type=float, default=3600.0, help="Bulk wait timeout seconds.")
@click.option("--poll-interval", type=float, default=2.0, help="Bulk poll interval seconds.")
@click.option(
    "--api-key",
    default=None,
    help="Optional Pangram API key. Defaults to PANGRAM_API_KEY from the environment.",
)
@click.option("--dry-run", is_flag=True, help="Validate input and write no API results.")
def cli(
    input_path: Path,
    output_path: Path,
    submit_report_path: Path,
    timeout: float,
    poll_interval: float,
    api_key: str | None,
    dry_run: bool,
) -> None:
    """Submit frozen detector rows to Pangram bulk detection."""
    if dry_run:
        report = _dry_run_report(input_path=input_path, output_path=output_path)
        _write_json(submit_report_path, report)
        click.echo(f"pangram_bulk_dry_run={submit_report_path} items={report['item_count']}")
        return

    report = run_pangram_bulk_detection(
        input_path=input_path,
        output_path=output_path,
        submit_report_path=submit_report_path,
        client=_new_pangram_client(api_key),
        timeout=timeout,
        poll_interval=poll_interval,
    )
    failed = len(report["failed_items"])
    click.echo(
        "pangram_bulk_results={output} bulk_id={bulk_id} items={items} failed={failed}".format(
            output=output_path,
            bulk_id=report["bulk_id"],
            items=report["item_count"],
            failed=failed,
        )
    )


if __name__ == "__main__":
    cli()
