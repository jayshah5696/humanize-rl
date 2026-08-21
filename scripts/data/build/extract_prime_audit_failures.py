# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click

DEFAULT_OUTPUT_PATH = Path("data/processed/sft/prime_audit_failure_set_env0315.jsonl")
DEFAULT_HIGH_REWARD_THRESHOLD = 0.75
DEFAULT_COMPLETION_MAX_CHARS = 2000
BAD_PHRASES = (
    "we got",
    "stuff",
    "you guys",
    "hope you're doing well",
    "thanks for hanging in there",
    "operational game",
    "vital",
    "crucial",
    "leverage",
    "delve",
    "tapestry",
    "testament to",
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _as_json(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _chat_content(value: Any) -> str:
    decoded = _as_json(value)
    if isinstance(decoded, list):
        parts: list[str] = []
        for item in decoded:
            if isinstance(item, dict):
                content = item.get("content")
                if isinstance(content, str):
                    parts.append(content)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part for part in parts if part).strip()
    if isinstance(decoded, dict):
        content = decoded.get("content")
        return str(content).strip() if content is not None else ""
    return str(decoded or "").strip()


def load_rollout_samples(path: Path) -> dict[str, dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        samples = data
    elif isinstance(data, dict):
        samples = _as_json(data.get("samples", []))
    else:
        raise click.ClickException(
            f"{path} has unsupported Prime rollout JSON root: {type(data).__name__}"
        )
    if not isinstance(samples, list):
        raise click.ClickException(f"{path} does not contain a samples list")

    indexed: dict[str, dict[str, Any]] = {}
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = sample.get("sample_id")
        if sample_id is not None:
            indexed[str(sample_id)] = sample
    return indexed


def _audit_label(path: Path, audit: dict[str, Any]) -> str:
    rollout_path = Path(str(audit.get("rollout_path") or ""))
    run_id = rollout_path.parent.name if rollout_path.parent.name else "unknown_run"
    return f"{run_id}:{path.stem}"


def _phrase_hits(text: str) -> list[str]:
    lowered = text.lower()
    return [phrase for phrase in BAD_PHRASES if phrase in lowered]


def _reasons(sample: dict[str, Any], high_reward_threshold: float) -> list[str]:
    reasons = [str(item) for item in sample.get("failed_diagnostics", [])]
    reward = sample.get("rescored_reward")
    if isinstance(reward, int | float) and reward >= high_reward_threshold:
        reasons.append("high_rescored_reward")
    if not reasons:
        reasons.append("prime_audit_top_sample")
    return sorted(set(reasons))


def _include_sample(sample: dict[str, Any], high_reward_threshold: float) -> bool:
    failed = sample.get("failed_diagnostics", [])
    if isinstance(failed, list) and failed:
        return True
    reward = sample.get("rescored_reward")
    return isinstance(reward, int | float) and reward >= high_reward_threshold


def _resolve_rollout_path(raw_path: str, audit_path: Path) -> Path:
    rollout_path = Path(raw_path)
    if rollout_path.exists():
        return rollout_path
    candidate = audit_path.parent / rollout_path.name
    if candidate.exists():
        return candidate
    return rollout_path


def failure_rows_from_audit(
    *,
    audit_path: Path,
    high_reward_threshold: float = DEFAULT_HIGH_REWARD_THRESHOLD,
    completion_max_chars: int = DEFAULT_COMPLETION_MAX_CHARS,
    allow_missing_rollouts: bool = False,
) -> list[dict[str, Any]]:
    audit = _load_json(audit_path)
    if not isinstance(audit, dict):
        raise click.ClickException(f"{audit_path} must contain a JSON object")

    raw_rollout_path = str(audit.get("rollout_path") or "")
    rollout_path = _resolve_rollout_path(raw_rollout_path, audit_path)
    rollout_samples: dict[str, dict[str, Any]] = {}
    if rollout_path.exists():
        rollout_samples = load_rollout_samples(rollout_path)
    elif not allow_missing_rollouts:
        raise click.ClickException(f"missing rollout file: {raw_rollout_path}")

    top_samples = audit.get("top_samples", [])
    if not isinstance(top_samples, list):
        raise click.ClickException(f"{audit_path} top_samples must be a list")

    label = _audit_label(audit_path, audit)
    rows: list[dict[str, Any]] = []
    for rank, sample in enumerate(top_samples, start=1):
        if not isinstance(sample, dict) or not _include_sample(sample, high_reward_threshold):
            continue

        sample_id = str(sample.get("sample_id") or "")
        rollout_sample = rollout_samples.get(sample_id, {})
        completion = _chat_content(rollout_sample.get("completion"))
        if not completion:
            completion = str(sample.get("completion_preview") or "").strip()
        completion = completion[:completion_max_chars]

        task_id = str(sample.get("task_id") or "")
        comparison = f"{label}:rank_{rank}:sample_{sample_id or 'missing'}"
        rows.append(
            {
                "task_id": task_id,
                "comparison": comparison,
                "reasons": _reasons(sample, high_reward_threshold),
                "phrase_hits": _phrase_hits(completion),
                "base_completion_preview": "",
                "rl_completion_preview": completion,
                "metadata": {
                    "audit_path": str(audit_path),
                    "rollout_path": raw_rollout_path,
                    "problem_id": sample.get("problem_id"),
                    "sample_id": sample.get("sample_id"),
                    "rank_in_audit": rank,
                    "sample_reward": sample.get("sample_reward"),
                    "rescored_reward": sample.get("rescored_reward"),
                    "failed_diagnostics": sample.get("failed_diagnostics", []),
                    "source": "prime_rollout_audit_top_sample",
                },
            }
        )
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    )


@click.command(context_settings={"show_default": True})
@click.option(
    "--audit",
    "audit_paths",
    multiple=True,
    required=True,
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    help="Prime rollout audit JSON. May be repeated.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT_PATH,
    help="Failure-set JSONL for generate_sft_references_from_failures.py.",
)
@click.option(
    "--high-reward-threshold",
    default=DEFAULT_HIGH_REWARD_THRESHOLD,
    type=float,
    help="Include samples at or above this rescored reward even with no diagnostics.",
)
@click.option("--limit", type=int, default=None, help="Maximum rows to write.")
@click.option("--completion-max-chars", type=int, default=DEFAULT_COMPLETION_MAX_CHARS)
@click.option(
    "--allow-missing-rollouts",
    is_flag=True,
    help="Use audit completion previews when rollout files are unavailable.",
)
def cli(
    audit_paths: tuple[Path, ...],
    output_path: Path,
    high_reward_threshold: float,
    limit: int | None,
    completion_max_chars: int,
    allow_missing_rollouts: bool,
) -> None:
    """Extract SFT repair failure rows from Prime rollout audit reports."""
    rows: list[dict[str, Any]] = []
    for audit_path in audit_paths:
        rows.extend(
            failure_rows_from_audit(
                audit_path=audit_path,
                high_reward_threshold=high_reward_threshold,
                completion_max_chars=completion_max_chars,
                allow_missing_rollouts=allow_missing_rollouts,
            )
        )
    if limit is not None:
        rows = rows[:limit]
    write_jsonl(output_path, rows)
    click.echo(
        f"failure_rows={len(rows)} audits={len(audit_paths)} output={output_path}"
    )


if __name__ == "__main__":
    cli()
