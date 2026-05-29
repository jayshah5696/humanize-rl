"""Generic arka pipeline runner with per-run author / output overrides.

Reads a v03 mode YAML, optionally overrides `llm.model` (author) and
`output.path`, writes a temp YAML, and invokes `arka` as a subprocess.

See plan §7.1 (`scripts/data/build/run_arka_pipeline.py`).

Usage:

    uv run scripts/data/build/run_arka_pipeline.py \\
        --config configs/v03/v03_rewrite_humanize.yaml \\
        --author google/gemini-3.1-pro-preview \\
        --out data/processed/v03_rewrite_humanize_gemini-3.1-pro.jsonl
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import click
import yaml

_SHIMS_DIR = Path(__file__).resolve().parent / "_shims"


def _author_slug(author: str) -> str:
    return author.replace("/", "_").replace(":", "_")


def _stamp_author(prompt: str, author: str, temperature: float | None) -> str:
    """Append a small author-system nudge to the transform_generator prompt.

    Plan §3.3 calls for per-author temperature + system nudge; we keep this
    in-process so YAMLs don't need to be duplicated per author.
    """
    nudges = {
        "google/gemini-3.1-pro-preview":
            "(You write production-quality writing briefs for a creative "
            "ops team.)",
        "openai/gpt-5.4-mini":
            "(You draft test prompts for an SAT-style writing exam.)",
        "google/gemini-3.1-flash-lite-preview":
            "(You generate short workplace-message briefs as if writing "
            "slack DMs.)",
    }
    nudge = nudges.get(author)
    if not nudge:
        return prompt
    return f"{nudge}\n\n{prompt}"


def _author_temperature(author: str) -> float | None:
    return {
        "google/gemini-3.1-pro-preview": 0.7,
        "openai/gpt-5.4-mini": 0.9,
        "google/gemini-3.1-flash-lite-preview": 0.6,
    }.get(author)


def _resolve_relative_paths(cfg: dict, config_parent: Path) -> dict:
    """Resolve YAML-relative paths to absolute (config_parent / relative).

    Needed because we move the resolved YAML to a tempfile elsewhere; arka
    computes project_root from `config_path.parent`, so relative paths in the
    YAML break unless we pre-resolve them.
    """
    def _resolve(p: str) -> str:
        path = Path(p)
        if path.is_absolute():
            return str(path)
        return str((config_parent / path).resolve())

    for stage in cfg.get("pipeline", []):
        t = stage.get("type")
        if t == "seed_source" and stage.get("path"):
            stage["path"] = _resolve(stage["path"])
        if t == "labeling_engine" and stage.get("rubric_path"):
            stage["rubric_path"] = _resolve(stage["rubric_path"])
    out_cfg = cfg.get("output", {})
    if out_cfg.get("path"):
        out_cfg["path"] = _resolve(out_cfg["path"])
    return cfg


def _apply_overrides(
    cfg: dict,
    author: str | None,
    out: Path | None,
    seed_path: Path | None,
    target_count: int | None,
) -> dict:
    if author:
        cfg.setdefault("llm", {})["model"] = author
        temp = _author_temperature(author)
        for stage in cfg.get("pipeline", []):
            if stage.get("type") in {"transform_generator", "prompt_based_generator"}:
                if "prompt_template" in stage:
                    stage["prompt_template"] = _stamp_author(
                        stage["prompt_template"], author, temp
                    )
                if temp is not None:
                    stage["temperature"] = temp
                # also override llm_override.model if present
                if "llm_override" in stage:
                    stage["llm_override"]["model"] = author
    if out:
        cfg.setdefault("output", {})["path"] = str(out.resolve())
    if seed_path:
        for stage in cfg.get("pipeline", []):
            if stage.get("type") == "seed_source":
                # Resolve to absolute; arka computes project_root from
                # config_path.parent, but absolute paths bypass that.
                stage["path"] = str(seed_path.resolve())
                break
    if target_count is not None:
        for stage in cfg.get("pipeline", []):
            if stage.get("type") in {
                "transform_generator",
                "prompt_based_generator",
                "evol_instruct_generator",
            }:
                stage["target_count"] = target_count
    return cfg


@click.command(context_settings={"show_default": True})
@click.option(
    "--config", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True, help="Mode YAML to run.",
)
@click.option(
    "--author", type=str, default=None,
    help="Override llm.model. Author-specific temperature + system nudge applied.",
)
@click.option(
    "--out", type=click.Path(dir_okay=False, path_type=Path), default=None,
    help="Override output.path.",
)
@click.option(
    "--seed-path", type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None, help="Override seed_source.path (e.g. for smoke runs).",
)
@click.option(
    "--target-count", type=int, default=None,
    help="Override generator target_count (smoke runs).",
)
@click.option(
    "--run-id", type=str, default=None,
    help="Optional arka --run-id (defaults to auto-uuid).",
)
@click.option(
    "--dry-run", is_flag=True,
    help="Write resolved YAML and print arka command; do not invoke.",
)
@click.option(
    "--reasoning-exclude", is_flag=True,
    help="Inject OpenRouter `extra_body={'reasoning': {'exclude': true}}` "
         "into every chat.completions.create call via PYTHONPATH shim. "
         "Use for Gemini 3.1 Pro to stop reasoning from burning the "
         "completion budget.",
)
@click.option(
    "--extra-body", type=str, default=None,
    help="JSON string passed to OpenRouter extra_body (overrides "
         "--reasoning-exclude). Example: '{\"reasoning\":{\"max_tokens\":500}}'.",
)
@click.option(
    "--skip-errors", is_flag=True,
    help="Use the in-process arka runner (_arka_runner.py) that wraps the "
         "transform_generator per-row loop in try/except. Use when a model "
         "is known to fail some prompts (length limits, content filters).",
)
def main(
    config: Path,
    author: str | None,
    out: Path | None,
    seed_path: Path | None,
    target_count: int | None,
    run_id: str | None,
    dry_run: bool,
    reasoning_exclude: bool,
    extra_body: str | None,
    skip_errors: bool,
) -> None:
    """Run an arka pipeline with per-author overrides."""
    cfg = yaml.safe_load(config.read_text())
    cfg = _resolve_relative_paths(cfg, config.parent.resolve())
    cfg = _apply_overrides(cfg, author, out, seed_path, target_count)

    arka_path = shutil.which("arka") or shutil.which("uv")
    if not arka_path:
        raise click.ClickException("Neither `arka` nor `uv` on PATH.")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False,
        dir=config.parent,  # keep relative paths in YAML valid
    ) as tmp:
        yaml.safe_dump(cfg, tmp, sort_keys=False)
        tmp_path = Path(tmp.name)

    try:
        if shutil.which("arka"):
            cmd = ["arka", "--config", str(tmp_path)]
        else:
            cmd = ["uv", "run", "arka", "--config", str(tmp_path)]
        if run_id:
            cmd += ["--run-id", run_id]

        click.echo(f"resolved config → {tmp_path}")
        if author:
            click.echo(f"author     → {author}")
        if out:
            click.echo(f"output     → {out}")
        click.echo(f"cmd        → {' '.join(cmd)}")

        if dry_run:
            return

        env = os.environ.copy()
        extra_payload: dict | None = None
        if extra_body:
            extra_payload = json.loads(extra_body)
        elif reasoning_exclude:
            extra_payload = {"reasoning": {"exclude": True}}
        shim_needed = False
        if extra_payload is not None:
            env["HUMANIZE_OPENROUTER_EXTRA_BODY"] = json.dumps(extra_payload)
            click.echo(f"extra_body  → {env['HUMANIZE_OPENROUTER_EXTRA_BODY']}")
            shim_needed = True
        if skip_errors:
            click.secho(
                "  ! --skip-errors no longer wired (arka transform patch "
                "removed); ignored.", fg="yellow",
            )
        if shim_needed:
            existing_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = (
                f"{_SHIMS_DIR}{os.pathsep}{existing_pp}" if existing_pp
                else str(_SHIMS_DIR)
            )

        if skip_errors:
            runner_path = Path(__file__).resolve().parent / "_arka_runner.py"
            cmd = ["uv", "run", str(runner_path), "--config", str(tmp_path)]
            if run_id:
                cmd += ["--run-id", run_id]
            click.echo(f"in-proc cmd → {' '.join(cmd)}")

        result = subprocess.run(cmd, check=False, env=env)
        sys.exit(result.returncode)
    finally:
        if not dry_run:
            try:
                tmp_path.unlink()
            except OSError:
                pass


if __name__ == "__main__":
    main()
