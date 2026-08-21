# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import click

DEFAULT_RUNS_ROOT = Path("runs/prime_training_smoke")
DEFAULT_OUTPUT = DEFAULT_RUNS_ROOT / "ablation_matrix_env0315.json"
EVAL_METRICS = {
    "mix_v2_p5050": "eval/eval_mix_v2_p5050_env0315/avg@1",
    "v02_strict": "eval/eval_v02_strict_env0315/avg@1",
    "v03_strict": "eval/eval_v03_strict_env0315/avg@1",
}
STEP_RE = re.compile(r"(?:^|_)step(\d+)(?:_|$)")


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _discover_run_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path.parent
        for path in root.glob("*/bundle_report.json")
        if (path.parent / "run.json").exists() and (path.parent / "metrics.json").exists()
    )


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _metric_delta(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    points: list[tuple[int, float]] = []
    for row in rows:
        step = row.get("step")
        value = _number(row.get(key))
        if isinstance(step, int) and value is not None:
            points.append((step, value))

    if len(points) < 2:
        return {
            "first_step": None,
            "final_step": None,
            "first": None,
            "final": None,
            "delta": None,
        }

    first_step, first = points[0]
    final_step, final = points[-1]
    return {
        "first_step": first_step,
        "final_step": final_step,
        "first": round(first, 6),
        "final": round(final, 6),
        "delta": round(final - first, 6),
    }


def _audit_step(path: str) -> int:
    match = STEP_RE.search(Path(path).stem)
    if not match:
        return -1
    return int(match.group(1))


def _latest_rollout_summary(bundle: dict[str, Any]) -> dict[str, Any]:
    audits = bundle.get("rollout_audits", [])
    if not isinstance(audits, list) or not audits:
        return {"step": None, "mean_reward": None, "gate_passed": False}

    latest = max(audits, key=lambda audit: _audit_step(str(audit.get("path", ""))))
    reward = latest.get("recomputed_reward", {})
    return {
        "step": _audit_step(str(latest.get("path", ""))),
        "mean_reward": round(float(reward.get("mean") or 0.0), 6),
        "gate_passed": bool(latest.get("gate_passed")),
    }


def _decision(
    *,
    status: str,
    bundle_passed: bool,
    detector_passed: bool,
    deltas: dict[str, dict[str, Any]],
    min_p50_delta: float,
    max_strict_drop: float,
) -> dict[str, Any]:
    failures: list[str] = []
    if status != "COMPLETED":
        failures.append(f"status={status}")
    if not bundle_passed:
        failures.append("bundle_gate_failed")
    if not detector_passed:
        failures.append("detector_mimic_gate_failed")

    p50_delta = deltas["mix_v2_p5050"]["delta"]
    if p50_delta is None:
        failures.append("mix_v2_p5050_delta_missing")
    elif p50_delta < min_p50_delta:
        failures.append(
            f"mix_v2_p5050_delta {p50_delta:+.6f} below {min_p50_delta:+.6f}"
        )

    min_strict_delta = -max_strict_drop
    for label in ("v02_strict", "v03_strict"):
        delta = deltas[label]["delta"]
        if delta is None:
            failures.append(f"{label}_delta_missing")
        elif delta < min_strict_delta:
            failures.append(f"{label}_delta {delta:+.6f} below {min_strict_delta:+.6f}")

    return {"passed": not failures, "failures": failures}


def _score_candidate(deltas: dict[str, dict[str, Any]]) -> float:
    return round(sum(float(deltas[label]["delta"] or 0.0) for label in EVAL_METRICS), 6)


def _summarize_run(
    run_dir: Path,
    *,
    min_p50_delta: float,
    max_strict_drop: float,
) -> dict[str, Any]:
    run = _load_json(run_dir / "run.json").get("run", {})
    usage = _load_json(run_dir / "usage.json") if (run_dir / "usage.json").exists() else {}
    metrics = _load_json(run_dir / "metrics.json").get("metrics", [])
    bundle = _load_json(run_dir / "bundle_report.json")
    if not isinstance(metrics, list):
        metrics = []

    deltas = {label: _metric_delta(metrics, key) for label, key in EVAL_METRICS.items()}
    detector_summary = bundle.get("detector_mimic", {}).get("summary", {})
    decision = _decision(
        status=str(run.get("status") or ""),
        bundle_passed=bool(bundle.get("promotion_gate", {}).get("passed")),
        detector_passed=bool(detector_summary.get("gate_passed")),
        deltas=deltas,
        min_p50_delta=min_p50_delta,
        max_strict_drop=max_strict_drop,
    )

    return {
        "run_id": run.get("id") or bundle.get("run_id") or run_dir.name,
        "run_name": run.get("name"),
        "run_dir": str(run_dir),
        "status": run.get("status"),
        "model": run.get("base_model"),
        "learning_rate": run.get("learning_rate"),
        "max_steps": run.get("max_steps"),
        "total_cost_usd": round(float(usage.get("total_cost_usd") or 0.0), 6),
        "total_tokens": int(usage.get("total_tokens") or 0),
        "bundle_gate_passed": bool(bundle.get("promotion_gate", {}).get("passed")),
        "detector_mimic": {
            "gate_passed": bool(detector_summary.get("gate_passed")),
            "total_rows": int(detector_summary.get("total_rows") or 0),
            "false_positive_rows": int(detector_summary.get("false_positive_rows") or 0),
            "false_negative_rows": int(detector_summary.get("false_negative_rows") or 0),
        },
        "latest_rollout": _latest_rollout_summary(bundle),
        "eval_deltas": deltas,
        "selection_gate": decision,
        "selection_score": _score_candidate(deltas),
    }


def build_prime_ablation_matrix(
    *,
    run_dirs: list[Path],
    output_path: Path,
    min_p50_delta: float = 0.0,
    max_strict_drop: float = 0.0,
) -> dict[str, Any]:
    """Build a cross-run Prime ablation matrix from saved run artifacts."""
    rows = [
        _summarize_run(
            run_dir,
            min_p50_delta=min_p50_delta,
            max_strict_drop=max_strict_drop,
        )
        for run_dir in run_dirs
    ]
    passed = [row for row in rows if row["selection_gate"]["passed"]]
    best = max(passed, key=lambda row: row["selection_score"], default=None)
    report = {
        "artifact": "prime_ablation_matrix",
        "criteria": {
            "min_p50_delta": min_p50_delta,
            "max_strict_drop": max_strict_drop,
            "requires_completed_status": True,
            "requires_bundle_gate": True,
            "requires_detector_mimic_gate": True,
        },
        "summary": {
            "run_count": len(rows),
            "selection_pass_count": len(passed),
            "best_run_id": best["run_id"] if best else None,
            "best_run_name": best["run_name"] if best else None,
            "best_model": best["model"] if best else None,
        },
        "runs": rows,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option(
    "--runs-root",
    type=click.Path(path_type=Path, file_okay=False),
    default=DEFAULT_RUNS_ROOT,
    help="Directory containing per-run Prime audit folders.",
)
@click.option(
    "--run-dir",
    "run_dirs",
    multiple=True,
    type=click.Path(path_type=Path, file_okay=False),
    help="Specific run directory. Defaults to discovering runs under --runs-root.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    default=DEFAULT_OUTPUT,
    help="Ablation matrix JSON output.",
)
@click.option("--min-p50-delta", default=0.0, type=float)
@click.option("--max-strict-drop", default=0.0, type=float)
def cli(
    runs_root: Path,
    run_dirs: tuple[Path, ...],
    output_path: Path,
    min_p50_delta: float,
    max_strict_drop: float,
) -> None:
    """Summarize saved Prime run bundles into one selection matrix."""
    selected_dirs = list(run_dirs) if run_dirs else _discover_run_dirs(runs_root)
    if not selected_dirs:
        raise click.ClickException("no run directories with bundle_report.json found")
    report = build_prime_ablation_matrix(
        run_dirs=selected_dirs,
        output_path=output_path,
        min_p50_delta=min_p50_delta,
        max_strict_drop=max_strict_drop,
    )
    summary = report["summary"]
    click.echo(
        "runs={runs} selection_pass={passed} best={best} report={report}".format(
            runs=summary["run_count"],
            passed=summary["selection_pass_count"],
            best=summary["best_run_id"] or "none",
            report=output_path,
        )
    )


if __name__ == "__main__":
    cli()
