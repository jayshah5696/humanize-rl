from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from scripts.eval.build_prime_ablation_matrix import build_prime_ablation_matrix, cli


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def _run_dir(
    root: Path,
    run_id: str,
    *,
    status: str = "COMPLETED",
    model: str = "Qwen/Qwen3.5-0.8B",
    p50_final: float = 0.55,
    v02_final: float = 0.12,
    v03_final: float = 0.14,
) -> Path:
    run_dir = root / run_id
    _write_json(
        run_dir / "run.json",
        {
            "run": {
                "id": run_id,
                "name": f"{run_id}-name",
                "status": status,
                "base_model": model,
                "learning_rate": 0.00005,
                "max_steps": 50,
            }
        },
    )
    _write_json(
        run_dir / "usage.json",
        {"total_cost_usd": 0.0347, "total_tokens": 740403},
    )
    _write_json(
        run_dir / "metrics.json",
        {
            "metrics": [
                {
                    "step": 0,
                    "eval/eval_mix_v2_p5050_env0315/avg@1": 0.40,
                    "eval/eval_v02_strict_env0315/avg@1": 0.02,
                    "eval/eval_v03_strict_env0315/avg@1": 0.01,
                },
                {
                    "step": 50,
                    "eval/eval_mix_v2_p5050_env0315/avg@1": p50_final,
                    "eval/eval_v02_strict_env0315/avg@1": v02_final,
                    "eval/eval_v03_strict_env0315/avg@1": v03_final,
                },
            ]
        },
    )
    _write_json(
        run_dir / "bundle_report.json",
        {
            "artifact": "prime_run_audit_bundle",
            "run_id": run_id,
            "rollout_audits": [
                {
                    "path": f"{run_dir}/audit_step50_env0315.json",
                    "gate_passed": True,
                    "recomputed_reward": {"mean": 0.52},
                }
            ],
            "detector_mimic": {
                "summary": {
                    "gate_passed": True,
                    "total_rows": 22,
                    "false_positive_rows": 0,
                    "false_negative_rows": 0,
                }
            },
            "promotion_gate": {"passed": True, "failures": []},
        },
    )
    return run_dir


def test_build_prime_ablation_matrix_selects_best_passing_run(tmp_path: Path) -> None:
    good = _run_dir(tmp_path, "good_run", p50_final=0.45, v02_final=0.08, v03_final=0.09)
    best = _run_dir(tmp_path, "best_run", p50_final=0.50, v02_final=0.12, v03_final=0.16)
    bad = _run_dir(tmp_path, "bad_run", p50_final=0.52, v02_final=-0.20, v03_final=0.11)
    output = tmp_path / "matrix.json"

    report = build_prime_ablation_matrix(
        run_dirs=[good, best, bad],
        output_path=output,
    )

    assert report["summary"]["run_count"] == 3
    assert report["summary"]["selection_pass_count"] == 2
    assert report["summary"]["best_run_id"] == "best_run"
    bad_row = next(row for row in report["runs"] if row["run_id"] == "bad_run")
    assert bad_row["selection_gate"]["passed"] is False
    assert "v02_strict_delta -0.220000 below -0.000000" in bad_row[
        "selection_gate"
    ]["failures"]
    assert output.exists()


def test_build_prime_ablation_matrix_rejects_stopped_run(tmp_path: Path) -> None:
    stopped = _run_dir(tmp_path, "stopped_run", status="STOPPED")

    report = build_prime_ablation_matrix(
        run_dirs=[stopped],
        output_path=tmp_path / "matrix.json",
    )

    assert report["summary"]["selection_pass_count"] == 0
    assert report["runs"][0]["selection_gate"]["failures"] == ["status=STOPPED"]


def test_build_prime_ablation_matrix_cli_discovers_runs(tmp_path: Path) -> None:
    _run_dir(tmp_path, "good_run")
    output = tmp_path / "matrix.json"

    result = CliRunner().invoke(
        cli,
        [
            "--runs-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0
    assert "runs=1 selection_pass=1 best=good_run" in result.output
    assert json.loads(output.read_text())["summary"]["best_run_id"] == "good_run"
