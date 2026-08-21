import hashlib
import json
import shutil
import tarfile
from pathlib import Path

from scripts.train.verify_prime_sft_launch_readiness import (
    build_launch_readiness_report,
)

PINNED_RUN_SCRIPT = """#!/usr/bin/env bash
git fetch --depth 1 origin d700753 || true
git checkout d700753
git config --global url."https://github.com/".insteadOf git@github.com:
git config --global url."https://github.com/".insteadOf ssh://git@github.com/
cat > sitecustomize.py <<'PY'
try:
    import torch

    torch.backends.cudnn.enabled = False
except Exception:
    pass
PY
export PYTHONPATH="/workspace/prime-rl:${PYTHONPATH:-}"
export TORCH_CUDNN_V8_API_DISABLED=1
git config -f .gitmodules --get-regexp '^submodule\\..*\\.url$' | while read -r key url; do
  case "$url" in
    git@github.com:*) https_url="https://github.com/${url#git@github.com:}" ;;
    ssh://git@github.com/*) https_url="https://github.com/${url#ssh://git@github.com/}" ;;
    *) https_url="$url" ;;
  esac
  git config -f .gitmodules "$key" "$https_url"
done
git submodule sync --recursive
git submodule update --init --recursive --force
PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')"
if ! "$PYTHON_BIN" -c 'import flash_attn_2_cuda' >/dev/null 2>&1; then
  apt_install cuda-nvcc-12-8 g++-12 ninja-build
  export FLASH_ATTN_CUDA_ARCHS="80"
  "$PYTHON_BIN" -m pip install --no-cache-dir --no-build-isolation --no-deps flash-attn==2.8.3.post1
fi
uv run sft @ /workspace/prime_sft_launch_kit/config.toml
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _write_config(path: Path) -> None:
    path.write_text(
        """
output_dir = "outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315"

[model]
name = "Qwen/Qwen3.5-2B"

[data]
name = "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat"
splits = ["train"]
""".strip()
        + "\n"
    )


def _write_launch_archive(
    *,
    archive_path: Path,
    config: Path,
    launch_manifest: Path,
    eval_manifest: Path,
    run_script: str = PINNED_RUN_SCRIPT,
    readme: str = "# kit\n",
) -> None:
    kit_dir = archive_path.parent / "prime_sft_launch_kit"
    if kit_dir.exists():
        shutil.rmtree(kit_dir)
    kit_dir.mkdir()
    (kit_dir / "config.toml").write_text(config.read_text())
    (kit_dir / "manifest.json").write_text(launch_manifest.read_text())
    (kit_dir / "sft_eval_manifest.json").write_text(eval_manifest.read_text())
    (kit_dir / "run_sft.sh").write_text(run_script)
    (kit_dir / "README.md").write_text(readme)
    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(kit_dir.iterdir()):
            tar.add(path, arcname=f"prime_sft_launch_kit/{path.name}")


def _write_matching_artifacts(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    config = tmp_path / "config.toml"
    preflight = tmp_path / "preflight.json"
    eval_manifest = tmp_path / "sft_eval_manifest.json"
    after_sft_template = tmp_path / "after_sft_template.toml"
    pangram_bulk_items = tmp_path / "pangram_bulk_items.json"
    launch_manifest = tmp_path / "launch_manifest.json"
    launch_archive = tmp_path / "launch_kit.tar.gz"
    _write_config(config)
    after_sft_template.write_text(
        'model = "Qwen/Qwen3.5-2B"\n'
        'checkpoint_id = "FILL_WITH_READY_SFT_CHECKPOINT_ID"\n'
    )
    pangram_bulk_items.write_text(
        json.dumps({"artifact": "pangram_bulk_items", "items": []}) + "\n"
    )
    _write_json(
        preflight,
        {
            "artifact": "prime_sft_preflight",
            "checks": [
                {
                    "detail": "Prime CLI version: 0.6.14",
                    "name": "Prime CLI version",
                    "passed": True,
                }
            ],
            "config": str(config),
            "gate": {"passed": True, "failed_checks": []},
            "launch_policy": {
                "runner": "prime_sandbox",
                "wandb_source": {
                    "local_env_required": True,
                    "prime_secret_allowed": False,
                },
            },
        },
    )
    _write_json(
        eval_manifest,
        {
            "artifact": "sft_eval_manifest",
            "checkpoint_id": "READY_SFT_CHECKPOINT_ID",
            "checkpoint_slug": None,
            "config": {"path": str(config), "exists": True, "required_now": True},
            "after_sft_template": {
                "path": str(after_sft_template),
                "exists": True,
                "required_now": True,
                "sha256": _sha256(after_sft_template),
            },
            "pangram_bulk_items": {
                "path": str(pangram_bulk_items),
                "exists": True,
                "required_now": False,
                "sha256": _sha256(pangram_bulk_items),
            },
            "promotion_root": "runs/prime_sft_promotion/TEMPLATE",
            "required_artifacts": {
                "after_sft_template": str(after_sft_template),
            },
        },
    )
    _write_json(
        launch_manifest,
        {
            "artifact": "prime_sft_launch_kit",
            "prime_rl_ref": "d700753",
            "config": {
                "source_path": str(config),
                "sha256": _sha256(config),
                "data": "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat",
                "model": "Qwen/Qwen3.5-2B",
            },
            "sft_eval_manifest": {
                "source_path": str(eval_manifest),
                "sha256": _sha256(eval_manifest),
                "checkpoint_id": "READY_SFT_CHECKPOINT_ID",
                "promotion_root": "runs/prime_sft_promotion/TEMPLATE",
            },
        },
    )
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=launch_manifest,
        eval_manifest=eval_manifest,
    )
    (launch_manifest.parent / "run_sft.sh").write_text(PINNED_RUN_SCRIPT)
    (launch_manifest.parent / "README.md").write_text("# kit\n")
    return config, preflight, launch_manifest, eval_manifest, launch_archive


def test_build_launch_readiness_report_passes_matching_artifacts(tmp_path: Path) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is True
    assert report["failures"] == []
    assert report["preflight"]["gate_passed"] is True
    assert report["preflight"]["sha256"] == _sha256(preflight)
    assert report["preflight"]["checks"] == [
        {
            "detail": "Prime CLI version: 0.6.14",
            "name": "Prime CLI version",
            "passed": True,
        }
    ]
    assert report["preflight"]["launch_policy"] == {
        "runner": "prime_sandbox",
        "wandb_source": {
            "local_env_required": True,
            "prime_secret_allowed": False,
        },
    }
    assert report["launch_kit"]["config_sha256"] == _sha256(config)
    assert report["launch_kit"]["prime_rl_ref"] == "d700753"
    assert report["launch_kit"]["expected_prime_rl_ref"] == "d700753"
    assert report["launch_kit"]["runner_uses_expected_prime_rl_ref"] is True
    assert report["launch_archive"]["path"] == str(launch_archive)
    assert report["launch_archive"]["runner_sha256"] == _sha256(
        launch_manifest.parent / "run_sft.sh"
    )
    assert report["launch_archive"]["readme_sha256"] == _sha256(
        launch_manifest.parent / "README.md"
    )
    assert "prime_sft_launch_kit/sft_eval_manifest.json" in report["launch_archive"][
        "members"
    ]
    assert report["sft_eval_manifest"]["checkpoint_id"] == "READY_SFT_CHECKPOINT_ID"
    assert report["sft_eval_manifest"]["pangram_bulk_items"]["exists"] is True
    assert report["sft_eval_manifest"]["pangram_bulk_items"]["current_sha256"] == _sha256(
        tmp_path / "pangram_bulk_items.json"
    )
    assert output.exists()


def test_build_launch_readiness_report_blocks_wandb_missing_preflight(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    _write_json(
        preflight,
        {
            "artifact": "prime_sft_preflight",
            "config": str(config),
            "gate": {
                "passed": False,
                "failed_checks": ["WANDB_API_KEY source"],
            },
        },
    )
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "preflight gate failed: WANDB_API_KEY source" in report["failures"]


def test_build_launch_readiness_report_rejects_prime_secret_only_policy(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    payload = json.loads(preflight.read_text())
    payload["launch_policy"]["wandb_source"] = {
        "local_env_required": False,
        "prime_secret_allowed": True,
    }
    _write_json(preflight, payload)
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "preflight allows Prime-only W&B secret for sandbox launch" in report[
        "failures"
    ]


def test_build_launch_readiness_report_rejects_unpinned_prime_rl_ref(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    payload = json.loads(launch_manifest.read_text())
    payload["prime_rl_ref"] = "main"
    _write_json(launch_manifest, payload)
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=launch_manifest,
        eval_manifest=eval_manifest,
    )
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch kit prime_rl_ref main != d700753" in report["failures"]


def test_build_launch_readiness_report_rejects_runner_wrong_prime_rl_ref(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    bad_runner = PINNED_RUN_SCRIPT.replace("d700753", "main")
    (launch_manifest.parent / "run_sft.sh").write_text(bad_runner)
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=launch_manifest,
        eval_manifest=eval_manifest,
        run_script=bad_runner,
    )
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch runner does not checkout expected prime_rl_ref d700753" in report[
        "failures"
    ]


def test_build_launch_readiness_report_fails_config_hash_mismatch(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    payload = json.loads(launch_manifest.read_text())
    payload["config"]["sha256"] = "bad-sha"
    _write_json(launch_manifest, payload)
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch kit config sha256 does not match current config" in report["failures"]


def test_build_launch_readiness_report_fails_stale_launch_archive(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    stale_manifest = tmp_path / "stale_manifest.json"
    stale_payload = json.loads(launch_manifest.read_text())
    stale_payload["config"]["sha256"] = "stale-sha"
    _write_json(stale_manifest, stale_payload)
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=stale_manifest,
        eval_manifest=eval_manifest,
    )
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch archive manifest does not match launch manifest" in report[
        "failures"
    ]


def test_build_launch_readiness_report_fails_stale_archive_runner(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    local_runner = launch_manifest.parent / "run_sft.sh"
    local_runner.write_text(PINNED_RUN_SCRIPT)
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=launch_manifest,
        eval_manifest=eval_manifest,
        run_script="#!/usr/bin/env bash\necho stale\n",
    )
    local_runner.write_text(PINNED_RUN_SCRIPT)
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch archive run_sft.sh does not match launch kit" in report["failures"]


def test_build_launch_readiness_report_fails_stale_archive_readme(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    local_readme = launch_manifest.parent / "README.md"
    local_readme.write_text("# kit\n")
    _write_launch_archive(
        archive_path=launch_archive,
        config=config,
        launch_manifest=launch_manifest,
        eval_manifest=eval_manifest,
        readme="# stale\n",
    )
    local_readme.write_text("# kit\n")
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "launch archive README.md does not match launch kit" in report["failures"]


def test_build_launch_readiness_report_fails_stale_after_sft_template_hash(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    payload = json.loads(eval_manifest.read_text())
    payload["after_sft_template"]["sha256"] = "stale"
    _write_json(eval_manifest, payload)
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "after-SFT template sha256 does not match eval manifest" in report[
        "failures"
    ]


def test_build_launch_readiness_report_fails_stale_pangram_bulk_items_hash(
    tmp_path: Path,
) -> None:
    config, preflight, launch_manifest, eval_manifest, launch_archive = (
        _write_matching_artifacts(tmp_path)
    )
    payload = json.loads(eval_manifest.read_text())
    payload["pangram_bulk_items"]["sha256"] = "stale"
    _write_json(eval_manifest, payload)
    output = tmp_path / "readiness.json"

    report = build_launch_readiness_report(
        config_path=config,
        preflight_report_path=preflight,
        launch_manifest_path=launch_manifest,
        eval_manifest_path=eval_manifest,
        launch_archive_path=launch_archive,
        output_path=output,
    )

    assert report["passed"] is False
    assert "Pangram bulk items sha256 does not match eval manifest" in report[
        "failures"
    ]
