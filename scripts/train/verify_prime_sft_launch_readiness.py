# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "click>=8.1",
# ]
# ///
from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path
from typing import Any

import click

EXPECTED_PRIME_RL_REF = "d700753"


def _load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise click.ClickException(f"{path} must contain a JSON object")
    return data


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


def _as_check_reports(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    checks = []
    for item in value:
        if not isinstance(item, dict):
            continue
        checks.append(
            {
                "detail": str(item.get("detail") or ""),
                "name": str(item.get("name") or ""),
                "passed": bool(item.get("passed")),
            }
        )
    return checks


def _as_mapping(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return value


def _runner_uses_prime_rl_ref(runner_text: str, expected_prime_rl_ref: str) -> bool:
    return (
        f"git fetch --depth 1 origin {expected_prime_rl_ref}" in runner_text
        and f"git checkout {expected_prime_rl_ref}" in runner_text
        and 'url."https://github.com/".insteadOf git@github.com:' in runner_text
        and 'url."https://github.com/".insteadOf ssh://git@github.com/' in runner_text
        and "git config -f .gitmodules" in runner_text
        and 'https_url="https://github.com/' in runner_text
        and "git submodule sync --recursive" in runner_text
        and "git submodule update --init --recursive --force" in runner_text
        and "cat > sitecustomize.py" in runner_text
        and "torch.backends.cudnn.enabled = False" in runner_text
        and 'PYTHONPATH="/workspace/prime-rl:${PYTHONPATH:-}"' in runner_text
        and "TORCH_CUDNN_V8_API_DISABLED=1" in runner_text
        and "apt_install cuda-nvcc-12-8 g++-12 ninja-build" in runner_text
        and "import flash_attn_2_cuda" in runner_text
        and 'FLASH_ATTN_CUDA_ARCHS="80"' in runner_text
        and "--no-build-isolation --no-deps flash-attn==2.8.3.post1" in runner_text
    )


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _archive_member_bytes(archive_path: Path, member_name: str) -> bytes | None:
    with tarfile.open(archive_path, "r:gz") as tar:
        try:
            member = tar.getmember(member_name)
        except KeyError:
            return None
        handle = tar.extractfile(member)
        if handle is None:
            return None
        return handle.read()


def _archive_members(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, "r:gz") as tar:
        return sorted(tar.getnames())


def build_launch_readiness_report(
    *,
    config_path: Path,
    preflight_report_path: Path,
    launch_manifest_path: Path,
    eval_manifest_path: Path,
    launch_archive_path: Path,
    output_path: Path,
    expected_prime_rl_ref: str = EXPECTED_PRIME_RL_REF,
) -> dict[str, Any]:
    """Verify the SFT launch handoff artifacts are mutually consistent."""
    failures: list[str] = []
    config_sha = _sha256(config_path)
    preflight = _load_json(preflight_report_path)
    launch_manifest = _load_json(launch_manifest_path)
    eval_manifest = _load_json(eval_manifest_path)
    archive_members = _archive_members(launch_archive_path)
    launch_kit_dir = launch_manifest_path.parent

    if preflight.get("artifact") != "prime_sft_preflight":
        failures.append("preflight artifact is not prime_sft_preflight")
    if launch_manifest.get("artifact") != "prime_sft_launch_kit":
        failures.append("launch manifest artifact is not prime_sft_launch_kit")
    if eval_manifest.get("artifact") != "sft_eval_manifest":
        failures.append("eval manifest artifact is not sft_eval_manifest")

    prime_rl_ref = str(launch_manifest.get("prime_rl_ref") or "")
    if prime_rl_ref != expected_prime_rl_ref:
        actual = prime_rl_ref or "<missing>"
        failures.append(
            f"launch kit prime_rl_ref {actual} != {expected_prime_rl_ref}"
        )

    config_text = str(config_path)
    if preflight.get("config") != config_text:
        failures.append("preflight config path does not match launch config")

    preflight_gate = preflight.get("gate", {})
    failed_checks = _as_list(preflight_gate.get("failed_checks"))
    preflight_checks = _as_check_reports(preflight.get("checks"))
    preflight_policy = _as_mapping(preflight.get("launch_policy"))
    wandb_policy = _as_mapping(preflight_policy.get("wandb_source"))
    if not preflight_policy:
        failures.append("preflight launch_policy is missing")
    elif preflight_policy.get("runner") != "prime_sandbox":
        failures.append("preflight runner policy is not prime_sandbox")
    if not wandb_policy:
        failures.append("preflight W&B source policy is missing")
    else:
        if bool(wandb_policy.get("prime_secret_allowed")):
            failures.append("preflight allows Prime-only W&B secret for sandbox launch")
        if not bool(wandb_policy.get("local_env_required")):
            failures.append("preflight does not require local WANDB_API_KEY")
    if not bool(preflight_gate.get("passed")):
        detail = ", ".join(failed_checks) if failed_checks else "unknown"
        failures.append(f"preflight gate failed: {detail}")

    launch_config = launch_manifest.get("config", {})
    if not isinstance(launch_config, dict):
        launch_config = {}
        failures.append("launch manifest missing config object")
    if launch_config.get("source_path") != config_text:
        failures.append("launch kit config source_path does not match launch config")
    if launch_config.get("sha256") != config_sha:
        failures.append("launch kit config sha256 does not match current config")

    eval_config = eval_manifest.get("config", {})
    if not isinstance(eval_config, dict):
        eval_config = {}
        failures.append("eval manifest missing config object")
    if eval_config.get("path") != config_text:
        failures.append("eval manifest config path does not match launch config")

    required_eval_artifacts = eval_manifest.get("required_artifacts")
    if not isinstance(required_eval_artifacts, dict):
        required_eval_artifacts = {}
        failures.append("eval manifest missing required_artifacts object")

    after_sft_template = eval_manifest.get("after_sft_template")
    after_sft_template_report: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "current_sha256": None,
        "exists": False,
    }
    if not isinstance(after_sft_template, dict):
        failures.append("eval manifest missing after_sft_template object")
    else:
        template_path_text = str(after_sft_template.get("path") or "")
        after_sft_template_report["path"] = template_path_text or None
        after_sft_template_report["sha256"] = after_sft_template.get("sha256")
        if not template_path_text:
            failures.append("eval manifest after_sft_template path is empty")
        elif required_eval_artifacts.get("after_sft_template") != template_path_text:
            failures.append("eval manifest after_sft_template required path mismatch")
        else:
            template_path = Path(template_path_text)
            after_sft_template_report["exists"] = template_path.exists()
            if not template_path.is_file():
                failures.append("after-SFT template is missing")
            else:
                template_sha = _sha256(template_path)
                after_sft_template_report["current_sha256"] = template_sha
                if after_sft_template.get("sha256") != template_sha:
                    failures.append(
                        "after-SFT template sha256 does not match eval manifest"
                    )

    pangram_bulk_items_report: dict[str, Any] = {
        "path": None,
        "sha256": None,
        "current_sha256": None,
        "exists": False,
        "required_now": False,
    }
    pangram_bulk_items = eval_manifest.get("pangram_bulk_items")
    if isinstance(pangram_bulk_items, dict):
        pangram_path_text = str(pangram_bulk_items.get("path") or "")
        pangram_bulk_items_report["path"] = pangram_path_text or None
        pangram_bulk_items_report["sha256"] = pangram_bulk_items.get("sha256")
        pangram_bulk_items_report["required_now"] = bool(
            pangram_bulk_items.get("required_now")
        )
        if pangram_path_text:
            pangram_path = Path(pangram_path_text)
            pangram_bulk_items_report["exists"] = pangram_path.exists()
            if pangram_path.is_file():
                pangram_sha = _sha256(pangram_path)
                pangram_bulk_items_report["current_sha256"] = pangram_sha
                if pangram_bulk_items.get("sha256") not in (None, pangram_sha):
                    failures.append(
                        "Pangram bulk items sha256 does not match eval manifest"
                    )
            elif bool(pangram_bulk_items.get("required_now")):
                failures.append("Pangram bulk items file is missing")
        elif bool(pangram_bulk_items.get("required_now")):
            failures.append("Pangram bulk items path is empty")

    packaged_eval = launch_manifest.get("sft_eval_manifest")
    if not isinstance(packaged_eval, dict):
        packaged_eval = {}
        failures.append("launch manifest missing sft_eval_manifest object")
    if packaged_eval.get("source_path") != str(eval_manifest_path):
        failures.append("launch kit eval manifest source_path does not match")
    if packaged_eval.get("sha256") != _sha256(eval_manifest_path):
        failures.append("launch kit eval manifest sha256 does not match current manifest")

    required_archive_members = {
        "prime_sft_launch_kit/README.md",
        "prime_sft_launch_kit/config.toml",
        "prime_sft_launch_kit/manifest.json",
        "prime_sft_launch_kit/run_sft.sh",
        "prime_sft_launch_kit/sft_eval_manifest.json",
    }
    missing_archive_members = sorted(required_archive_members - set(archive_members))
    for member in missing_archive_members:
        failures.append(f"launch archive missing {member}")

    kit_file_shas: dict[str, str | None] = {}
    archive_file_shas: dict[str, str | None] = {}
    runner_uses_expected_prime_rl_ref = False
    for filename, report_key in (
        ("run_sft.sh", "runner_sha256"),
        ("README.md", "readme_sha256"),
    ):
        local_path = launch_kit_dir / filename
        local_sha = None
        if local_path.exists():
            local_sha = _sha256(local_path)
            if filename == "run_sft.sh":
                runner_text = local_path.read_text()
                runner_uses_expected_prime_rl_ref = _runner_uses_prime_rl_ref(
                    runner_text,
                    expected_prime_rl_ref,
                )
                if not runner_uses_expected_prime_rl_ref:
                    failures.append(
                        "launch runner does not checkout expected prime_rl_ref "
                        f"{expected_prime_rl_ref}"
                    )
        else:
            failures.append(f"launch kit missing {filename}")
        kit_file_shas[report_key] = local_sha

        archive_member = _archive_member_bytes(
            launch_archive_path, f"prime_sft_launch_kit/{filename}"
        )
        archive_sha = _sha256_bytes(archive_member) if archive_member is not None else None
        archive_file_shas[report_key] = archive_sha
        if local_sha is not None and archive_sha is not None and archive_sha != local_sha:
            failures.append(f"launch archive {filename} does not match launch kit")

    archive_config = _archive_member_bytes(
        launch_archive_path, "prime_sft_launch_kit/config.toml"
    )
    if archive_config is not None and _sha256_bytes(archive_config) != config_sha:
        failures.append("launch archive config does not match current config")

    archive_manifest = _archive_member_bytes(
        launch_archive_path, "prime_sft_launch_kit/manifest.json"
    )
    if archive_manifest is not None:
        if _sha256_bytes(archive_manifest) != _sha256(launch_manifest_path):
            failures.append("launch archive manifest does not match launch manifest")

    archive_eval_manifest = _archive_member_bytes(
        launch_archive_path, "prime_sft_launch_kit/sft_eval_manifest.json"
    )
    if archive_eval_manifest is not None:
        if _sha256_bytes(archive_eval_manifest) != _sha256(eval_manifest_path):
            failures.append("launch archive eval manifest does not match current manifest")

    report = {
        "artifact": "prime_sft_launch_readiness",
        "passed": not failures,
        "failures": failures,
        "config": {
            "path": config_text,
            "sha256": config_sha,
        },
        "preflight": {
            "path": str(preflight_report_path),
            "sha256": _sha256(preflight_report_path),
            "gate_passed": bool(preflight_gate.get("passed")),
            "failed_checks": failed_checks,
            "checks": preflight_checks,
            "launch_policy": preflight_policy,
        },
        "launch_kit": {
            "manifest_path": str(launch_manifest_path),
            "prime_rl_ref": prime_rl_ref or None,
            "expected_prime_rl_ref": expected_prime_rl_ref,
            "runner_uses_expected_prime_rl_ref": runner_uses_expected_prime_rl_ref,
            "config_source_path": launch_config.get("source_path"),
            "config_sha256": launch_config.get("sha256"),
            "eval_manifest_source_path": packaged_eval.get("source_path"),
            "eval_manifest_sha256": packaged_eval.get("sha256"),
            "runner_sha256": kit_file_shas["runner_sha256"],
            "readme_sha256": kit_file_shas["readme_sha256"],
        },
        "launch_archive": {
            "path": str(launch_archive_path),
            "sha256": _sha256(launch_archive_path),
            "members": archive_members,
            "runner_sha256": archive_file_shas["runner_sha256"],
            "readme_sha256": archive_file_shas["readme_sha256"],
        },
        "sft_eval_manifest": {
            "path": str(eval_manifest_path),
            "checkpoint_id": eval_manifest.get("checkpoint_id"),
            "checkpoint_slug": eval_manifest.get("checkpoint_slug"),
            "promotion_root": eval_manifest.get("promotion_root"),
            "after_sft_template": after_sft_template_report,
            "pangram_bulk_items": pangram_bulk_items_report,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


@click.command(context_settings={"show_default": True})
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Prime prime-rl SFT config intended for launch.",
)
@click.option(
    "--preflight-report",
    "preflight_report_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Report from prime_sft_preflight.py.",
)
@click.option(
    "--launch-manifest",
    "launch_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="manifest.json from the Prime SFT launch kit.",
)
@click.option(
    "--eval-manifest",
    "eval_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="SFT eval/promotion manifest packaged with the launch kit.",
)
@click.option(
    "--launch-archive",
    "launch_archive_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    required=True,
    help="Tarball that will be uploaded to the Prime sandbox.",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path, dir_okay=False),
    required=True,
    help="Launch-readiness report JSON.",
)
@click.option(
    "--expected-prime-rl-ref",
    default=EXPECTED_PRIME_RL_REF,
    show_default=True,
    help="Pinned PrimeIntellect-ai/prime-rl ref expected in the launch kit.",
)
@click.option("--no-fail-on-gate", is_flag=True)
def cli(
    config_path: Path,
    preflight_report_path: Path,
    launch_manifest_path: Path,
    eval_manifest_path: Path,
    launch_archive_path: Path,
    output_path: Path,
    expected_prime_rl_ref: str,
    no_fail_on_gate: bool,
) -> None:
    """Verify Prime SFT launch artifacts before spending a tracked run."""
    report = build_launch_readiness_report(
        config_path=config_path,
        preflight_report_path=preflight_report_path,
        launch_manifest_path=launch_manifest_path,
        eval_manifest_path=eval_manifest_path,
        launch_archive_path=launch_archive_path,
        output_path=output_path,
        expected_prime_rl_ref=expected_prime_rl_ref,
    )
    click.echo(
        "prime_sft_launch_readiness={gate} failures={failures} report={report}".format(
            gate="pass" if report["passed"] else "fail",
            failures=len(report["failures"]),
            report=output_path,
        )
    )
    if not report["passed"] and not no_fail_on_gate:
        raise click.ClickException("Prime SFT launch readiness failed")


if __name__ == "__main__":
    cli()
