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
import subprocess
import tarfile
import tomllib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

DEFAULT_CONFIG = Path(
    "configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml"
)
DEFAULT_OUTPUT_DIR = Path("runs/prime_sft_launch_kit/qwen35_2b_env0315")
DEFAULT_PRIME_RL_REF = "d700753"
EXPECTED_SPLIT_ROWS_BY_DATASET = {
    "jayshah5696/humanize-rl-prime-sft-messages-env0314": {
        "train": 4313,
        "validation": 239,
        "test": 241,
    },
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50": {
        "train": 4358,
        "validation": 242,
        "test": 243,
    },
}
KIT_DIR_NAME = "prime_sft_launch_kit"


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def current_git_commit() -> str | None:
    return _run_git(["rev-parse", "HEAD"])


def current_git_dirty() -> bool:
    return bool(_run_git(["status", "--porcelain"]))


def build_manifest(
    *,
    config_path: Path,
    prime_rl_ref: str,
    git_commit: str | None,
    git_dirty: bool,
) -> dict[str, Any]:
    config = _load_toml(config_path)
    dataset_name = str(config.get("data", {}).get("name") or "")
    val_data = config.get("val", {}).get("data", {})
    return {
        "artifact": "prime_sft_launch_kit",
        "created_at": datetime.now(UTC).isoformat(),
        "prime_rl_ref": prime_rl_ref,
        "git": {
            "commit": git_commit,
            "dirty": git_dirty,
        },
        "config": {
            "source_path": str(config_path),
            "sha256": _sha256(config_path),
            "model": config.get("model", {}).get("name"),
            "renderer": config.get("renderer", {}).get("name"),
            "data": config.get("data", {}).get("name"),
            "train_splits": config.get("data", {}).get("splits"),
            "validation_splits": val_data.get("splits"),
            "output_dir": config.get("output_dir"),
            "wandb_project": config.get("wandb", {}).get("project"),
            "wandb_name": config.get("wandb", {}).get("name"),
            "save_adapter_separately": config.get("ckpt", {})
            .get("weights", {})
            .get("save_adapter_separately"),
        },
        "expected_dataset": {
            "name": dataset_name,
            "splits": EXPECTED_SPLIT_ROWS_BY_DATASET.get(dataset_name, {}),
            "format": "messages",
        },
    }


def _run_sft_script(prime_rl_ref: str) -> str:
    return f"""#!/usr/bin/env bash
set -euo pipefail

if [ -z "${{HF_TOKEN:-}}" ]; then
  echo "HF_TOKEN is required" >&2
  exit 2
fi

if [ -z "${{WANDB_API_KEY:-}}" ]; then
  echo "WANDB_API_KEY is required" >&2
  exit 2
fi

export HF_TOKEN
export WANDB_API_KEY
export WANDB_MODE="${{WANDB_MODE:-online}}"
export PATH="$HOME/.local/bin:$PATH"

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y git curl
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

cd /workspace
if [ ! -d prime-rl ]; then
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git prime-rl
fi

cd /workspace/prime-rl
git fetch --depth 1 origin {prime_rl_ref} || true
git checkout {prime_rl_ref}

uv run sft @ /workspace/{KIT_DIR_NAME}/config.toml
"""


def _readme(prime_rl_ref: str, archive_path: Path) -> str:
    archive_local_path = archive_path.as_posix()
    archive_name = archive_path.name
    return f"""# Prime SFT Launch Kit

This kit is for Prime open `prime-rl` dataset SFT. Do not submit this config
through `prime train`; Hosted Training uses the env/rollout schema.

Upload the archive:

```bash
prime --plain sandbox upload <sandbox_id> {archive_local_path} /tmp/{archive_name}
```

Unpack it in the sandbox:

```bash
prime --plain sandbox run <sandbox_id> -- bash -lc 'mkdir -p /workspace && tar -xzf /tmp/{archive_name} -C /workspace'
```

Run SFT with secrets passed from your local shell:

```bash
prime --plain sandbox run <sandbox_id> \\
  -e HF_TOKEN="$HF_TOKEN" \\
  -e WANDB_API_KEY="$WANDB_API_KEY" \\
  --timeout 86400 \\
  -- bash -lc 'bash /workspace/{KIT_DIR_NAME}/run_sft.sh'
```

The runner clones `PrimeIntellect-ai/prime-rl`, checks out `{prime_rl_ref}`, and
executes:

```bash
uv run sft @ /workspace/{KIT_DIR_NAME}/config.toml
```
"""


def write_launch_kit(
    *,
    config_path: Path,
    output_dir: Path,
    prime_rl_ref: str,
    git_commit: str | None,
    git_dirty: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.toml")

    manifest = build_manifest(
        config_path=config_path,
        prime_rl_ref=prime_rl_ref,
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    run_script = output_dir / "run_sft.sh"
    run_script.write_text(_run_sft_script(prime_rl_ref))
    run_script.chmod(0o755)

    archive_path = output_dir.with_suffix(".tar.gz")
    (output_dir / "README.md").write_text(_readme(prime_rl_ref, archive_path))

    with tarfile.open(archive_path, "w:gz") as tar:
        for path in sorted(output_dir.iterdir()):
            tar.add(path, arcname=f"{KIT_DIR_NAME}/{path.name}")

    return archive_path


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(path_type=Path),
    default=DEFAULT_CONFIG,
    show_default=True,
    help="Prime prime-rl SFT config to package.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=DEFAULT_OUTPUT_DIR,
    show_default=True,
    help="Directory to write launch-kit files.",
)
@click.option(
    "--prime-rl-ref",
    default=DEFAULT_PRIME_RL_REF,
    show_default=True,
    help="PrimeIntellect-ai/prime-rl commit or ref to check out in the sandbox.",
)
def cli(config_path: Path, output_dir: Path, prime_rl_ref: str) -> None:
    """Create a secret-free Prime SFT launch kit for a sandbox/pod."""

    archive_path = write_launch_kit(
        config_path=config_path,
        output_dir=output_dir,
        prime_rl_ref=prime_rl_ref,
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
    )
    click.echo(f"launch_kit={output_dir}")
    click.echo(f"archive={archive_path}")
    click.echo(
        f"next=prime sandbox upload <sandbox_id> {archive_path} /tmp/{archive_path.name}"
    )


if __name__ == "__main__":
    cli()
