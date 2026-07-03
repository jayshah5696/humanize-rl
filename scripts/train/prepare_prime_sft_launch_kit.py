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
    "jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat": {
        "train": 4358,
        "validation": 242,
        "test": 243,
    },
}
KIT_DIR_NAME = "prime_sft_launch_kit"
SFT_EVAL_MANIFEST_NAME = "sft_eval_manifest.json"


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


def _eval_manifest_summary(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"SFT eval manifest is not valid JSON: {path}") from exc
    if payload.get("artifact") != "sft_eval_manifest":
        raise click.ClickException("SFT eval manifest artifact must be sft_eval_manifest")
    return {
        "source_path": str(path),
        "kit_path": f"{KIT_DIR_NAME}/{SFT_EVAL_MANIFEST_NAME}",
        "sha256": _sha256(path),
        "checkpoint_id": payload.get("checkpoint_id"),
        "checkpoint_slug": payload.get("checkpoint_slug"),
        "promotion_root": payload.get("promotion_root"),
        "gate_order": payload.get("gate_order", []),
    }


def build_manifest(
    *,
    config_path: Path,
    prime_rl_ref: str,
    git_commit: str | None,
    git_dirty: bool,
    eval_manifest_path: Path | None = None,
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
        "sft_eval_manifest": _eval_manifest_summary(eval_manifest_path),
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

apt_install() {{
  if command -v sudo >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y "$@"
  else
    apt-get update
    apt-get install -y "$@"
  fi
}}

if ! command -v git >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt_install git curl
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
cat > sitecustomize.py <<'PY'
try:
    import torch

    torch.backends.cudnn.enabled = False
except Exception:
    pass
PY
export PYTHONPATH="/workspace/prime-rl:${{PYTHONPATH:-}}"
export TORCH_CUDNN_V8_API_DISABLED=1
git config --global url."https://github.com/".insteadOf git@github.com:
git config --global url."https://github.com/".insteadOf ssh://git@github.com/
if [ -f .gitmodules ]; then
  git config -f .gitmodules --get-regexp '^submodule\\..*\\.url$' | while read -r key url; do
    case "$url" in
      git@github.com:*) https_url="https://github.com/${{url#git@github.com:}}" ;;
      ssh://git@github.com/*) https_url="https://github.com/${{url#ssh://git@github.com/}}" ;;
      *) https_url="$url" ;;
    esac
    git config -f .gitmodules "$key" "$https_url"
  done
fi
git submodule sync --recursive
git submodule update --init --recursive --force

PYTHON_BIN="$(uv run python -c 'import sys; print(sys.executable)')"
if ! "$PYTHON_BIN" -c 'import flash_attn_2_cuda' >/dev/null 2>&1; then
  if [ ! -x /usr/local/cuda-12.8/bin/nvcc ] || ! gcc -print-prog-name=cc1plus | grep -q '^/'; then
    apt_install cuda-nvcc-12-8 g++-12 ninja-build
  fi
  "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PYTHON_BIN" -m pip install 'setuptools<81,>=77' wheel ninja
  export CUDA_HOME=/usr/local/cuda-12.8
  export PATH="$CUDA_HOME/bin:$PATH"
  export TORCH_CUDA_ARCH_LIST="8.0"
  export FLASH_ATTN_CUDA_ARCHS="80"
  export MAX_JOBS="${{MAX_JOBS:-4}}"
  "$PYTHON_BIN" -m pip install --no-cache-dir --no-build-isolation --no-deps flash-attn==2.8.3.post1
fi

uv run sft @ /workspace/{KIT_DIR_NAME}/config.toml
"""


def _readme(
    prime_rl_ref: str,
    archive_path: Path,
    *,
    includes_eval_manifest: bool,
) -> str:
    archive_local_path = archive_path.as_posix()
    archive_name = archive_path.name
    eval_manifest_note = ""
    if includes_eval_manifest:
        eval_manifest_note = f"""
This kit includes `{SFT_EVAL_MANIFEST_NAME}`. After SFT finishes, use it as the
post-SFT verification, rollout-audit, promotion, and SFT-to-RL handoff checklist.
"""
    return f"""# Prime SFT Launch Kit

This kit is for Prime open `prime-rl` dataset SFT. Do not submit this config
through `prime train`; Hosted Training uses the env/rollout schema.
{eval_manifest_note}

Upload the archive:

```bash
prime --plain sandbox upload <sandbox_id> {archive_local_path} /tmp/{archive_name}
```

Unpack it in the sandbox:

```bash
prime --plain sandbox run <sandbox_id> -- bash -lc 'mkdir -p /workspace && tar -xzf /tmp/{archive_name} -C /workspace'
```

Run SFT with secrets passed from your local shell. Prime global secrets are not
automatically injected by `prime sandbox run`; this command needs local
`HF_TOKEN` and `WANDB_API_KEY` values because it passes them with `-e`:

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
    eval_manifest_path: Path | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(config_path, output_dir / "config.toml")
    if eval_manifest_path is not None:
        shutil.copyfile(eval_manifest_path, output_dir / SFT_EVAL_MANIFEST_NAME)

    manifest = build_manifest(
        config_path=config_path,
        prime_rl_ref=prime_rl_ref,
        git_commit=git_commit,
        git_dirty=git_dirty,
        eval_manifest_path=eval_manifest_path,
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )

    run_script = output_dir / "run_sft.sh"
    run_script.write_text(_run_sft_script(prime_rl_ref))
    run_script.chmod(0o755)

    archive_path = output_dir.with_suffix(".tar.gz")
    (output_dir / "README.md").write_text(
        _readme(
            prime_rl_ref,
            archive_path,
            includes_eval_manifest=eval_manifest_path is not None,
        )
    )

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
@click.option(
    "--eval-manifest",
    "eval_manifest_path",
    type=click.Path(path_type=Path, exists=True, dir_okay=False),
    default=None,
    help="Optional SFT eval/promotion manifest to include in the launch kit.",
)
def cli(
    config_path: Path,
    output_dir: Path,
    prime_rl_ref: str,
    eval_manifest_path: Path | None,
) -> None:
    """Create a secret-free Prime SFT launch kit for a sandbox/pod."""

    archive_path = write_launch_kit(
        config_path=config_path,
        output_dir=output_dir,
        prime_rl_ref=prime_rl_ref,
        git_commit=current_git_commit(),
        git_dirty=current_git_dirty(),
        eval_manifest_path=eval_manifest_path,
    )
    click.echo(f"launch_kit={output_dir}")
    click.echo(f"archive={archive_path}")
    click.echo(
        f"next=prime sandbox upload <sandbox_id> {archive_path} /tmp/{archive_path.name}"
    )


if __name__ == "__main__":
    cli()
