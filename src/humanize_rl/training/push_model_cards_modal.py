"""Push refreshed README model cards to the merged and LoRA HF repos.

This is a one-shot artifact-hygiene job. It does **not** re-merge or
re-upload any weights. It uploads `README.md` only.

Detached usage:

```bash
uvx modal run --detach \
  src/humanize_rl/training/push_model_cards_modal.py \
  --merged-repo jayshah5696/gemma4-e2b-humanize-unsloth-merged \
  --lora-repo jayshah5696/gemma4-e2b-humanize-unsloth-lora
```
"""

from __future__ import annotations

import modal

app = modal.App("humanize-rl-gemma4-push-model-cards")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("huggingface_hub>=0.34.0")
    .add_local_file(
        "src/humanize_rl/training/_model_cards.py",
        remote_path="/workspace/_model_cards.py",
    )
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=10 * 60,
    single_use_containers=True,
)
def push_cards(
    merged_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
    lora_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-lora",
    base_model: str = "unsloth/gemma-4-E2B-it",
    lora_rank: int = 8,
    lora_alpha: int = 8,
) -> dict:
    import io
    import os
    import sys

    from huggingface_hub import HfApi

    sys.path.insert(0, "/workspace")
    from _model_cards import (  # type: ignore[import-not-found]
        lora_model_card,
        merged_model_card,
    )

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    merged_card = merged_model_card(
        base_model=base_model,
        lora_repo=lora_repo,
    )
    lora_card = lora_model_card(
        base_model=base_model,
        merged_repo=merged_repo,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )

    api.upload_file(
        path_or_fileobj=io.BytesIO(merged_card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=merged_repo,
        commit_message="Refresh model card with verification report",
    )
    api.upload_file(
        path_or_fileobj=io.BytesIO(lora_card.encode("utf-8")),
        path_in_repo="README.md",
        repo_id=lora_repo,
        commit_message="Refresh model card with verification metadata",
    )
    return {
        "merged_repo": merged_repo,
        "lora_repo": lora_repo,
        "merged_card_bytes": len(merged_card),
        "lora_card_bytes": len(lora_card),
    }


@app.local_entrypoint()
def main(
    merged_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
    lora_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-lora",
    base_model: str = "unsloth/gemma-4-E2B-it",
    lora_rank: int = 8,
    lora_alpha: int = 8,
) -> None:
    call = push_cards.spawn(
        merged_repo=merged_repo,
        lora_repo=lora_repo,
        base_model=base_model,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
    )
    print(f"Spawned Modal call: {call.object_id}")
