from __future__ import annotations

import modal

app = modal.App("humanize-rl-gemma4-e2b-merge")

model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
merge_volume = modal.Volume.from_name("humanize-rl-merge-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "bitsandbytes>=0.46.0",
        "datasets>=3.6.0",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.0",
        "peft>=0.16.0",
        "protobuf>=5.0.0",
        "sentencepiece>=0.2.0",
        "transformers>=4.54.0",
        "trl>=0.19.1",
        "unsloth[cu128-torch270]>=2025.7.8",
        "unsloth_zoo>=2025.7.10",
    )
    .env({"HF_HOME": "/model_cache", "HF_XET_HIGH_PERFORMANCE": "1"})
)

with image.imports():
    import unsloth  # noqa: F401,I001
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": model_cache_volume, "/merge": merge_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 60 * 60,
    retries=0,
    single_use_containers=True,
)
def merge_and_push(
    base_model: str = "unsloth/gemma-4-E2B-it",
    lora_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-lora",
    merged_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
) -> dict:
    import os
    from pathlib import Path

    token = os.environ.get("HF_TOKEN")
    out_dir = Path("/merge/gemma4-e2b-humanize-merged")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading base model: {base_model}")
    model = AutoModelForImageTextToText.from_pretrained(
        base_model,
        torch_dtype="auto",
        device_map="auto",
        token=token,
    )
    processor = AutoProcessor.from_pretrained(base_model, token=token)

    print(f"Loading LoRA adapter: {lora_repo}")
    model = PeftModel.from_pretrained(model, lora_repo, token=token)
    print("Merging adapter into base model")
    model = model.merge_and_unload()

    print(f"Saving merged model to {out_dir}")
    model.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(out_dir)
    (out_dir / "README.md").write_text(
        "---\n"
        "license: apache-2.0\n"
        f"base_model: {base_model}\n"
        "library_name: transformers\n"
        "pipeline_tag: image-text-to-text\n"
        "tags:\n"
        "  - gemma-4\n"
        "  - unsloth\n"
        "  - humanize-rl\n"
        "  - merged\n"
        "---\n\n"
        "# Gemma 4 E2B Humanize-RL merged model\n\n"
        f"Merged model from `{base_model}` and LoRA adapter `{lora_repo}`.\n\n"
        "This artifact was produced by a Modal merge-only job using Transformers + PEFT merge_and_unload.\n"
    )

    print(f"Pushing merged model folder to {merged_repo}")
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id=merged_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=merged_repo,
        repo_type="model",
        folder_path=str(out_dir),
        path_in_repo=".",
        commit_message="Upload merged Gemma 4 E2B Humanize-RL model",
    )
    merge_volume.commit()
    return {"merged_repo": merged_repo, "out_dir": str(out_dir)}


@app.local_entrypoint()
def main():
    call = merge_and_push.spawn()
    print(f"Spawned merge call: {call.object_id}")
