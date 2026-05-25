from __future__ import annotations

import modal

app = modal.App("humanize-rl-gemma4-merge-verify-push")
model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
merge_volume = modal.Volume.from_name("humanize-rl-merge-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "huggingface_hub>=0.34.0",
        "peft>=0.16.0",
        "pillow>=12.0.0",
        "protobuf>=5.0.0",
        "sentencepiece>=0.2.0",
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "transformers>=4.54.0",
    )
    .env({"HF_HOME": "/model_cache", "HF_XET_HIGH_PERFORMANCE": "1"})
)

PROMPT = "Rewrite this message so it sounds natural, warm, and professional. Keep it concise and preserve all facts.\n\nHi Saurabh. Yes looking forward to working with you. Let me know if you want any specific things to read or refer. Ernest mentioned about LangGraph which I have already used in few projects. Love to hear from you."


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": model_cache_volume, "/merge": merge_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 60 * 60,
)
def merge_verify_push() -> dict:
    import json
    import os
    import shutil
    from pathlib import Path

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    base = "unsloth/gemma-4-E2B-it"
    lora = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"
    repo = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
    out_dir = Path("/merge/gemma4-e2b-humanize-verified-merged")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    processor = AutoProcessor.from_pretrained(base, token=token)

    def generate(model, processor_obj) -> str:
        inputs = processor_obj.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=160, do_sample=False, use_cache=True
            )
        return processor_obj.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    print("Loading base + LoRA")
    direct = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )
    direct = PeftModel.from_pretrained(direct, lora, token=token)
    direct.eval()
    direct_text = generate(direct, processor)
    print("direct:", direct_text)

    print("Merging in memory")
    merged = direct.merge_and_unload()
    merged.eval()
    merged_text = generate(merged, processor)
    print("merged_in_memory:", merged_text)
    if direct_text != merged_text:
        raise RuntimeError("In-memory merge parity failed")

    print(f"Saving merged model to {out_dir}")
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(out_dir)

    del direct
    del merged
    torch.cuda.empty_cache()

    print("Reloading saved merged model for parity check")
    reloaded_processor = AutoProcessor.from_pretrained(out_dir)
    reloaded = AutoModelForImageTextToText.from_pretrained(
        out_dir, torch_dtype=torch.bfloat16, device_map="auto"
    )
    reloaded.eval()
    reloaded_text = generate(reloaded, reloaded_processor)
    print("reloaded:", reloaded_text)
    if reloaded_text != direct_text:
        raise RuntimeError(
            f"Saved reload parity failed. direct={direct_text!r} reloaded={reloaded_text!r}"
        )

    readme = out_dir / "README.md"
    readme.write_text(
        "---\n"
        "license: apache-2.0\n"
        f"base_model: {base}\n"
        "library_name: transformers\n"
        "pipeline_tag: image-text-to-text\n"
        "tags:\n"
        "  - gemma-4\n"
        "  - unsloth\n"
        "  - humanize-rl\n"
        "  - merged\n"
        "---\n\n"
        "# Gemma 4 E2B Humanize-RL merged model\n\n"
        f"Verified merged model from `{base}` and LoRA adapter `{lora}`.\n\n"
        "Merge parity was checked by comparing direct LoRA generation, in-memory merged generation, and saved/reloaded merged generation on a held-out rewrite prompt.\n"
    )

    index_path = out_dir / "model.safetensors.index.json"
    if not index_path.exists():
        weight_map = {}
        total_size = 0
        for shard in sorted(out_dir.glob("model-*.safetensors")):
            from safetensors import safe_open

            with safe_open(shard, framework="pt") as handle:
                for key in handle.keys():
                    weight_map[key] = shard.name
                    total_size += (
                        handle.get_tensor(key).numel()
                        * handle.get_tensor(key).element_size()
                    )
        index_path.write_text(
            json.dumps(
                {"metadata": {"total_size": total_size}, "weight_map": weight_map},
                indent=2,
            )
        )

    print(f"Uploading verified merged folder to {repo}")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=repo,
        repo_type="model",
        folder_path=str(out_dir),
        path_in_repo=".",
        commit_message="Upload verified merged Gemma 4 E2B Humanize-RL model",
    )
    merge_volume.commit()
    return {"repo": repo, "direct": direct_text, "reloaded": reloaded_text}


@app.local_entrypoint()
def main():
    print(merge_verify_push.remote())
