from __future__ import annotations

import modal

app = modal.App("humanize-rl-gemma4-merge-parity")
model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)

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
    volumes={"/model_cache": model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=3600,
)
def verify() -> dict:
    import os

    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    base = "unsloth/gemma-4-E2B-it"
    lora = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"

    processor = AutoProcessor.from_pretrained(base, token=token)

    def gen(model):
        inputs = processor.apply_chat_template(
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
        return processor.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    print("Loading LoRA direct")
    direct = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )
    direct = PeftModel.from_pretrained(direct, lora, token=token)
    direct.eval()
    direct_text = gen(direct)
    print("direct:", direct_text)

    print("Merging in-memory")
    merged = direct.merge_and_unload()
    merged.eval()
    merged_text = gen(merged)
    print("merged_in_memory:", merged_text)

    return {"direct": direct_text, "merged_in_memory": merged_text}


@app.local_entrypoint()
def main():
    print(verify.remote())
