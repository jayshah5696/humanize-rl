from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("humanize-rl-gemma4-eval")
model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "huggingface_hub>=0.34.0",
        "pillow>=12.0.0",
        "protobuf>=5.0.0",
        "sentencepiece>=0.2.0",
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "transformers>=4.54.0",
    )
    .env({"HF_HOME": "/model_cache", "HF_XET_HIGH_PERFORMANCE": "1"})
)

PROMPTS = [
    {
        "id": "eval_001_saurabh_intro",
        "category": "rewrite_professional",
        "instruction": "Rewrite this message so it sounds natural, warm, and professional. Keep it concise and preserve all facts.\n\nHi Saurabh. Yes looking forward to working with you. Let me know if you want any specific things to read or refer. Ernest mentioned about LangGraph which I have already used in few projects. Love to hear from you.",
    },
    {
        "id": "eval_002_slack_staging",
        "category": "slack_rewrite",
        "instruction": "Clean up this Slack update. Make it casual, direct, and not overly polished.\n\nPlease be advised that the staging environment has been restored to full operational status. The root cause was identified as a missing environment variable, and we will continue monitoring performance parameters throughout the afternoon.",
    },
    {
        "id": "eval_003_client_hotfix",
        "category": "email_direct",
        "instruction": "Write a concise email to a client telling them the hotfix for their integration is live. Mention that we're monitoring it for the next hour. Use placeholders for any missing names.",
    },
    {
        "id": "eval_004_manager_pto",
        "category": "email_direct_placeholders",
        "instruction": "Draft a short email to my manager asking for PTO next Friday. Mention that I'll send a handoff note before I leave. Do not invent manager names, teammate names, project names, or dates beyond next Friday.",
    },
    {
        "id": "eval_005_launch_note",
        "category": "product_copy",
        "instruction": "Rewrite this release note so it sounds like a human product manager wrote it, not a marketing bot.\n\nWe are thrilled to announce the launch of our enhanced dashboard experience, designed to empower teams with actionable insights and streamline cross-functional alignment across mission-critical workflows.",
    },
    {
        "id": "eval_006_incident_summary",
        "category": "incident_summary",
        "instruction": "Turn this rough incident note into a clear two-sentence update for leadership.\n\napi latency jumped after deploy. rolled back at 2:15. queue drained by 2:40. no data loss. still checking logs to confirm exact trigger.",
    },
    {
        "id": "eval_007_candidate_rejection",
        "category": "email_direct",
        "instruction": "Write a warm but concise candidate rejection email after a final interview. Do not over-apologize, do not use corporate filler, and do not invent role, company, or candidate names.",
    },
    {
        "id": "eval_008_technical_explain",
        "category": "technical_explanation",
        "instruction": "Explain why database indexes can speed up reads but slow down writes. Keep it natural and concrete, like you're explaining it to a junior engineer in Slack.",
    },
    {
        "id": "eval_009_founder_update",
        "category": "founder_update",
        "instruction": "Rewrite this investor update to sound more candid and less inflated.\n\nWe have achieved significant momentum across multiple strategic pillars and are excited to continue executing on our vision. While challenges remain, we believe our differentiated approach positions us for long-term success.",
    },
    {
        "id": "eval_010_peer_ping",
        "category": "slack_direct",
        "instruction": "Write a quick message to a peer engineer asking if they can review my pull request before standup. Keep it under 25 words and don't sound needy.",
    },
]


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": model_cache_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
)
def generate_eval() -> list[dict]:
    import os

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    token = os.environ.get("HF_TOKEN")
    variants = {
        "modal_base": "unsloth/gemma-4-E2B-it",
        "modal_merged": "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
    }
    rows = []
    for variant, model_name in variants.items():
        print(f"Loading {variant}: {model_name}", flush=True)
        processor = AutoProcessor.from_pretrained(model_name, token=token)
        model = AutoModelForImageTextToText.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        model.eval()
        for prompt in PROMPTS:
            messages = [{"role": "user", "content": prompt["instruction"]}]
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=180,
                    do_sample=False,
                    use_cache=True,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
            response = processor.decode(new_tokens, skip_special_tokens=True).strip()
            rows.append({**prompt, "model_variant": variant, "response": response})
            print(f"{variant} {prompt['id']}: {response[:120]}", flush=True)
        del model
        torch.cuda.empty_cache()
    return rows


@app.local_entrypoint()
def main(
    output_path: str = "outputs_gemma4_humanize_modal_full_eval/generations.jsonl",
):
    rows = generate_eval.remote()
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    print(f"Wrote {len(rows)} generations to {path}")
