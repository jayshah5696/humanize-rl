"""Sampled side-by-side comparison for D3 reward-hacking inspection.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 6 D3.

Loads the base SFT model with and without a LoRA adapter and generates
K samples per task at temperature 1.0 (matches training). Greedy eval
masks small LoRA shifts; sampled eval reveals reward-hacking patterns
(terseness collapse, evasion, repetition, fact loss).

Output: ``outputs/ablations/sampled_inspection_<adapter>.json`` with
per-task baseline + adapter sample sets, plus aggregate diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal

app = modal.App("humanize-rl-sample-inspect")
model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "humanize-rl-checkpoints", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.8.0",
        "torchvision",
        "datasets>=4.0.0",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.0",
        "peft>=0.13.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "sentencepiece>=0.2.0",
        "tokenizers>=0.22.0,<=0.23.0",
        "transformers>=5.5.0",
        "scikit-learn>=1.3.0",
        "fasttext-wheel>=0.9.2",
    )
    .env(
        {
            "HF_HOME": "/model_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_file(
        "data/rl/humanize_tasks_v01_smoke.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_v01_smoke.jsonl",
    )
    .add_local_file(
        "data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
    )
    .add_local_file(
        "models/track_a_10k/ridge.pkl",
        remote_path="/workspace/models/track_a_10k/ridge.pkl",
    )
    .add_local_file(
        "models/distilled/baseline_ridge.pkl",
        remote_path="/workspace/models/distilled/baseline_ridge.pkl",
    )
)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/model_cache": model_cache_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=30 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
)
def sample_remote(
    model_name: str,
    adapter_dir: str,
    task_path: str,
    split: str,
    max_examples: int,
    n_samples: int,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> dict[str, Any]:
    import os
    import sys

    sys.path.insert(0, "/workspace/src")
    os.chdir("/workspace")

    import torch
    from peft import PeftModel
    from peft.tuners.lora.model import LoraModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ClippableLinear

    from humanize_rl.reward.reward import load_ridge_scorer, score_response
    from humanize_rl.reward.tasks import load_tasks

    token = os.environ.get("HF_TOKEN")
    processor = AutoProcessor.from_pretrained(model_name, token=token)
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token", None
    ):
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForImageTextToText.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, token=token
    ).to("cuda")

    # PEFT patch (Bug F).
    original = LoraModel._create_and_replace

    def patched(self, peft_config, adapter_name, target, target_name, parent, current_key=None, **kw):
        if isinstance(target, Gemma4ClippableLinear):
            return original(self, peft_config, adapter_name, target.linear, "linear", target, current_key=current_key, **kw)
        return original(self, peft_config, adapter_name, target, target_name, parent, current_key=current_key, **kw)

    LoraModel._create_and_replace = patched
    try:
        adapter_model = PeftModel.from_pretrained(base_model, adapter_dir, token=token, adapter_name="default")
    finally:
        LoraModel._create_and_replace = original
    adapter_model.eval()

    tasks = [t for t in load_tasks(Path(task_path)) if t.split == split][:max_examples]
    scorer = load_ridge_scorer()

    def sample_set(model_to_use, task) -> list[dict[str, Any]]:
        messages = [{"role": "user", "content": task.instruction}]
        if task.input_text:
            messages[0]["content"] = f"{task.instruction}\n\nSource:\n{task.input_text}"
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        ).to(model_to_use.device)
        outputs = []
        for s in range(n_samples):
            with torch.inference_mode():
                output_ids = model_to_use.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    use_cache=True,
                )
            new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
            response = processor.decode(new_tokens, skip_special_tokens=True).strip()
            result = score_response(task, response, scorer)
            outputs.append({
                "sample_idx": s,
                "response": response,
                "response_length": len(response),
                "reward": result.reward,
                "weighted_components": result.weighted_components,
                "penalties": result.penalties,
            })
        return outputs

    rows = []
    for task in tasks:
        print(f"[sample] {task.id}", flush=True)
        # Disable adapter for baseline samples; enable for adapter samples.
        with adapter_model.disable_adapter():
            baseline = sample_set(adapter_model, task)
        adp = sample_set(adapter_model, task)
        rows.append({
            "task_id": task.id,
            "family": task.family,
            "reward_profile": task.reward_profile,
            "instruction": task.instruction,
            "input_text": task.input_text,
            "required_facts": task.required_facts,
            "baseline": baseline,
            "adapter": adp,
        })

    return {
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "n_tasks": len(tasks),
        "n_samples_per_task": n_samples,
        "temperature": temperature,
        "top_p": top_p,
        "rows": rows,
    }


@app.local_entrypoint()
def main(
    model_name: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
    adapter_dir: str = "/checkpoints/gemma4-rl-ablation-a2-scalar-softened/final_adapter",
    task_path: str = "/workspace/data/rl/humanize_tasks_v01_smoke.jsonl",
    split: str = "validation",
    max_examples: int = 8,
    n_samples: int = 4,
    temperature: float = 1.0,
    top_p: float = 1.0,
    max_new_tokens: int = 200,
    output_path: str = "outputs/ablations/sampled_inspection_a2.json",
) -> None:
    result = sample_remote.remote(
        model_name, adapter_dir, task_path, split,
        max_examples, n_samples, temperature, top_p, max_new_tokens,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2))
    print(f"Wrote {path}")
