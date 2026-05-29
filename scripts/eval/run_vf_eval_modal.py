"""Slice 3 Modal eval for Humanize-RL verifier rewards.

Runs generation + the same reward stack used by ``humanize_rl_env`` on Modal.
Use once before full training and once after, then compare summaries.

Examples:
    rtk uvx modal run scripts/run_vf_eval_modal.py \
      --variant baseline \
      --output-path outputs/baseline_eval.json

    rtk uvx modal run scripts/run_vf_eval_modal.py \
      --variant adapter \
      --adapter-dir /checkpoints/gemma4-e2b-humanize-rl-a100-full-v1/final_adapter \
      --output-path outputs/post_train_eval.json
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any, Literal

import modal

app = modal.App("humanize-rl-vf-eval")
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
        "accelerate>=0.34.0",
        "bitsandbytes>=0.46.0",
        "datasets>=4.0.0",
        "fasttext-wheel>=0.9.2",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.0",
        "peft>=0.13.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "scikit-learn>=1.3.0",
        "sentencepiece>=0.2.0",
        "torch>=2.8.0",
        "torchvision",
        "transformers>=5.5.0",
        "verifiers>=0.1.8",
    )
    .env(
        {
            "HF_HOME": "/model_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
        }
    )
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_dir(
        "environments/humanize_rl_env",
        remote_path="/workspace/environments/humanize_rl_env",
    )
    .add_local_file(
        "data/rl/humanize_tasks_v01_smoke.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_v01_smoke.jsonl",
    )
    .add_local_file(
        "data/rl/humanize_tasks_rl_mix_v1_filtered_softened_midband.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_rl_mix_v1_filtered_softened_midband.jsonl",
    )
    .add_local_file(
        "data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
    )
    .add_local_file(
        "data/rl/humanize_tasks_v02.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_v02.jsonl",
    )
    .add_local_file(
        "data/rl/humanize_tasks_v03_filtered.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_v03_filtered.jsonl",
    )
    .add_local_file(
        "data/rl/eval_e5_hard_penalty_mix_v2.jsonl",
        remote_path="/workspace/data/rl/eval_e5_hard_penalty_mix_v2.jsonl",
    )
    .add_local_file(
        "data/rl/eval_e6_longform_mix_v2.jsonl",
        remote_path="/workspace/data/rl/eval_e6_longform_mix_v2.jsonl",
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

Variant = Literal["baseline", "adapter", "hf_adapter"]


def _mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={"/model_cache": model_cache_volume, "/checkpoints": checkpoint_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=0),
    single_use_containers=True,
)
def run_eval_remote(
    variant: Variant,
    model_name: str,
    adapter_dir: str | None,
    adapter_repo: str | None,
    task_path: str,
    split: str,
    max_examples: int,
    max_new_tokens: int,
) -> dict[str, Any]:
    import os
    import sys

    sys.path.insert(0, "/workspace/src")
    sys.path.insert(0, "/workspace/environments/humanize_rl_env")
    os.chdir("/workspace")

    import torch
    import verifiers as vf
    from humanize_rl_env import load_environment
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

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        token=token,
    ).to("cuda")

    original_create_and_replace = LoraModel._create_and_replace

    def patched_create_and_replace(
        self: Any,
        peft_config: Any,
        adapter_name: str,
        target: Any,
        target_name: str,
        parent: Any,
        current_key: Any = None,
        **kwargs: Any,
    ) -> Any:
        if isinstance(target, Gemma4ClippableLinear):
            return original_create_and_replace(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kwargs,
            )
        return original_create_and_replace(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kwargs,
        )

    LoraModel._create_and_replace = patched_create_and_replace
    try:
        if variant == "adapter":
            if not adapter_dir:
                raise ValueError("adapter variant requires --adapter-dir")
            model = PeftModel.from_pretrained(model, adapter_dir, token=token)
        elif variant == "hf_adapter":
            if not adapter_repo:
                raise ValueError("hf_adapter variant requires --adapter-repo")
            model = PeftModel.from_pretrained(model, adapter_repo, token=token)
    finally:
        LoraModel._create_and_replace = original_create_and_replace
    model.eval()

    env = load_environment(task_path=task_path, split=split)
    if type(env).__name__ != "SingleTurnEnv":
        raise RuntimeError("humanize_rl_env did not return vf.SingleTurnEnv")
    if vf.__name__ != "verifiers":
        raise RuntimeError("unexpected verifiers module import")

    tasks = load_tasks(Path(task_path))
    if split != "all":
        tasks = [task for task in tasks if task.split == split]
    if max_examples > 0:
        tasks = tasks[:max_examples]

    scorer = load_ridge_scorer()
    rows: list[dict[str, Any]] = []
    for task in tasks:
        messages = [{"role": "user", "content": task.instruction}]
        if task.input_text:
            messages[0]["content"] = f"{task.instruction}\n\nSource:\n{task.input_text}"
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
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
        response = processor.decode(new_tokens, skip_special_tokens=True).strip()
        result = score_response(task, response, scorer)
        rows.append(
            {
                "task_id": task.id,
                "family": task.family,
                "reward_profile": task.reward_profile,
                "response": response,
                "reward": result.reward,
                "raw_reward": result.raw_reward,
                "components": result.components,
                "weighted_components": result.weighted_components,
                "penalties": result.penalties,
            }
        )

    rewards = [float(row["reward"]) for row in rows]
    risks = [sum(float(v) for v in row["penalties"].values()) for row in rows]
    summary = {
        "variant": variant,
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "adapter_repo": adapter_repo,
        "task_path": task_path,
        "split": split,
        "rows": rows,
        "mean_reward": _mean(rewards),
        "mean_risk_penalty": _mean(risks),
        "risk_penalty_negative_count": sum(1 for value in risks if value < 0),
        "ridge_scorer_loaded": scorer is not None,
        "verifiers_env": "humanize_rl_env",
    }
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))
    return summary


@app.local_entrypoint()
def main(
    variant: Variant = "baseline",
    model_name: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
    adapter_dir: str | None = None,
    adapter_repo: str | None = None,
    task_path: str = "/workspace/data/rl/humanize_tasks_v01_smoke.jsonl",
    split: str = "validation",
    max_examples: int = 10,
    max_new_tokens: int = 180,
    output_path: str = "outputs/baseline_eval.json",
) -> None:
    result = run_eval_remote.remote(
        variant,
        model_name,
        adapter_dir,
        adapter_repo,
        task_path,
        split,
        max_examples,
        max_new_tokens,
    )
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote eval summary to {path}")
