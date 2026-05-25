from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any

import modal

app = modal.App("humanize-rl-gemma4-e2b-grpo")

GPU_TYPE = "A100-40GB"
TIMEOUT_HOURS = 4
MAX_RETRIES = 0

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
        "triton>=3.4.0",
        "torchvision",
        "bitsandbytes>=0.46.0",
        "datasets>=4.0.0",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.0",
        "pydantic>=2.0.0",
        "pyyaml>=6.0.0",
        "sentencepiece>=0.2.0",
        "timm",
        "tokenizers>=0.22.0,<=0.23.0",
        "transformers>=5.5.0",
        "trl>0.19.0,<=0.24.0",
        "wandb>=0.21.0",
        "unsloth_zoo[base] @ git+https://github.com/unslothai/unsloth-zoo",
        "unsloth[base] @ git+https://github.com/unslothai/unsloth",
        "git+https://github.com/triton-lang/triton.git@0add68262ab0a2e33b84524346cb27cbb2787356#subdirectory=python/triton_kernels",
    )
    .env(
        {
            "HF_HOME": "/model_cache",
            "HF_HUB_ENABLE_HF_TRANSFER": "1",
            "HF_XET_HIGH_PERFORMANCE": "1",
            "UNSLOTH_COMPILE_DISABLE": "1",
        }
    )
    .add_local_dir("src", remote_path="/workspace/src")
    .add_local_file(
        "data/rl/humanize_tasks_v01_smoke.jsonl",
        remote_path="/workspace/data/rl/humanize_tasks_v01_smoke.jsonl",
    )
    .add_local_file(
        "configs/rl/gemma4_e2b_rl_smoke.yaml",
        remote_path="/workspace/configs/rl/gemma4_e2b_rl_smoke.yaml",
    )
    .add_local_file(
        "configs/rl/gemma4_e2b_rl_smoke_v2.yaml",
        remote_path="/workspace/configs/rl/gemma4_e2b_rl_smoke_v2.yaml",
    )
)

with image.imports():
    import unsloth  # noqa: F401,I001
    import torch
    import yaml
    from unsloth import FastVisionModel


@dataclass
class GRPOSmokeConfig:
    model_name: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
    experiment_name: str | None = None
    hf_lora_repo: str = "jayshah5696/gemma4-e2b-humanize-rl-smoke-lora"
    task_path: str = "/workspace/data/rl/humanize_tasks_v01_smoke.jsonl"
    train_split: str = "train"
    eval_split: str = "validation"
    max_seq_length: int = 1024
    max_prompt_length: int = 768
    max_completion_length: int = 128
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-5
    max_steps: int = 10
    num_generations: int = 2
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    warmup_ratio: float = 0.1
    weight_decay: float = 0.001
    optim: str = "adamw_8bit"
    loss_type: str = "bnpo"
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    delta: float = 1.5
    mask_truncated_completions: bool = True
    temperature: float = 1.0
    seed: int = 3407
    sample_before_after: int = 12
    generation_batch_size: int = 2
    artifact_generation_samples: int = 2
    push_to_hub: bool = False
    report_to: str = "none"

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_name = f"gemma4-e2b-humanize-grpo-{timestamp}"


def _load_config(path: str) -> GRPOSmokeConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    if "task_path" in data and not str(data["task_path"]).startswith("/"):
        data["task_path"] = f"/workspace/{data['task_path']}"
    return GRPOSmokeConfig(**data)


def _processor_prompt(prompt: list[dict[str, str]]) -> list[dict[str, Any]]:
    processed: list[dict[str, Any]] = []
    for message in prompt:
        content = message["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        processed.append({"role": message["role"], "content": content})
    return processed


def _prepare_tokenizer_for_batched_generation(tokenizer: Any) -> None:
    if getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"


def _generation_texts(
    tokenizer: Any,
    model: Any,
    prompts: list[list[dict[str, str]]],
    max_tokens: int,
) -> list[str]:
    rendered_prompts = [
        tokenizer.apply_chat_template(
            _processor_prompt(prompt),
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    _prepare_tokenizer_for_batched_generation(tokenizer)
    inputs = tokenizer(
        text=rendered_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            use_cache=True,
        )
    prompt_width = inputs["input_ids"].shape[-1]
    responses = []
    for row_index in range(len(rendered_prompts)):
        new_tokens = output_ids[row_index][prompt_width:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
    return responses


def _batched(
    items: list[dict[str, Any]], batch_size: int
) -> list[list[dict[str, Any]]]:
    safe_batch_size = max(1, batch_size)
    return [
        items[index : index + safe_batch_size]
        for index in range(0, len(items), safe_batch_size)
    ]


def _sample_generations(
    tokenizer: Any,
    model: Any,
    rows: list[dict[str, Any]],
    max_tokens: int,
    limit: int,
    label: str,
    batch_size: int = 1,
) -> list[dict[str, Any]]:
    from humanize_rl.reward.reward import score_response
    from humanize_rl.reward.tasks import RLTask

    model.eval()
    samples: list[dict[str, Any]] = []
    selected_rows = rows[:limit]
    for batch in _batched(selected_rows, batch_size):
        responses = _generation_texts(
            tokenizer,
            model,
            [row["prompt"] for row in batch],
            max_tokens,
        )
        for row, response in zip(batch, responses, strict=True):
            task = RLTask.model_validate(row["task"])
            reward = score_response(task, response)
            samples.append(
                {
                    "phase": label,
                    "task_id": row["task_id"],
                    "family": row["family"],
                    "reward": reward.reward,
                    "penalties": reward.penalties,
                    "components": reward.components,
                    "prompt": row["prompt"],
                    "response": response,
                }
            )
            print(
                f"{label} {row['task_id']} reward={reward.reward:.3f}: {response[:120]}",
                flush=True,
            )
    model.train()
    return samples


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def _mean_reward(rows: list[dict[str, Any]]) -> float:
    if not rows:
        return 0.0
    return sum(float(row["reward"]) for row in rows) / len(rows)


def _gpu_snapshot(torch_module: Any) -> dict[str, Any]:
    if not torch_module.cuda.is_available():
        return {"available": False}
    device = torch_module.cuda.current_device()
    free_bytes, total_bytes = torch_module.cuda.mem_get_info(device)
    return {
        "available": True,
        "name": torch_module.cuda.get_device_name(device),
        "device": device,
        "bf16_supported": torch_module.cuda.is_bf16_supported(),
        "free_gb": round(free_bytes / 1024**3, 2),
        "total_gb": round(total_bytes / 1024**3, 2),
        "reserved_gb": round(torch_module.cuda.memory_reserved(device) / 1024**3, 2),
        "allocated_gb": round(torch_module.cuda.memory_allocated(device) / 1024**3, 2),
    }


def _load_policy(config: GRPOSmokeConfig) -> tuple[Any, Any, float]:
    """Load Gemma 4 E2B policy.

    The Unsloth ``FastVisionModel.from_pretrained`` load-report prints
    layers ``[num_hidden_layers - num_kv_shared_layers, num_hidden_layers)``
    of ``self_attn.{k_proj,v_proj,k_norm,v_norm}`` as MISSING. This is a
    false alarm: Gemma 4 shares KV across those layers by design, so the
    Transformers ``Gemma4TextAttention.__init__`` does not instantiate those
    modules (see PR huggingface/transformers#45328) and the forward pass
    never reads them. We additionally cross-check with a Transformers load
    that reports the canonical missing/unexpected key sets so that any real
    missing weight surfaces as a hard error.
    """
    import os

    started = perf_counter()
    model, tokenizer = FastVisionModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        load_in_4bit=False,
        fast_inference=False,
        token=os.environ.get("HF_TOKEN"),
    )
    _assert_only_shared_kv_keys_missing(model)
    return model, tokenizer, perf_counter() - started


def _assert_only_shared_kv_keys_missing(model: Any) -> None:
    """Verify that any zero-initialised attention weights belong to KV-shared layers.

    Raises ``RuntimeError`` if a non-shared layer has zero ``k_proj`` /
    ``v_proj`` weight, which would indicate a real broken checkpoint.
    """
    import torch

    text_cfg = getattr(model.config, "text_config", model.config)
    n_layers = int(getattr(text_cfg, "num_hidden_layers", 0))
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0))
    if n_layers == 0 or n_shared == 0:
        return
    shared_start = n_layers - n_shared
    language_model = getattr(model, "language_model", None)
    if language_model is None:
        language_model = getattr(getattr(model, "model", model), "language_model", None)
    if language_model is None:
        return
    layers = getattr(language_model, "layers", None)
    if layers is None:
        return
    real_zero_layers: list[int] = []
    for idx, layer in enumerate(layers):
        if idx >= shared_start:
            continue
        attn = getattr(layer, "self_attn", None)
        k = getattr(attn, "k_proj", None) if attn is not None else None
        v = getattr(attn, "v_proj", None) if attn is not None else None
        if k is not None and torch.all(k.weight == 0).item():
            real_zero_layers.append(idx)
        if v is not None and torch.all(v.weight == 0).item():
            real_zero_layers.append(idx)
    if real_zero_layers:
        raise RuntimeError(
            "Non-shared decoder layers have zero k/v weights: "
            f"{sorted(set(real_zero_layers))}. Checkpoint is broken."
        )
    print(
        f"[gemma4] {n_shared} KV-shared layers (idx {shared_start}..{n_layers - 1}) "
        "intentionally omit k_proj/v_proj/k_norm/v_norm; MISSING report above is benign.",
        flush=True,
    )


def _drop_unused_mm_token_type_ids_for_generate(model: Any) -> None:
    original_generate = model.generate

    @wraps(original_generate)
    def generate_without_mm_token_type_ids(self: Any, *args: Any, **kwargs: Any) -> Any:
        kwargs.pop("mm_token_type_ids", None)
        return original_generate(*args, **kwargs)

    model.generate = MethodType(generate_without_mm_token_type_ids, model)


def _attach_lora(config: GRPOSmokeConfig, model: Any) -> Any:
    model = FastVisionModel.get_peft_model(
        model,
        r=config.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=config.lora_alpha,
        use_gradient_checkpointing="unsloth",
        random_state=config.seed,
    )
    _drop_unused_mm_token_type_ids_for_generate(model)
    return model


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface"), modal.Secret.from_name("wandb")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    single_use_containers=True,
)
def train_grpo(config_path: str) -> dict[str, Any]:
    import os
    import sys

    sys.path.insert(0, "/workspace/src")
    os.environ.setdefault("WANDB_DISABLED", "true")

    from trl import GRPOConfig, GRPOTrainer

    from humanize_rl.reward.grpo_dataset import load_grpo_dataset, load_grpo_rows
    from humanize_rl.reward.grpo_rewards import WEIGHTED_REWARD_FUNCS

    config = _load_config(config_path)
    output_dir = Path("/checkpoints") / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config.model_name}", flush=True)
    run_started = perf_counter()
    model, tokenizer, model_load_seconds = _load_policy(config)
    print(f"Model loaded in {model_load_seconds:.1f}s", flush=True)
    print(f"GPU after load: {_gpu_snapshot(torch)}", flush=True)

    model = _attach_lora(config, model)
    print(f"GPU after LoRA attach: {_gpu_snapshot(torch)}", flush=True)

    task_path = Path(config.task_path)
    train_dataset = load_grpo_dataset(task_path, split=config.train_split)
    eval_rows = load_grpo_rows(task_path, split=config.eval_split)
    train_rows = load_grpo_rows(task_path, split=config.train_split)

    before_rows = _sample_generations(
        tokenizer,
        model,
        eval_rows,
        config.max_completion_length,
        config.sample_before_after,
        "before",
        config.generation_batch_size,
    )
    _write_jsonl(output_dir / "before_generations.jsonl", before_rows)

    training_args = GRPOConfig(
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=max(1, int(config.max_steps * config.warmup_ratio)),
        lr_scheduler_type="linear",
        optim=config.optim,
        logging_steps=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        save_steps=max(config.max_steps, 1),
        report_to=config.report_to,
        output_dir=str(output_dir / "trainer"),
        epsilon=config.epsilon,
        epsilon_high=config.epsilon_high,
        delta=config.delta,
        loss_type=config.loss_type,
        mask_truncated_completions=config.mask_truncated_completions,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=config.seed,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=WEIGHTED_REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
    )

    trainer_stats = trainer.train()

    adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    after_rows = _sample_generations(
        tokenizer,
        model,
        eval_rows,
        config.max_completion_length,
        config.sample_before_after,
        "after",
        config.generation_batch_size,
    )
    _write_jsonl(output_dir / "after_generations.jsonl", after_rows)

    if config.push_to_hub:
        model.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))

    summary = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "hf_lora_repo": config.hf_lora_repo if config.push_to_hub else None,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "before_mean_reward": _mean_reward(before_rows),
        "after_mean_reward": _mean_reward(after_rows),
        "trainer_stats": str(trainer_stats),
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds": perf_counter() - run_started,
        "gpu": _gpu_snapshot(torch),
        "config": config.__dict__,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    checkpoint_volume.commit()
    return summary


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    single_use_containers=True,
)
def verify_model_artifact(config_path: str) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/workspace/src")

    from humanize_rl.reward.grpo_dataset import load_grpo_rows

    config = _load_config(config_path)
    started = perf_counter()
    print(f"Verifying model artifact: {config.model_name}", flush=True)
    model, tokenizer, model_load_seconds = _load_policy(config)
    eval_rows = load_grpo_rows(Path(config.task_path), split=config.eval_split)
    generation_started = perf_counter()
    samples = _sample_generations(
        tokenizer,
        model,
        eval_rows,
        config.max_completion_length,
        config.artifact_generation_samples,
        "artifact",
        config.generation_batch_size,
    )
    summary = {
        "model_name": config.model_name,
        "loaded": True,
        "model_load_seconds": model_load_seconds,
        "generation_seconds": perf_counter() - generation_started,
        "elapsed_seconds": perf_counter() - started,
        "sample_count": len(samples),
        "mean_reward": _mean_reward(samples),
        "gpu": _gpu_snapshot(torch),
        "samples": samples,
    }
    output_dir = Path("/checkpoints") / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "artifact_verification.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    _write_jsonl(output_dir / "artifact_generations.jsonl", samples)
    checkpoint_volume.commit()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    single_use_containers=True,
)
def preflight_grpo_setup(config_path: str) -> dict[str, Any]:
    import sys

    sys.path.insert(0, "/workspace/src")

    from trl import GRPOConfig, GRPOTrainer

    from humanize_rl.reward.grpo_dataset import load_grpo_dataset, load_grpo_rows
    from humanize_rl.reward.grpo_rewards import WEIGHTED_REWARD_FUNCS

    config = _load_config(config_path)
    started = perf_counter()
    model, tokenizer, model_load_seconds = _load_policy(config)
    model = _attach_lora(config, model)
    train_dataset = load_grpo_dataset(Path(config.task_path), split=config.train_split)
    train_rows = load_grpo_rows(Path(config.task_path), split=config.train_split)
    training_args = GRPOConfig(
        temperature=config.temperature,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_steps=max(1, int(config.max_steps * config.warmup_ratio)),
        lr_scheduler_type="linear",
        optim=config.optim,
        logging_steps=1,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        num_generations=config.num_generations,
        max_prompt_length=config.max_prompt_length,
        max_completion_length=config.max_completion_length,
        max_steps=1,
        save_steps=1,
        report_to="none",
        output_dir="/checkpoints/grpo-preflight/trainer",
        epsilon=config.epsilon,
        epsilon_high=config.epsilon_high,
        delta=config.delta,
        loss_type=config.loss_type,
        mask_truncated_completions=config.mask_truncated_completions,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=config.seed,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=WEIGHTED_REWARD_FUNCS,
        args=training_args,
        train_dataset=train_dataset,
    )
    sample_rows = train_rows[: config.num_generations]
    completions = [[{"role": "assistant", "content": "ok"}] for _ in sample_rows]
    reward_probe = [
        func(completions, task=[row["task"] for row in sample_rows])[0]
        for func in WEIGHTED_REWARD_FUNCS
    ]
    summary = {
        "model_name": config.model_name,
        "trainer_class": trainer.__class__.__name__,
        "train_rows": len(train_rows),
        "reward_probe": reward_probe,
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds": perf_counter() - started,
        "gpu": _gpu_snapshot(torch),
    }
    output_dir = Path("/checkpoints") / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "grpo_preflight_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    checkpoint_volume.commit()
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return summary


@app.local_entrypoint()
def main(
    config_path: str = "/workspace/configs/rl/gemma4_e2b_rl_smoke.yaml",
    mode: str = "train",
) -> None:
    if mode == "verify-artifact":
        print(
            f"Launching detached-safe artifact verification: config_path={config_path}"
        )
        call = verify_model_artifact.spawn(config_path)
        print(f"Spawned Modal call: {call.object_id}")
        return
    if mode == "preflight":
        print(f"Launching detached-safe GRPO preflight: config_path={config_path}")
        call = preflight_grpo_setup.spawn(config_path)
        print(f"Spawned Modal call: {call.object_id}")
        return
    if mode != "train":
        raise ValueError("mode must be one of: verify-artifact, preflight, train")
    print(f"Launching detached-safe Modal GRPO with spawn(): config_path={config_path}")
    call = train_grpo.spawn(config_path)
    print(f"Spawned Modal call: {call.object_id}")
