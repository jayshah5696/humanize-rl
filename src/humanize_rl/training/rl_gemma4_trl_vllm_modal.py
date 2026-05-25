"""Slice 1 — TRL + vLLM colocate GRPO entrypoint for Gemma 4 E2B.

Plain transformers + PEFT + TRL GRPOTrainer. No Unsloth import path.

See docs/plans/gemma4_rl_modal_20usd_budget_plan.md "Slice 1".

Run (detached, A100-40GB):
    rtk uvx modal run --detach \\
      src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \\
      --mode train \\
      --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml

The Modal entrypoint always uses ``.spawn(...)`` so the local CLI exits
once the remote call is registered (detached-safe).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local pytest path
    modal = None  # type: ignore[assignment]

GPU_TYPE = "A100-40GB"
TIMEOUT_HOURS = 4
MAX_RETRIES = 0

# Bug A — Gemma 4 final_logit_softcapping lives only on text_config; TRL
# reads it via flat getattr and silently resolves to 0 (no softcap) which
# breaks logprob parity with vLLM and explodes KL. We mirror it up
# manually after model load. See plan "Bug A" section.
GEMMA4_EXPECTED_FINAL_LOGIT_SOFTCAP = 30.0

if modal is not None:
    app = modal.App("humanize-rl-gemma4-trl-vllm")
    model_cache_volume = modal.Volume.from_name(
        "humanize-rl-model-cache", create_if_missing=True
    )
    checkpoint_volume = modal.Volume.from_name(
        "humanize-rl-checkpoints", create_if_missing=True
    )
else:  # pragma: no cover - local pytest path

    class _LocalStub:
        def function(self, *args: Any, **kwargs: Any):  # type: ignore[no-untyped-def]
            def deco(fn):  # type: ignore[no-untyped-def]
                return fn

            return deco

        def local_entrypoint(self):  # type: ignore[no-untyped-def]
            def deco(fn):  # type: ignore[no-untyped-def]
                return fn

            return deco

    app = _LocalStub()  # type: ignore[assignment]
    model_cache_volume = None  # type: ignore[assignment]
    checkpoint_volume = None  # type: ignore[assignment]

def _build_image() -> Any:
    if modal is None:  # pragma: no cover - local pytest path
        return None
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install(
            "torch>=2.8.0",
            "torchvision",
            "bitsandbytes>=0.46.0",
            "datasets>=4.0.0",
            "hf-transfer>=0.1.9",
            "huggingface_hub>=0.34.0",
            "peft>=0.13.0",
            "pydantic>=2.0.0",
            "pyyaml>=6.0.0",
            "sentencepiece>=0.2.0",
            "tokenizers>=0.22.0,<=0.23.0",
            # Bug C: use_cache=False corrupts Gemma 4 E2B before transformers 5.5.0.
            "transformers>=5.5.0",
            # TRL v1.x is the first line that supports transformers v5 and
            # native Gemma 4 (response schema + tool-use). It also makes Bug A
            # (final_logit_softcapping mirror) a defensive no-op because the
            # softcap is now applied inside model.forward. Bug B / E were
            # fixed before v1.0.0.
            "trl>=1.0.0,<1.2.0",
            # vLLM 0.19.1 is the first release with Gemma 4 + transformers 5.5.3.
            "vllm>=0.19.1,<0.21.0",
            "accelerate>=0.34.0",
            "wandb>=0.21.0",
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
            "configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml",
        )
    )


image = _build_image()

if modal is not None and image is not None:
    with image.imports():  # type: ignore[attr-defined]
        import torch  # noqa: F401
        import yaml  # noqa: F401
else:  # pragma: no cover - local pytest path
    import yaml  # noqa: F401


@dataclass
class GRPOProbeConfig:
    """Slice 1 probe config schema. Mirrors the YAML."""

    model_name: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"
    experiment_name: str | None = None
    hf_lora_repo: str = "jayshah5696/gemma4-e2b-humanize-rl-a100-probe-lora"
    task_path: str = "/workspace/data/rl/humanize_tasks_v01_smoke.jsonl"
    train_split: str = "train"
    eval_split: str = "validation"
    max_seq_length: int = 2048
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-5
    max_steps: int = 4
    num_generations: int = 6
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.1
    weight_decay: float = 0.001
    max_grad_norm: float = 0.5
    optim: str = "adamw_8bit"
    loss_type: str = "bnpo"
    epsilon: float = 0.2
    epsilon_high: float = 0.28
    delta: float = 1.5
    mask_truncated_completions: bool = True
    temperature: float = 1.0
    seed: int = 3407
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.5
    sample_before_after: int = 4
    generation_batch_size: int = 2
    artifact_generation_samples: int = 2
    push_to_hub: bool = False
    report_to: str = "none"

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_name = f"gemma4-e2b-humanize-trl-vllm-{timestamp}"


def _load_config(path: str) -> GRPOProbeConfig:
    config_path = Path(path)
    data = yaml.safe_load(config_path.read_text()) if config_path.exists() else {}
    if "task_path" in data and not str(data["task_path"]).startswith("/"):
        data["task_path"] = f"/workspace/{data['task_path']}"
    return GRPOProbeConfig(**data)


def mirror_gemma4_final_logit_softcap(model: Any) -> float | None:
    """Bug A guard.

    Gemma4 puts ``final_logit_softcapping`` on ``config.text_config`` only.
    TRL's GRPOTrainer reads ``getattr(model.config, "final_logit_softcapping", 0)``
    flat, so without this mirror the trainer applies softcap=0 → logprob
    divergence vs vLLM → KL blowup. Returns the mirrored value (or None).
    """
    text_cfg = getattr(model.config, "text_config", None)
    sc = getattr(text_cfg, "final_logit_softcapping", None) if text_cfg else None
    top = getattr(model.config, "final_logit_softcapping", None)
    if sc is not None and top in (None, 0, 0.0):
        model.config.final_logit_softcapping = sc
    return getattr(model.config, "final_logit_softcapping", None)


def assert_version_pins() -> dict[str, str]:
    """Hard-assert installed versions of trl/transformers/vllm."""
    import importlib.metadata as md

    def _ver(pkg: str) -> str:
        return md.version(pkg)

    pins = {
        "trl": _ver("trl"),
        "transformers": _ver("transformers"),
        "vllm": _ver("vllm"),
    }

    def _parse(v: str) -> tuple[int, ...]:
        return tuple(int(part) for part in v.split(".")[:3] if part.isdigit())

    if _parse(pins["trl"]) < (1, 0, 0):
        raise RuntimeError(
            f"trl {pins['trl']} < 1.0.0 (need v5 transformers + native Gemma 4)"
        )
    if _parse(pins["transformers"]) < (5, 5, 0):
        raise RuntimeError(
            f"transformers {pins['transformers']} < 5.5.0 (Bug C unfixed)"
        )
    if _parse(pins["vllm"]) < (0, 19, 1):
        raise RuntimeError(
            f"vllm {pins['vllm']} < 0.19.1 (no Gemma 4 + transformers v5 support)"
        )
    return pins


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
        "peak_allocated_gb": round(
            torch_module.cuda.max_memory_allocated(device) / 1024**3, 2
        ),
    }


def _load_policy(
    config: GRPOProbeConfig,
) -> tuple[Any, Any, float, float | None]:
    """Load Gemma 4 E2B with plain transformers, mirror Bug A softcap."""
    import os

    from transformers import AutoModelForImageTextToText, AutoProcessor

    started = perf_counter()
    model = AutoModelForImageTextToText.from_pretrained(
        config.model_name,
        torch_dtype=torch.bfloat16,
        token=os.environ.get("HF_TOKEN"),
    ).to("cuda")
    softcap = mirror_gemma4_final_logit_softcap(model)
    if softcap != GEMMA4_EXPECTED_FINAL_LOGIT_SOFTCAP:
        raise RuntimeError(
            "Bug A mirror failed: expected "
            f"final_logit_softcapping={GEMMA4_EXPECTED_FINAL_LOGIT_SOFTCAP}, "
            f"got {softcap}. Gemma 4 logits would not be softcapped."
        )
    processor = AutoProcessor.from_pretrained(
        config.model_name, token=os.environ.get("HF_TOKEN")
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(
        tokenizer, "eos_token", None
    ):
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, perf_counter() - started, softcap


def _attach_lora(config: GRPOProbeConfig, model: Any) -> Any:
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=config.lora_rank,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora_config)


def _train_grpo_decorator() -> Any:
    if modal is None:  # pragma: no cover - local pytest path
        def passthrough(fn):  # type: ignore[no-untyped-def]
            return fn

        return passthrough
    return app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={
            "/model_cache": model_cache_volume,
            "/checkpoints": checkpoint_volume,
        },
        secrets=[
            modal.Secret.from_name("huggingface"),
            modal.Secret.from_name("wandb"),
        ],
        timeout=TIMEOUT_HOURS * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
        single_use_containers=True,
    )


@_train_grpo_decorator()
def train_grpo(config_path: str) -> dict[str, Any]:
    """Slice 1 probe: 4 GRPO steps on A100-40GB, TRL+vLLM colocate."""
    import os
    import sys

    sys.path.insert(0, "/workspace/src")
    os.environ.setdefault("WANDB_DISABLED", "true")

    from trl import GRPOConfig, GRPOTrainer

    from humanize_rl.reward.grpo_dataset import load_grpo_dataset, load_grpo_rows
    from humanize_rl.reward.grpo_rewards import WEIGHTED_REWARD_FUNCS

    pins = assert_version_pins()
    print(f"[pins] {pins}", flush=True)

    config = _load_config(config_path)
    output_dir = Path("/checkpoints") / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config.model_name}", flush=True)
    run_started = perf_counter()
    model, tokenizer, model_load_seconds, softcap = _load_policy(config)
    print(
        f"Model loaded in {model_load_seconds:.1f}s, "
        f"final_logit_softcapping={softcap}",
        flush=True,
    )
    print(f"GPU after load: {_gpu_snapshot(torch)}", flush=True)

    model = _attach_lora(config, model)
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU after LoRA attach: {_gpu_snapshot(torch)}", flush=True)

    task_path = Path(config.task_path)
    train_dataset = load_grpo_dataset(task_path, split=config.train_split)
    train_rows = load_grpo_rows(task_path, split=config.train_split)

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
        max_grad_norm=config.max_grad_norm,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        seed=config.seed,
        # vLLM colocate
        use_vllm=config.use_vllm,
        vllm_mode=config.vllm_mode,
        vllm_gpu_memory_utilization=config.vllm_gpu_memory_utilization,
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

    log_history = list(getattr(trainer.state, "log_history", []))
    metrics = dict(getattr(trainer_stats, "metrics", {}) or {})
    gpu = _gpu_snapshot(torch)

    summary = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "hf_lora_repo": config.hf_lora_repo if config.push_to_hub else None,
        "train_rows": len(train_rows),
        "trainer_metrics": metrics,
        "log_history": log_history,
        "metrics": {
            "train_samples_per_second": metrics.get("train_samples_per_second"),
            "train_runtime": metrics.get("train_runtime"),
            "peak_vram_gb": gpu.get("peak_allocated_gb"),
        },
        "peak_vram_gb": gpu.get("peak_allocated_gb"),
        "train_runtime_seconds": metrics.get("train_runtime"),
        "steps_completed": config.max_steps,
        "model_load_seconds": model_load_seconds,
        "elapsed_seconds": perf_counter() - run_started,
        "gpu": gpu,
        "version_pins": pins,
        "final_logit_softcapping": softcap,
        "config": config.__dict__,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=str)
    )
    checkpoint_volume.commit()
    print(json.dumps({"final_logit_softcapping": softcap, "pins": pins}, indent=2))
    return summary


@app.local_entrypoint()
def main(
    config_path: str = (
        "/workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml"
    ),
    mode: str = "train",
) -> None:
    if mode != "train":
        raise SystemExit(f"unknown mode {mode!r}; only 'train' is supported")
    print(f"Launching detached-safe TRL+vLLM probe: config_path={config_path}")
    call = train_grpo.spawn(config_path)
    print(f"Spawned Modal call: {call.object_id}")
