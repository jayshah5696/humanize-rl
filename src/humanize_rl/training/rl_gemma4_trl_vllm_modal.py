"""Slices 1–3 — TRL + vLLM colocate GRPO entrypoint for Gemma 4 E2B.

Plain transformers + PEFT + TRL GRPOTrainer. No Unsloth import path.

See docs/plans/gemma4_rl_modal_20usd_budget_plan.md.

Slice 1 probe run (detached, A100-40GB):
    uvx modal run --detach \\
      src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \\
      --mode train \\
      --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml

Slice 2 pilot run (detached, A100-40GB, 50 steps, ~$3.50):
    uvx modal run --detach \\
      src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \\
      --mode train \\
      --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_pilot.yaml

Slice 3 full run (detached, A100-40GB):
    uvx modal run --detach \\
      src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py \\
      --mode train \\
      --config-path /workspace/configs/rl/gemma4_e2b_rl_a100_full.yaml

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
A100_40GB_USD_PER_SEC = 0.000583

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
            # Ridge scorer deps — required for 50/50 reward formula.
            # baselines.py imports fasttext at module level; sklearn is needed
            # for TfidfVectorizer + Ridge deserialization.
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
            "configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml",
        )
        .add_local_file(
            "configs/rl/gemma4_e2b_rl_a100_pilot.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_pilot.yaml",
        )
        .add_local_file(
            "configs/rl/gemma4_e2b_rl_a100_full.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_full.yaml",
        )
        # Slice 4 filtered mix — softened reward, mid-band thresholds.
        .add_local_file(
            "data/rl/humanize_tasks_rl_mix_v1_filtered_softened_midband.jsonl",
            remote_path="/workspace/data/rl/humanize_tasks_rl_mix_v1_filtered_softened_midband.jsonl",
        )
        # Slice 5 ablation configs (A0/A1/A2).
        .add_local_file(
            "configs/rl/ablations/reward_a0_current_components.yaml",
            remote_path="/workspace/configs/rl/ablations/reward_a0_current_components.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/reward_a1_scalar_current.yaml",
            remote_path="/workspace/configs/rl/ablations/reward_a1_scalar_current.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/reward_a2_scalar_softened.yaml",
            remote_path="/workspace/configs/rl/ablations/reward_a2_scalar_softened.yaml",
        )
        # Slice 6 stability ablations (B0/B1/B2).
        .add_local_file(
            "configs/rl/ablations/stability_b0_g8_acc8_lr2e5.yaml",
            remote_path="/workspace/configs/rl/ablations/stability_b0_g8_acc8_lr2e5.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/stability_b1_g16_acc16_lr1e5.yaml",
            remote_path="/workspace/configs/rl/ablations/stability_b1_g16_acc16_lr1e5.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/stability_b2_g16_acc16_lr2e5.yaml",
            remote_path="/workspace/configs/rl/ablations/stability_b2_g16_acc16_lr2e5.yaml",
        )
        # Slice 7 reward-scaling ablations (C0/C1/C2).
        .add_local_file(
            "configs/rl/ablations/scale_c0_group.yaml",
            remote_path="/workspace/configs/rl/ablations/scale_c0_group.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/scale_c1_batch.yaml",
            remote_path="/workspace/configs/rl/ablations/scale_c1_batch.yaml",
        )
        .add_local_file(
            "configs/rl/ablations/scale_c2_none.yaml",
            remote_path="/workspace/configs/rl/ablations/scale_c2_none.yaml",
        )
        # Slice 8 pilot config (200-step B1 + Slice-7 winner).
        .add_local_file(
            "configs/rl/gemma4_e2b_rl_a100_mix_v1_pilot.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_mix_v1_pilot.yaml",
        )
        # Full run — mix_v2 (v01+v02+v03 filtered) + 600-step config.
        .add_local_file(
            "data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
            remote_path="/workspace/data/rl/humanize_tasks_rl_mix_v2_filtered_softened_midband.jsonl",
        )
        .add_local_file(
            "configs/rl/gemma4_e2b_rl_a100_full_v2.yaml",
            remote_path="/workspace/configs/rl/gemma4_e2b_rl_a100_full_v2.yaml",
        )
        # Ridge scorer pkls — required for 50/50 ridge+deterministic reward.
        # Without these, load_ridge_scorer() returns None and training falls
        # back to deterministic_only (ridge contributes 0%).
        .add_local_file(
            "models/track_a_10k/ridge.pkl",
            remote_path="/workspace/models/track_a_10k/ridge.pkl",
        )
        .add_local_file(
            "models/distilled/baseline_ridge.pkl",
            remote_path="/workspace/models/distilled/baseline_ridge.pkl",
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
    # Must satisfy: (pdtbs * world * grad_accum) % num_generations == 0
    # (TRL v1.x asserts this; see grpo_config.py __post_init__).
    gradient_accumulation_steps: int = 6
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
    wandb_project: str = "humanize-rl"
    wandb_entity: str | None = None
    stratify_batches: bool = False
    stratify_by: str = "reward_profile"
    # Reward-mode refactor (plan §6 / Slice 1).
    reward_mode: str = "current_components"
    penalty_cap: float = 1.0
    ridge_weight: float = 0.45
    deterministic_weight: float = 0.35
    risk_weight: float = 0.20
    # Slice 7 — reward scaling strategy (plan §8 Ablation C).
    # TRL v1.x GRPOConfig accepts: True/"group" (default), "batch",
    # False/"none". YAML can pass any of these; we forward verbatim.
    scale_rewards: str | bool = "group"

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


def patch_vllm_gemma4_kv_shared_k_norm() -> bool:
    """Bug G — vLLM 0.20.x loads Gemma 4 with strict k_norm requirement.

    Gemma 4 E2B has ``num_kv_shared_layers=20``. The last 20 decoder
    layers reuse KV from earlier layers and **never apply k_norm in
    forward()**, so their checkpoints omit ``k_norm.weight``. vLLM
    0.20.2's ``Gemma4Attention.__init__`` unconditionally builds a
    learnable ``RMSNorm`` for every layer's k_norm, then the weight
    loader strict-checks and fails:

        ValueError: Following weights were not initialized from
        checkpoint: {... .layers.<15..34>.self_attn.k_norm.weight}

    Upstream fix is vLLM PR #40117 (open since 2026-04-17, unmerged).
    We apply the 4-line equivalent inline by wrapping the constructor:
    when ``self.is_kv_shared_layer`` is True, replace the just-built
    ``self.k_norm`` with a weightless ``RMSNorm(has_weight=False)``.
    Returns True if the patch was applied, False if vllm isn't
    importable (local pytest path).
    """
    try:
        from vllm.model_executor.layers.layernorm import RMSNorm
        from vllm.model_executor.models import gemma4 as vllm_gemma4
    except ImportError:  # pragma: no cover - local pytest path
        return False

    Gemma4Attention = vllm_gemma4.Gemma4Attention
    if getattr(Gemma4Attention, "_humanize_rl_kv_shared_k_norm_patched", False):
        return True

    original_init = Gemma4Attention.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        if getattr(self, "is_kv_shared_layer", False):
            eps = self.k_norm.variance_epsilon
            self.k_norm = RMSNorm(self.head_dim, eps=eps, has_weight=False)

    Gemma4Attention.__init__ = patched_init
    Gemma4Attention._humanize_rl_kv_shared_k_norm_patched = True
    return True


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


def _patch_peft_for_gemma4_clippable_linear() -> Any:
    """Bug F (Gemma 4) — PEFT can't see ``Gemma4ClippableLinear``.

    Gemma 4 wraps every ``nn.Linear`` projection in a
    ``Gemma4ClippableLinear`` for activation clamping. The wrapper does
    NOT inherit from ``nn.Linear`` (transformers PR #45388 was closed,
    not merged). PEFT's LoRA ``_create_new_module`` only knows the exact
    ``torch.nn.Linear`` class → PEFT raises
    ``ValueError: Target module Gemma4ClippableLinear(...) is not supported``.

    Recipe lifted from unslothai/unsloth#4807: monkey-patch
    ``LoraModel._create_and_replace`` so that whenever it encounters a
    ``Gemma4ClippableLinear`` target it recurses into the inner
    ``target.linear`` (a plain ``nn.Linear``). Returns a callable that
    restores the original method.
    """
    try:
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4ClippableLinear,
        )
    except ImportError:  # pragma: no cover - older transformers
        return lambda: None

    from peft.tuners.lora.model import LoraModel

    original = LoraModel._create_and_replace

    def _patched(
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
            return original(
                self,
                peft_config,
                adapter_name,
                target.linear,
                "linear",
                target,
                current_key=current_key,
                **kwargs,
            )
        return original(
            self,
            peft_config,
            adapter_name,
            target,
            target_name,
            parent,
            current_key=current_key,
            **kwargs,
        )

    LoraModel._create_and_replace = _patched

    def restore() -> None:
        LoraModel._create_and_replace = original

    return restore


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
    restore = _patch_peft_for_gemma4_clippable_linear()
    try:
        return get_peft_model(model, lora_config)
    finally:
        restore()


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
    os.chdir("/workspace")  # make relative paths (models/*, data/*) resolve correctly

    from trl import GRPOConfig, GRPOTrainer

    from humanize_rl.reward.grpo_dataset import load_grpo_dataset, load_grpo_rows
    from humanize_rl.reward.grpo_rewards import (
        _RIDGE_SCORER,
        RewardModeConfig,
        build_reward_funcs,
    )

    pins = assert_version_pins()
    print(
        f"[ridge-scorer] {'LOADED — 50/50 reward active' if _RIDGE_SCORER is not None else 'MISSING — deterministic_only fallback'}",
        flush=True,
    )
    print(f"[pins] {pins}", flush=True)

    # Bug G guard — must run BEFORE vLLM imports Gemma4Attention via TRL.
    kv_shared_patched = patch_vllm_gemma4_kv_shared_k_norm()
    print(f"[vllm-gemma4-kv-shared-k-norm-patched] {kv_shared_patched}", flush=True)

    config = _load_config(config_path)
    if config.report_to == "wandb":
        os.environ.pop("WANDB_DISABLED", None)
        os.environ.setdefault("WANDB_PROJECT", config.wandb_project)
        os.environ.setdefault("WANDB_RUN_ID", str(config.experiment_name))
        os.environ.setdefault("WANDB_RESUME", "allow")
        if config.wandb_entity:
            os.environ.setdefault("WANDB_ENTITY", config.wandb_entity)
    else:
        os.environ.setdefault("WANDB_DISABLED", "true")

    output_dir = Path("/checkpoints") / str(config.experiment_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {config.model_name}", flush=True)
    run_started = perf_counter()
    model, tokenizer, model_load_seconds, softcap = _load_policy(config)
    print(
        f"Model loaded in {model_load_seconds:.1f}s, final_logit_softcapping={softcap}",
        flush=True,
    )
    print(f"GPU after load: {_gpu_snapshot(torch)}", flush=True)

    model = _attach_lora(config, model)
    torch.cuda.reset_peak_memory_stats()
    print(f"GPU after LoRA attach: {_gpu_snapshot(torch)}", flush=True)

    task_path = Path(config.task_path)
    stratify_batch_size = (
        config.gradient_accumulation_steps if config.stratify_batches else None
    )
    train_dataset = load_grpo_dataset(
        task_path,
        split=config.train_split,
        stratify_batch_size=stratify_batch_size,
        stratify_by=config.stratify_by,
    )
    train_rows = load_grpo_rows(
        task_path,
        split=config.train_split,
        stratify_batch_size=stratify_batch_size,
        stratify_by=config.stratify_by,
    )

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
        # TRL v1.x removed `max_prompt_length` (PR #4300); prompts are now
        # expected to fit naturally. We still keep `config.max_prompt_length`
        # for the schema test + future prompt-pre-truncation logic.
        max_completion_length=config.max_completion_length,
        max_steps=config.max_steps,
        save_steps=max(config.max_steps, 1),
        report_to=config.report_to,
        run_name=str(config.experiment_name),
        output_dir=str(output_dir / "trainer"),
        train_sampling_strategy=("sequential" if config.stratify_batches else "random"),
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
        # Slice 7 — reward-scaling ablation (plan §8 Ablation C).
        scale_rewards=config.scale_rewards,
    )
    print(
        f"[scale-rewards] {config.scale_rewards!r}",
        flush=True,
    )

    reward_cfg = RewardModeConfig(
        mode=config.reward_mode,  # type: ignore[arg-type]
        penalty_cap=config.penalty_cap,
        ridge_weight=config.ridge_weight,
        deterministic_weight=config.deterministic_weight,
        risk_weight=config.risk_weight,
    )
    reward_funcs = build_reward_funcs(reward_cfg)
    print(
        f"[reward-mode] {reward_cfg.mode} (n_funcs={len(reward_funcs)}, "
        f"penalty_cap={reward_cfg.penalty_cap})",
        flush=True,
    )

    from humanize_rl.training.wandb_ema_callback import EMATracker, build_callback

    ema_tracker = EMATracker()
    ema_callback = build_callback(ema_tracker)
    callbacks = [ema_callback] if ema_callback is not None else None
    print(
        f"[ema-callback] {'attached' if ema_callback is not None else 'unavailable'}",
        flush=True,
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        callbacks=callbacks,
    )

    trainer_stats = trainer.train()

    adapter_dir = output_dir / "final_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    uploaded_repo: str | None = None
    if config.push_to_hub:
        from huggingface_hub import HfApi

        token = os.environ.get("HF_TOKEN")
        model.push_to_hub(config.hf_lora_repo, token=token)
        tokenizer.push_to_hub(config.hf_lora_repo, token=token)
        HfApi().upload_folder(
            repo_id=config.hf_lora_repo,
            folder_path=str(output_dir),
            path_in_repo="run_artifacts",
            token=token,
            ignore_patterns=["trainer/checkpoint-*"],
        )
        uploaded_repo = config.hf_lora_repo

    log_history = list(getattr(trainer.state, "log_history", []))
    metrics = dict(getattr(trainer_stats, "metrics", {}) or {})
    gpu = _gpu_snapshot(torch)
    elapsed_seconds = perf_counter() - run_started

    summary = {
        "experiment_name": config.experiment_name,
        "model_name": config.model_name,
        "output_dir": str(output_dir),
        "adapter_dir": str(adapter_dir),
        "lora_path": str(adapter_dir),
        "hf_lora_repo": uploaded_repo,
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
        "elapsed_seconds": elapsed_seconds,
        "actual_cost_usd": round(elapsed_seconds * A100_40GB_USD_PER_SEC, 4),
        "gpu": gpu,
        "version_pins": pins,
        "final_logit_softcapping": softcap,
        "ridge_scorer_loaded": _RIDGE_SCORER is not None,
        "reward_profile": "50_50_ridge_deterministic"
        if _RIDGE_SCORER is not None
        else "deterministic_only",
        "final_ema": ema_tracker.snapshot(),
        "stratified_batches": config.stratify_batches,
        "stratify_by": config.stratify_by,
        "wandb_project": config.wandb_project if config.report_to == "wandb" else None,
        "wandb_run_id": str(config.experiment_name)
        if config.report_to == "wandb"
        else None,
        "vllm_gemma4_kv_shared_k_norm_patched": kv_shared_patched,
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
    config_path: str = ("/workspace/configs/rl/gemma4_e2b_rl_a100_capacity_probe.yaml"),
    mode: str = "train",
) -> None:
    if mode != "train":
        raise SystemExit(f"unknown mode {mode!r}; only 'train' is supported")
    print(f"Launching detached-safe TRL+vLLM probe: config_path={config_path}")
    call = train_grpo.spawn(config_path)
    print(f"Spawned Modal call: {call.object_id}")
