#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LocalMlxConfig:
    model_name: str
    data_dir: Path
    output_dir: Path
    adapter_path: Path
    max_seq_length: int
    max_steps: int | None
    train_limit: int | None
    batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    warmup_steps: int
    weight_decay: float
    seed: int
    report_to: str
    wandb_project: str
    wandb_run_name: str
    wandb_group: str
    train_on_completions: bool

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


def default_config() -> LocalMlxConfig:
    return LocalMlxConfig(
        model_name=os.environ.get("MODEL", "mlx-community/gemma-4-e2b-it-4bit"),
        data_dir=Path(os.environ.get("DATA_DIR", "data/processed/sft/gemma4_e2b_v04_mlx_smoke")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs_gemma4_humanize_mlx_local")),
        adapter_path=Path(os.environ.get("ADAPTER_PATH", "adapters/gemma4_e2b_v04_mlx_local")),
        max_seq_length=int(os.environ.get("MAX_SEQ_LENGTH", "512")),
        max_steps=int(os.environ["MAX_STEPS"]) if "MAX_STEPS" in os.environ else None,
        train_limit=int(os.environ["TRAIN_LIMIT"]) if "TRAIN_LIMIT" in os.environ else None,
        batch_size=int(os.environ.get("BATCH_SIZE", "1")),
        gradient_accumulation_steps=int(os.environ.get("GRAD_ACCUM", "4")),
        learning_rate=float(os.environ.get("LEARNING_RATE", "2e-4")),
        lora_r=int(os.environ.get("LORA_R", "8")),
        lora_alpha=int(os.environ.get("LORA_ALPHA", os.environ.get("LORA_R", "8"))),
        lora_dropout=float(os.environ.get("LORA_DROPOUT", "0.0")),
        warmup_steps=int(os.environ.get("WARMUP_STEPS", "5")),
        weight_decay=float(os.environ.get("WEIGHT_DECAY", "0.001")),
        seed=int(os.environ.get("SEED", "3407")),
        report_to=os.environ.get("REPORT_TO", "wandb"),
        wandb_project=os.environ.get("WANDB_PROJECT", "humanize-rl"),
        wandb_run_name=os.environ.get("WANDB_RUN_NAME", "gemma4-e2b-mlx-local"),
        wandb_group=os.environ.get("WANDB_GROUP", "gemma4-e2b-sft-v04-mlx"),
        train_on_completions=env_bool("TRAIN_ON_COMPLETIONS", True),
    )


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


def to_vlm_text_messages(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    messages = row["messages"]
    return {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": messages[0]["content"]}],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": messages[1]["content"]}],
            },
        ]
    }


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    commands = {
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "memsize_bytes": ["sysctl", "-n", "hw.memsize"],
    }
    for key, command in commands.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info[key] = result.stdout.strip()
    if "memsize_bytes" in info:
        info["memsize_gb"] = round(int(info["memsize_bytes"]) / 1024**3, 2)
    return info


def total_optimizer_steps(dataset_size: int, config: LocalMlxConfig) -> int:
    if config.max_steps is not None:
        return config.max_steps
    updates_per_epoch = dataset_size // config.effective_batch_size
    return max(1, updates_per_epoch)


def init_wandb(config: LocalMlxConfig, dataset_size: int, total_steps: int) -> Any | None:
    if config.report_to != "wandb":
        return None
    if not os.environ.get("WANDB_API_KEY") and not env_bool("WANDB_MODE_OFFLINE", False):
        os.environ.setdefault("WANDB_MODE", "offline")
    if env_bool("WANDB_MODE_OFFLINE", False):
        os.environ["WANDB_MODE"] = "offline"

    import wandb

    return wandb.init(
        project=config.wandb_project,
        name=config.wandb_run_name,
        group=config.wandb_group,
        config={
            **asdict(config),
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "adapter_path": str(config.adapter_path),
            "dataset_size": dataset_size,
            "total_optimizer_steps": total_steps,
            "effective_batch_size": config.effective_batch_size,
            "hardware": detect_hardware(),
        },
    )


def main() -> None:
    from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
    from mlx_tune.vlm import VLMSFTConfig

    config = default_config()
    train_path = config.data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train data: {train_path}")

    rows = load_rows(train_path, config.train_limit)
    converted_dataset = [to_vlm_text_messages(row) for row in rows]
    steps = total_optimizer_steps(len(converted_dataset), config)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_path.parent.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "run_config.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "data_dir": str(config.data_dir),
                "output_dir": str(config.output_dir),
                "adapter_path": str(config.adapter_path),
                "effective_batch_size": config.effective_batch_size,
                "total_optimizer_steps": steps,
                "dataset_size": len(converted_dataset),
                "hardware": detect_hardware(),
            },
            indent=2,
        )
    )

    wandb_run = init_wandb(config, len(converted_dataset), steps)
    started = time.perf_counter()

    print("=" * 70)
    print("Gemma 4 E2B MLX-Tune local training")
    print("=" * 70)
    print(f"Model: {config.model_name}")
    print(f"Rows: {len(converted_dataset)}")
    print(f"Steps: {steps}")
    print(f"Effective batch size: {config.effective_batch_size}")
    print(f"LoRA: r={config.lora_r}, alpha={config.lora_alpha}, dropout={config.lora_dropout}")
    print(f"Max seq length: {config.max_seq_length}")
    print(f"Output: {config.output_dir}")
    print(f"Adapter: {config.adapter_path}")

    model, processor = FastVisionModel.from_pretrained(
        config.model_name,
        load_in_4bit=True,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        random_state=config.seed,
    )

    FastVisionModel.for_training(model)
    trainer = VLMSFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=converted_dataset,
        args=VLMSFTConfig(
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            warmup_steps=config.warmup_steps,
            max_steps=steps,
            learning_rate=config.learning_rate,
            logging_steps=1,
            optim="adam",
            weight_decay=config.weight_decay,
            lr_scheduler_type="linear",
            seed=config.seed,
            output_dir=str(config.output_dir),
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=config.max_seq_length,
            train_on_completions=config.train_on_completions,
        ),
    )
    trainer_stats = trainer.train()
    runtime_seconds = time.perf_counter() - started
    metrics = {
        **trainer_stats.metrics,
        "runtime_seconds": runtime_seconds,
        "rows": len(converted_dataset),
        "steps": steps,
        "seconds_per_step": runtime_seconds / max(steps, 1),
    }
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    if wandb_run is not None:
        wandb_run.log(metrics)
        wandb_run.summary.update(metrics)

    FastVisionModel.for_inference(model)
    eval_prompts = [
        "Draft a quick Slack update saying staging is fixed.",
        "Write a concise email telling a client the hotfix is live and monitoring is ongoing.",
        "Rewrite this to sound less stiff: 'Please be advised that the deployment has completed successfully.'",
    ]
    generations: list[dict[str, str]] = []
    for prompt in eval_prompts:
        try:
            response = model.generate(prompt=prompt, max_tokens=96, temperature=0.3)
        except Exception as exc:
            response = f"GENERATION_ERROR: {exc}"
        generations.append({"prompt": prompt, "response": response})
    (config.output_dir / "eval_generations.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in generations)
    )

    model.save_pretrained(str(config.adapter_path))
    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact("gemma4-e2b-mlx-local-adapter", type="model")
        artifact.add_dir(str(config.adapter_path))
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

    print(json.dumps(metrics, indent=2))
    print(f"Saved adapter to {config.adapter_path}")


if __name__ == "__main__":
    main()
