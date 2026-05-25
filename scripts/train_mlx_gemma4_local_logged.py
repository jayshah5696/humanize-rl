#!/usr/bin/env python3
from __future__ import annotations

import csv
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
        data_dir=Path(os.environ.get("DATA_DIR", "data/processed/sft/gemma4_e2b_v04")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs_gemma4_humanize_mlx_logged")),
        adapter_path=Path(os.environ.get("ADAPTER_PATH", "adapters/gemma4_e2b_v04_mlx_logged")),
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
        wandb_run_name=os.environ.get("WANDB_RUN_NAME", "gemma4-e2b-mlx-logged"),
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
            {"role": "user", "content": [{"type": "text", "text": messages[0]["content"]}]},
            {"role": "assistant", "content": [{"type": "text", "text": messages[1]["content"]}]},
        ]
    }


def detect_hardware() -> dict[str, Any]:
    info: dict[str, Any] = {"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor()}
    for key, command in {
        "cpu_brand": ["sysctl", "-n", "machdep.cpu.brand_string"],
        "memsize_bytes": ["sysctl", "-n", "hw.memsize"],
    }.items():
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode == 0:
            info[key] = result.stdout.strip()
    if "memsize_bytes" in info:
        info["memsize_gb"] = round(int(info["memsize_bytes"]) / 1024**3, 2)
    return info


def total_optimizer_steps(dataset_size: int, config: LocalMlxConfig) -> int:
    if config.max_steps is not None:
        return config.max_steps
    return max(1, dataset_size // config.effective_batch_size)


def init_wandb(config: LocalMlxConfig, dataset_size: int, total_steps: int) -> Any | None:
    if config.report_to != "wandb":
        return None
    import wandb

    if not os.environ.get("WANDB_API_KEY") and not env_bool("WANDB_MODE_OFFLINE", False):
        os.environ.setdefault("WANDB_MODE", "offline")
    if env_bool("WANDB_MODE_OFFLINE", False):
        os.environ["WANDB_MODE"] = "offline"

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


def train_with_logging(model: Any, processor: Any, dataset: list[dict[str, Any]], config: LocalMlxConfig, steps: int, wandb_run: Any | None) -> dict[str, float]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import mlx_tune.vlm as vlm
    from mlx.utils import tree_map
    from mlx_tune import UnslothVisionDataCollator
    from tqdm import tqdm

    model.train()
    optimizer = optim.Adam(learning_rate=config.learning_rate)
    assistant_id = None
    if config.train_on_completions:
        assistant_id = vlm._detect_assistant_role_token(processor)
    trainer = vlm._VLMTrainerShim(model.model if hasattr(model, "model") else model, optimizer, train_on_completions=config.train_on_completions, assistant_id=assistant_id)
    collator = UnslothVisionDataCollator(model, processor)
    loss_and_grad_fn = nn.value_and_grad(trainer.model, trainer.loss_fn)

    csv_path = config.output_dir / "loss_history.csv"
    csv_file = csv_path.open("w", newline="")
    writer = csv.DictWriter(csv_file, fieldnames=["step", "micro_step", "loss", "avg_loss", "elapsed_seconds", "learning_rate"])
    writer.writeheader()

    progress = tqdm(range(steps), desc="Training")
    started = time.perf_counter()
    total_loss = 0.0
    step = 0
    micro_step = 0
    accum_loss = 0.0
    accumulated_grads = None

    while step < steps:
        for index in range(0, len(dataset), config.batch_size):
            if step >= steps:
                break
            batch_samples = dataset[index : index + config.batch_size]
            batch = collator(batch_samples)
            loss, grads = loss_and_grad_fn(trainer.model, batch)
            mx.eval(loss)
            accumulated_grads = grads if accumulated_grads is None else tree_map(lambda a, g: a + g, accumulated_grads, grads)
            accum_loss += loss.item()
            micro_step += 1

            if micro_step >= config.gradient_accumulation_steps:
                averaged_grads = tree_map(lambda g: g / config.gradient_accumulation_steps, accumulated_grads)
                trainer.optimizer.update(trainer.model, averaged_grads)
                mx.eval(trainer.model, trainer.optimizer.state)
                loss_value = accum_loss / config.gradient_accumulation_steps
                total_loss += loss_value
                step += 1
                avg_loss = total_loss / step
                elapsed = time.perf_counter() - started
                row = {
                    "step": step,
                    "micro_step": micro_step,
                    "loss": loss_value,
                    "avg_loss": avg_loss,
                    "elapsed_seconds": elapsed,
                    "learning_rate": config.learning_rate,
                }
                writer.writerow(row)
                csv_file.flush()
                if wandb_run is not None:
                    wandb_run.log({"train/loss": loss_value, "train/avg_loss": avg_loss, "train/elapsed_seconds": elapsed, "train/learning_rate": config.learning_rate}, step=step)
                micro_step = 0
                accum_loss = 0.0
                accumulated_grads = None
                progress.update(1)
                progress.set_postfix({"loss": f"{loss_value:.4f}", "avg_loss": f"{avg_loss:.4f}"})
        if len(dataset) == 0:
            break

    progress.close()
    csv_file.close()
    adapter_dir = config.output_dir / "adapters"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    vlm.save_adapter(trainer.model, str(adapter_dir / "adapters.safetensors"))
    if hasattr(model, "_adapter_path"):
        model._adapter_path = adapter_dir
    return {"train_loss": total_loss / max(step, 1), "steps": float(step), "runtime_seconds": time.perf_counter() - started}


def main() -> None:
    from mlx_tune import FastVisionModel

    config = default_config()
    train_path = config.data_dir / "train.jsonl"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train data: {train_path}")

    rows = load_rows(train_path, config.train_limit)
    dataset = [to_vlm_text_messages(row) for row in rows]
    steps = total_optimizer_steps(len(dataset), config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    config.adapter_path.parent.mkdir(parents=True, exist_ok=True)
    run_config = {
        **asdict(config),
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "adapter_path": str(config.adapter_path),
        "effective_batch_size": config.effective_batch_size,
        "total_optimizer_steps": steps,
        "dataset_size": len(dataset),
        "hardware": detect_hardware(),
    }
    (config.output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2))
    wandb_run = init_wandb(config, len(dataset), steps)

    print("=" * 70)
    print("Gemma 4 E2B MLX-Tune local training with dense loss logging")
    print("=" * 70)
    print(json.dumps(run_config, indent=2))

    model, processor = FastVisionModel.from_pretrained(config.model_name, load_in_4bit=True)
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

    metrics = train_with_logging(model, processor, dataset, config, steps, wandb_run)
    metrics["rows"] = len(dataset)
    metrics["seconds_per_step"] = metrics["runtime_seconds"] / max(metrics["steps"], 1)
    (config.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    model.save_pretrained(str(config.adapter_path))
    if wandb_run is not None:
        import wandb

        artifact = wandb.Artifact("gemma4-e2b-mlx-local-adapter", type="model")
        artifact.add_dir(str(config.adapter_path))
        wandb_run.log_artifact(artifact)
        wandb_run.summary.update(metrics)
        wandb_run.finish()
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
