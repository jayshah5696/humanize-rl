#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def load_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
        if len(rows) >= limit:
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


def main() -> None:
    from mlx_tune import FastVisionModel, UnslothVisionDataCollator, VLMSFTTrainer
    from mlx_tune.vlm import VLMSFTConfig

    model_name = os.environ.get("MODEL", "mlx-community/gemma-4-e2b-it-4bit")
    data_dir = Path(os.environ.get("DATA_DIR", "data/processed/sft/gemma4_e2b_v04_mlx_smoke"))
    output_dir = os.environ.get("OUTPUT_DIR", "outputs_gemma4_humanize_mlx_smoke")
    adapter_path = os.environ.get("ADAPTER_PATH", "adapters/gemma4_e2b_v04_mlx_tune_smoke")
    max_seq_length = int(os.environ.get("MAX_SEQ_LENGTH", "512"))
    max_steps = int(os.environ.get("MAX_STEPS", os.environ.get("ITERS", "2")))
    train_limit = int(os.environ.get("TRAIN_LIMIT", "8"))
    batch_size = int(os.environ.get("BATCH_SIZE", "1"))
    grad_accum = int(os.environ.get("GRAD_ACCUM", "4"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))
    lora_rank = int(os.environ.get("LORA_R", "4"))
    lora_alpha = int(os.environ.get("LORA_ALPHA", str(lora_rank)))

    rows = load_rows(data_dir / "train.jsonl", train_limit)
    converted_dataset = [to_vlm_text_messages(row) for row in rows]

    print(f"Loading {model_name}")
    model, processor = FastVisionModel.from_pretrained(
        model_name,
        load_in_4bit=True,
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0,
        bias="none",
        random_state=3407,
    )

    print("Pre-training inference smoke")
    FastVisionModel.for_inference(model)
    try:
        response = model.generate(
            prompt="Draft a quick Slack update saying staging is fixed.",
            max_tokens=64,
            temperature=0.3,
        )
        print(f"Pre-training response: {response}")
    except Exception as exc:
        print(f"Pre-training inference error: {exc}")

    print("Training")
    FastVisionModel.for_training(model)
    trainer = VLMSFTTrainer(
        model=model,
        tokenizer=processor,
        data_collator=UnslothVisionDataCollator(model, processor),
        train_dataset=converted_dataset,
        args=VLMSFTConfig(
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            warmup_steps=1,
            max_steps=max_steps,
            learning_rate=learning_rate,
            logging_steps=1,
            optim="adam",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=output_dir,
            report_to="none",
            remove_unused_columns=False,
            dataset_text_field="",
            dataset_kwargs={"skip_prepare_dataset": True},
            max_length=max_seq_length,
        ),
    )
    trainer_stats = trainer.train()
    print(f"Training metrics: {trainer_stats.metrics}")

    print("Post-training inference smoke")
    FastVisionModel.for_inference(model)
    try:
        response = model.generate(
            prompt="Draft a quick Slack update saying staging is fixed.",
            max_tokens=64,
            temperature=0.3,
        )
        print(f"Post-training response: {response}")
    except Exception as exc:
        print(f"Post-training inference error: {exc}")

    model.save_pretrained(adapter_path)
    print(f"MLX-Tune smoke adapter written to {adapter_path}")


if __name__ == "__main__":
    main()
