from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import modal

app = modal.App("humanize-rl-gemma4-e2b-sft")

GPU_TYPE = "H100"
TIMEOUT_HOURS = 8
MAX_RETRIES = 0

model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
dataset_cache_volume = modal.Volume.from_name(
    "humanize-rl-dataset-cache", create_if_missing=True
)
checkpoint_volume = modal.Volume.from_name(
    "humanize-rl-checkpoints", create_if_missing=True
)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "bitsandbytes>=0.46.0",
        "datasets>=3.6.0",
        "hf-transfer>=0.1.9",
        "huggingface_hub>=0.34.0",
        "peft>=0.16.0",
        "protobuf>=5.0.0",
        "sentencepiece>=0.2.0",
        "transformers>=4.54.0",
        "trl>=0.19.1",
        "unsloth[cu128-torch270]>=2025.7.8",
        "unsloth_zoo>=2025.7.10",
        "wandb>=0.21.0",
    )
    .env({"HF_HOME": "/model_cache", "HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

with image.imports():
    import unsloth  # noqa: F401,I001
    import torch
    import wandb
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, train_on_responses_only


@dataclass
class TrainingConfig:
    model_name: str = "unsloth/gemma-4-E2B-it"
    dataset_name: str = "jayshah5696/humanize-rl-sft-dataset"
    max_seq_length: int = 512
    load_in_4bit: bool = False
    load_in_16bit: bool = True
    lora_r: int = 8
    lora_alpha: int = 8
    lora_dropout: float = 0.0
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_train_epochs: int = 1
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_8bit"
    logging_steps: int = 1
    save_steps: int = 100
    eval_steps: int = 100
    seed: int = 3407
    train_limit: int | None = None
    eval_size: int = 250
    experiment_name: str | None = None
    hf_lora_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"
    hf_merged_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.experiment_name = f"gemma4-e2b-humanize-r{self.lora_r}-{timestamp}"


def _format_messages(examples, tokenizer):
    texts = []
    for messages in examples["messages"]:
        texts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            ).removeprefix("<bos>")
        )
    return {"text": texts}


def _load_dataset(config: TrainingConfig, tokenizer):
    dataset = load_dataset(config.dataset_name, split="train")
    if config.train_limit is not None:
        dataset = dataset.select(range(min(config.train_limit, len(dataset))))
    split = dataset.train_test_split(
        test_size=min(config.eval_size, max(1, len(dataset) // 10)), seed=config.seed
    )
    train_dataset = split["train"]
    eval_dataset = split["test"]
    train_dataset = train_dataset.map(
        lambda batch: _format_messages(batch, tokenizer),
        batched=True,
        remove_columns=train_dataset.column_names,
        num_proc=2,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: _format_messages(batch, tokenizer),
        batched=True,
        remove_columns=eval_dataset.column_names,
        num_proc=2,
    )
    return train_dataset, eval_dataset


@app.function(
    image=image,
    gpu=GPU_TYPE,
    volumes={
        "/model_cache": model_cache_volume,
        "/dataset_cache": dataset_cache_volume,
        "/checkpoints": checkpoint_volume,
    },
    secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
    timeout=TIMEOUT_HOURS * 60 * 60,
    retries=modal.Retries(initial_delay=0.0, max_retries=MAX_RETRIES),
    single_use_containers=True,
)
def train(config: TrainingConfig) -> dict:
    import os

    output_dir = f"/checkpoints/{config.experiment_name}"
    wandb.init(
        project="humanize-rl",
        name=config.experiment_name,
        id=config.experiment_name,
        resume="allow",
        group="gemma4-e2b-modal-unsloth",
        config=config.__dict__,
    )

    model, tokenizer = FastModel.from_pretrained(
        model_name=config.model_name,
        max_seq_length=config.max_seq_length,
        dtype=None,
        load_in_4bit=config.load_in_4bit,
        load_in_16bit=config.load_in_16bit,
        full_finetuning=False,
        token=os.environ.get("HF_TOKEN"),
    )
    tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")

    model = FastModel.get_peft_model(
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

    train_dataset, eval_dataset = _load_dataset(config, tokenizer)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=config.batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            num_train_epochs=config.num_train_epochs,
            learning_rate=config.learning_rate,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            eval_strategy="steps",
            save_strategy="steps",
            optim=config.optim,
            weight_decay=config.weight_decay,
            warmup_ratio=config.warmup_ratio,
            max_grad_norm=config.max_grad_norm,
            lr_scheduler_type=config.lr_scheduler_type,
            bf16=torch.cuda.is_bf16_supported(),
            fp16=not torch.cuda.is_bf16_supported(),
            seed=config.seed,
            output_dir=output_dir,
            report_to="wandb",
        ),
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part="<|turn>user\n",
        response_part="<|turn>model\n",
    )

    checkpoint_root = Path(output_dir)
    checkpoints = (
        list(checkpoint_root.glob("checkpoint-*")) if checkpoint_root.exists() else []
    )
    resume_from_checkpoint = None
    if checkpoints:
        resume_from_checkpoint = str(
            max(checkpoints, key=lambda path: int(path.name.split("-")[-1]))
        )
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    final_adapter_path = f"{output_dir}/final_adapter"
    merged_path = f"{output_dir}/merged_16bit"
    model.save_pretrained(final_adapter_path)
    tokenizer.save_pretrained(final_adapter_path)

    model.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))
    tokenizer.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    model.push_to_hub_merged(
        config.hf_merged_repo,
        tokenizer,
        save_method="merged_16bit",
        token=os.environ.get("HF_TOKEN"),
    )

    checkpoint_volume.commit()
    wandb.finish()
    return {
        "experiment_name": config.experiment_name,
        "trainer_stats": str(trainer_stats),
        "output_dir": output_dir,
        "lora_repo": config.hf_lora_repo,
        "merged_repo": config.hf_merged_repo,
    }


@app.local_entrypoint()
def main(
    train_limit: int | None = None,
    lora_r: int = 8,
    lora_alpha: int = 8,
    max_seq_length: int = 512,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    num_train_epochs: int = 1,
    experiment_name: str | None = None,
):
    config = TrainingConfig(
        train_limit=train_limit,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=num_train_epochs,
        experiment_name=experiment_name,
    )
    print(f"Launching Modal training with spawn(): {config}")
    call = train.spawn(config)
    print(f"Spawned Modal call: {call.object_id}")
