from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local unit-test path
    modal = None  # type: ignore[assignment]

DEFAULT_DATA_FILES = "data/v3/v04_sft_final_plus_llama_failure_refs_env0314.jsonl"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass
class GenericSFTConfig:
    model_name: str = "Qwen/Qwen3.5-0.8B"
    dataset_name: str = "jayshah5696/humanize-rl-sft-dataset"
    data_files: str = DEFAULT_DATA_FILES
    dataset_split: str = "train"
    max_seq_length: int = 2048
    train_limit: int | None = 128
    eval_size: int = 16
    max_steps: int = 10
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    max_grad_norm: float = 0.3
    lr_scheduler_type: str = "linear"
    optim: str = "adamw_torch"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = DEFAULT_TARGET_MODULES
    seed: int = 3407
    logging_steps: int = 1
    save_steps: int = 10
    eval_steps: int = 10
    enable_thinking: bool = False
    trust_remote_code: bool = True
    push_to_hub: bool = False
    hf_lora_repo: str = "jayshah5696/qwen35-08b-humanize-sft-smoke-lora"
    experiment_name: str | None = None

    def __post_init__(self) -> None:
        if self.experiment_name is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_slug = self.model_name.replace("/", "-").replace(":", "-")
            self.experiment_name = f"{model_slug}-humanize-sft-smoke-{timestamp}"


def messages_from_row(row: dict[str, Any]) -> list[dict[str, str]]:
    messages = row.get("messages")
    if isinstance(messages, list) and messages:
        return [
            {"role": str(message["role"]), "content": str(message["content"])}
            for message in messages
            if isinstance(message, dict)
        ]

    instruction = str(row.get("instruction") or "").strip()
    response = str(row.get("response") or "").strip()
    if not instruction or not response:
        raise ValueError("SFT row missing instruction/response")
    return [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": response},
    ]


def apply_chat_template(
    tokenizer: Any, messages: list[dict[str, str]], *, enable_thinking: bool
) -> str:
    kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": False,
        "enable_thinking": enable_thinking,
    }
    try:
        return str(tokenizer.apply_chat_template(messages, **kwargs))
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return str(tokenizer.apply_chat_template(messages, **kwargs))


def sft_config_kwargs(
    config: GenericSFTConfig, supported_params: set[str]
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "dataset_text_field": "text",
        "per_device_train_batch_size": config.batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "max_steps": config.max_steps,
        "learning_rate": config.learning_rate,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "optim": config.optim,
        "weight_decay": config.weight_decay,
        "warmup_ratio": config.warmup_ratio,
        "max_grad_norm": config.max_grad_norm,
        "lr_scheduler_type": config.lr_scheduler_type,
        "seed": config.seed,
        "report_to": "wandb",
    }
    if "max_seq_length" in supported_params:
        kwargs["max_seq_length"] = config.max_seq_length
    elif "max_length" in supported_params:
        kwargs["max_length"] = config.max_seq_length
    return {key: value for key, value in kwargs.items() if key in supported_params}


def trainer_tokenizer_kwargs(
    tokenizer: Any, supported_params: set[str]
) -> dict[str, Any]:
    if "processing_class" in supported_params:
        return {"processing_class": tokenizer}
    if "tokenizer" in supported_params:
        return {"tokenizer": tokenizer}
    return {}


if modal is not None:
    app = modal.App("humanize-rl-generic-sft")
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
            "wandb>=0.21.0",
        )
        .env(
            {
                "HF_HOME": "/model_cache",
                "HF_HUB_ENABLE_HF_TRANSFER": "1",
                "HF_XET_HIGH_PERFORMANCE": "1",
            }
        )
    )

    with image.imports():
        import os

        import torch
        import wandb
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer

    def _format_example(row: dict[str, Any], tokenizer: Any, config: GenericSFTConfig):
        messages = messages_from_row(row)
        text = apply_chat_template(
            tokenizer, messages, enable_thinking=config.enable_thinking
        )
        return {"text": text}

    def _load_sft_dataset(config: GenericSFTConfig, tokenizer: Any):
        dataset = load_dataset(
            config.dataset_name,
            data_files=config.data_files,
            split=config.dataset_split,
        )
        if config.train_limit is not None:
            dataset = dataset.select(range(min(config.train_limit, len(dataset))))
        split = dataset.train_test_split(
            test_size=min(config.eval_size, max(1, len(dataset) // 10)),
            seed=config.seed,
        )
        train_dataset = split["train"].map(
            lambda row: _format_example(row, tokenizer, config),
            remove_columns=split["train"].column_names,
        )
        eval_dataset = split["test"].map(
            lambda row: _format_example(row, tokenizer, config),
            remove_columns=split["test"].column_names,
        )
        return train_dataset, eval_dataset

    @app.function(
        image=image,
        gpu="A100-40GB",
        volumes={
            "/model_cache": model_cache_volume,
            "/checkpoints": checkpoint_volume,
        },
        secrets=[modal.Secret.from_name("wandb"), modal.Secret.from_name("huggingface")],
        timeout=4 * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=0),
        single_use_containers=True,
    )
    def train(config: GenericSFTConfig) -> dict[str, Any]:
        output_dir = f"/checkpoints/{config.experiment_name}"
        wandb.init(
            project="humanize-rl",
            name=config.experiment_name,
            id=config.experiment_name,
            resume="allow",
            group="generic-sft-smoke",
            config=asdict(config),
        )

        tokenizer = AutoTokenizer.from_pretrained(
            config.model_name,
            trust_remote_code=config.trust_remote_code,
            token=os.environ.get("HF_TOKEN"),
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=dtype,
            trust_remote_code=config.trust_remote_code,
            token=os.environ.get("HF_TOKEN"),
        )

        train_dataset, eval_dataset = _load_sft_dataset(config, tokenizer)
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=list(config.target_modules),
        )

        sft_params = set(inspect.signature(SFTConfig).parameters)
        sft_kwargs = sft_config_kwargs(config, sft_params)
        sft_kwargs.update(
            {
                key: value
                for key, value in {
                    "bf16": torch.cuda.is_bf16_supported(),
                    "fp16": not torch.cuda.is_bf16_supported(),
                    "output_dir": output_dir,
                }.items()
                if key in sft_params
            }
        )
        trainer_params = set(inspect.signature(SFTTrainer).parameters)
        trainer = SFTTrainer(
            model=model,
            peft_config=peft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            args=SFTConfig(**sft_kwargs),
            **trainer_tokenizer_kwargs(tokenizer, trainer_params),
        )
        trainer_stats = trainer.train()
        final_adapter_path = f"{output_dir}/final_adapter"
        trainer.model.save_pretrained(final_adapter_path)
        tokenizer.save_pretrained(final_adapter_path)
        if config.push_to_hub:
            trainer.model.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))
            tokenizer.push_to_hub(config.hf_lora_repo, token=os.environ.get("HF_TOKEN"))
        checkpoint_volume.commit()
        wandb.finish()
        return {
            "experiment_name": config.experiment_name,
            "trainer_stats": str(trainer_stats),
            "output_dir": output_dir,
            "final_adapter_path": final_adapter_path,
            "hf_lora_repo": config.hf_lora_repo if config.push_to_hub else None,
            "push_to_hub": config.push_to_hub,
        }

    @app.local_entrypoint()
    def main(
        model_name: str = "Qwen/Qwen3.5-0.8B",
        data_files: str = DEFAULT_DATA_FILES,
        train_limit: int = 128,
        max_steps: int = 10,
        max_seq_length: int = 2048,
        batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        learning_rate: float = 2e-4,
        experiment_name: str | None = None,
        push_to_hub: bool = False,
        hf_lora_repo: str = "jayshah5696/qwen35-08b-humanize-sft-smoke-lora",
    ) -> None:
        config = GenericSFTConfig(
            model_name=model_name,
            data_files=data_files,
            train_limit=train_limit,
            max_steps=max_steps,
            max_seq_length=max_seq_length,
            batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
            push_to_hub=push_to_hub,
            hf_lora_repo=hf_lora_repo,
        )
        print(f"Launching generic SFT Modal training with spawn(): {config}")
        call = train.spawn(config)
        print(f"Spawned Modal call: {call.object_id}")

else:
    app = None
    train = None

    def main(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("modal is required to launch generic SFT training")
