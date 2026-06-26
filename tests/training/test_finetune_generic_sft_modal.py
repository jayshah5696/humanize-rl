from __future__ import annotations

import pytest

from humanize_rl.training.finetune_generic_sft_modal import (
    DEFAULT_DATA_FILES,
    GenericSFTConfig,
    apply_chat_template,
    messages_from_row,
    sft_config_kwargs,
    trainer_tokenizer_kwargs,
)


def test_default_config_targets_qwen_smoke_and_uploaded_sft_mix() -> None:
    config = GenericSFTConfig()

    assert config.model_name == "Qwen/Qwen3.5-0.8B"
    assert config.dataset_name == "jayshah5696/humanize-rl-sft-dataset"
    assert config.data_files == DEFAULT_DATA_FILES
    assert config.enable_thinking is False
    assert config.push_to_hub is False
    assert config.max_steps <= 10


def test_messages_from_instruction_response_row() -> None:
    messages = messages_from_row({"instruction": "Write a note", "response": "Done."})

    assert messages == [
        {"role": "user", "content": "Write a note"},
        {"role": "assistant", "content": "Done."},
    ]


def test_messages_from_chat_row() -> None:
    row = {
        "messages": [
            {"role": "user", "content": "Write a note"},
            {"role": "assistant", "content": "Done."},
        ]
    }

    assert messages_from_row(row) == row["messages"]


def test_messages_from_row_rejects_missing_response() -> None:
    with pytest.raises(ValueError, match="missing instruction/response"):
        messages_from_row({"instruction": "Write a note", "response": ""})


def test_apply_chat_template_passes_qwen_thinking_flag() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.kwargs = None

        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN202
            self.kwargs = kwargs
            return "rendered"

    tokenizer = Tokenizer()
    rendered = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        enable_thinking=False,
    )

    assert rendered == "rendered"
    assert tokenizer.kwargs["enable_thinking"] is False
    assert tokenizer.kwargs["tokenize"] is False
    assert tokenizer.kwargs["add_generation_prompt"] is False


def test_apply_chat_template_falls_back_when_template_rejects_thinking_flag() -> None:
    class Tokenizer:
        def __init__(self) -> None:
            self.calls = []

        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN202
            self.calls.append(kwargs)
            if "enable_thinking" in kwargs:
                raise TypeError("unexpected keyword")
            return "fallback"

    tokenizer = Tokenizer()
    rendered = apply_chat_template(
        tokenizer,
        [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello"}],
        enable_thinking=False,
    )

    assert rendered == "fallback"
    assert "enable_thinking" in tokenizer.calls[0]
    assert "enable_thinking" not in tokenizer.calls[1]


def test_sft_config_kwargs_supports_trl_1_6_max_length() -> None:
    config = GenericSFTConfig(max_seq_length=1536)

    kwargs = sft_config_kwargs(
        config,
        {
            "dataset_text_field",
            "max_length",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "max_steps",
            "learning_rate",
        },
    )

    assert kwargs["max_length"] == 1536
    assert "max_seq_length" not in kwargs


def test_sft_config_kwargs_supports_older_max_seq_length() -> None:
    config = GenericSFTConfig(max_seq_length=1536)

    kwargs = sft_config_kwargs(config, {"dataset_text_field", "max_seq_length"})

    assert kwargs["max_seq_length"] == 1536
    assert "max_length" not in kwargs


def test_trainer_tokenizer_kwargs_supports_processing_class_and_tokenizer() -> None:
    tokenizer = object()

    assert trainer_tokenizer_kwargs(tokenizer, {"processing_class"}) == {
        "processing_class": tokenizer
    }
    assert trainer_tokenizer_kwargs(tokenizer, {"tokenizer"}) == {"tokenizer": tokenizer}
