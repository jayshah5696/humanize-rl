from __future__ import annotations

import json
from pathlib import Path

from scripts.data.build.generate_sft_references_from_failures import (
    build_reference_messages,
    build_sft_row,
    load_task_index,
)


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_task_index_maps_task_id(tmp_path: Path) -> None:
    task_path = tmp_path / "tasks.jsonl"
    _write_jsonl(
        task_path,
        [
            {
                "id": "rl_v03_000099",
                "instruction": "Rewrite this.",
                "input_text": "Source text",
                "split": "validation",
                "family": "rewrite_repair",
                "mode": "expansion",
            }
        ],
    )

    index = load_task_index([task_path])

    assert index["rl_v03_000099"]["mode"] == "expansion"


def test_build_reference_messages_include_failure_context() -> None:
    task = {
        "id": "rl_v03_000099",
        "instruction": "Explain the parity check program.",
        "input_text": "Byte A0 contains the value.",
        "constraints": {"max_words": 80, "return_only_answer": True},
        "required_facts": ["A0", "even parity"],
        "forbidden_facts": ["interrupts"],
    }
    failure = {
        "reasons": ["length_drift_ge_100_tokens"],
        "phrase_hits": ["hope you're doing well"],
        "base_completion_preview": "short answer",
        "rl_completion_preview": "very long answer",
    }

    messages = build_reference_messages(task, failure)
    content = messages[-1]["content"]

    assert "Explain the parity check program." in content
    assert "Byte A0 contains the value." in content
    assert "length_drift_ge_100_tokens" in content
    assert "hope you're doing well" in content
    assert "Return only the final answer" in content


def test_build_sft_row_has_instruction_response_and_metadata() -> None:
    task = {
        "id": "rl_v01_000016",
        "instruction": "Rewrite this note.",
        "input_text": "Formal source",
        "family": "rewrite_repair",
        "mode": "rewrite",
        "split": "validation",
    }
    failure = {
        "comparison": "p50_mix_v2_val20",
        "reasons": ["high_reward_humanizer_phrase"],
        "phrase_hits": ["so,"],
    }

    row = build_sft_row(task, failure, response="Clean target.")

    assert row["instruction"] == "Rewrite this note.\n\nSource:\nFormal source"
    assert row["response"] == "Clean target."
    assert row["metadata"]["task_id"] == "rl_v01_000016"
    assert row["metadata"]["failure_reasons"] == ["high_reward_humanizer_phrase"]
