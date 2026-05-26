#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import pickle
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from humanize_rl.scoring.aggregator import score_text

AI_TELL_PHRASES = (
    "certainly",
    "of course",
    "great question",
    "i'd be happy to",
    "it is worth noting",
    "it's worth noting",
    "furthermore",
    "moreover",
    "in conclusion",
    "please don't hesitate",
    "i hope this email finds you well",
    "here are a few options",
)


@dataclass(frozen=True)
class EvalConfig:
    prompts_path: Path
    output_dir: Path
    base_model: str
    adapter_path: Path
    scorer_path: Path
    max_tokens: int
    temperature: float


def default_config() -> EvalConfig:
    return EvalConfig(
        prompts_path=Path(os.environ.get("PROMPTS_PATH", "data/eval/gemma4_humanize_eval_prompts_v01.jsonl")),
        output_dir=Path(os.environ.get("OUTPUT_DIR", "outputs_gemma4_humanize_mlx_full_r8/eval_v01")),
        base_model=os.environ.get("BASE_MODEL", "mlx-community/gemma-4-e2b-it-4bit"),
        adapter_path=Path(os.environ.get("ADAPTER_PATH", "adapters/gemma4_e2b_v04_mlx_full_r8")),
        scorer_path=Path(os.environ.get("SCORER_PATH", "models/track_a_10k/ridge.pkl")),
        max_tokens=int(os.environ.get("MAX_TOKENS", "180")),
        temperature=float(os.environ.get("TEMPERATURE", "0.3")),
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def prepare_adapter_for_mlx_vlm(adapter_path: Path) -> Path:
    mlx_vlm_config = adapter_path / "adapter_config.json"
    if mlx_vlm_config.exists():
        data = json.loads(mlx_vlm_config.read_text())
        if "rank" in data:
            return adapter_path

    temp_dir = Path(tempfile.mkdtemp(prefix="gemma4_eval_adapter_"))
    shutil.copy(adapter_path / "adapters.safetensors", temp_dir / "adapters.safetensors")
    if (adapter_path / "config.json").exists():
        shutil.copy(adapter_path / "config.json", temp_dir / "config.json")

    source_config = Path("outputs_gemma4_humanize_mlx_full_r8/adapters/adapter_config.json")
    if source_config.exists():
        shutil.copy(source_config, temp_dir / "adapter_config.json")
    else:
        (temp_dir / "adapter_config.json").write_text(json.dumps({"rank": 8, "alpha": 1.0, "dropout": 0.0}, indent=2))
    return temp_dir


def parse_generation(stdout: str) -> str:
    if "<|turn>model" in stdout:
        after = stdout.split("<|turn>model", 1)[1]
    elif "Prompt:" in stdout:
        after = stdout.split("Prompt:", 1)[1]
    else:
        after = stdout
    if "==========" in after:
        after = after.split("==========", 1)[0]
    lines = []
    for line in after.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("Prompt:") or stripped.startswith("Generation:") or stripped.startswith("Peak memory:"):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines).strip()


def generate(prompt: str, model: str, adapter_path: Path | None, config: EvalConfig) -> str:
    command = [
        "uvx",
        "--from",
        "mlx-vlm",
        "--python",
        "3.12",
        "mlx_vlm.generate",
        "--model",
        model,
        "--prompt",
        prompt,
        "--max-tokens",
        str(config.max_tokens),
        "--temperature",
        str(config.temperature),
        "--skip-special-tokens",
    ]
    if adapter_path is not None:
        command.extend(["--adapter-path", str(adapter_path)])
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        return f"GENERATION_ERROR: {result.stderr.strip()}"
    return parse_generation(result.stdout)


def load_scorer(path: Path) -> Any | None:
    if not path.exists():
        return None
    with path.open("rb") as handle:
        return pickle.load(handle)


def ai_tell_count(text: str) -> int:
    lower = text.lower()
    return sum(1 for phrase in AI_TELL_PHRASES if phrase in lower)


def option_style_penalty(text: str) -> bool:
    lower = text.lower()
    return "option 1" in lower or "here are a few options" in lower


def score_rows(rows: list[dict[str, Any]], scorer: Any | None) -> list[dict[str, Any]]:
    texts = [row["response"] for row in rows]
    ai_probs = scorer.predict_binary(texts) if scorer is not None else [None] * len(rows)
    rubrics = scorer.predict_rubric(texts) if scorer is not None else [None] * len(rows)

    scored = []
    for row, ai_probability, rubric in zip(rows, ai_probs, rubrics, strict=True):
        l1 = score_text(row["response"])
        rubric_values = [] if rubric is None else [float(value) for value in rubric]
        scored.append(
            {
                **row,
                "scores": {
                    "layer1_overall": l1.overall,
                    "layer1_per_dim": l1.per_dim,
                    "distilled_ai_probability": None if ai_probability is None else float(ai_probability),
                    "distilled_rubric_mean": None if not rubric_values else sum(rubric_values) / len(rubric_values),
                    "distilled_rubric_values": rubric_values,
                    "ai_tell_count": ai_tell_count(row["response"]),
                    "option_style_penalty": option_style_penalty(row["response"]),
                    "word_count": len(row["response"].split()),
                },
            }
        )
    return scored


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    def mean(values: list[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    return {
        "count": len(scored),
        "layer1_mean": mean([row["scores"]["layer1_overall"] for row in scored]),
        "distilled_ai_probability_mean": mean([row["scores"]["distilled_ai_probability"] for row in scored if row["scores"]["distilled_ai_probability"] is not None]),
        "distilled_rubric_mean": mean([row["scores"]["distilled_rubric_mean"] for row in scored if row["scores"]["distilled_rubric_mean"] is not None]),
        "ai_tell_total": sum(row["scores"]["ai_tell_count"] for row in scored),
        "option_style_count": sum(1 for row in scored if row["scores"]["option_style_penalty"]),
        "word_count_mean": mean([row["scores"]["word_count"] for row in scored]),
    }


def render_report(base: list[dict[str, Any]], finetuned: list[dict[str, Any]], summary: dict[str, Any]) -> str:
    lines = [
        "# Gemma 4 E2B Base vs Local SFT Eval v01",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
        "## Per-prompt comparison",
        "",
    ]
    by_id_base = {row["id"]: row for row in base}
    by_id_ft = {row["id"]: row for row in finetuned}
    for prompt_id, base_row in by_id_base.items():
        ft_row = by_id_ft[prompt_id]
        lines.extend(
            [
                f"### {prompt_id} — {base_row['category']}",
                "",
                "**Instruction**",
                "",
                base_row["instruction"],
                "",
                "**Base output**",
                "",
                base_row["response"],
                "",
                "**Fine-tuned output**",
                "",
                ft_row["response"],
                "",
                "**Scores**",
                "",
                "```json",
                json.dumps({"base": base_row["scores"], "fine_tuned": ft_row["scores"]}, indent=2),
                "```",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> None:
    config = default_config()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_jsonl(config.prompts_path)
    scorer = load_scorer(config.scorer_path)
    eval_adapter = prepare_adapter_for_mlx_vlm(config.adapter_path)

    raw_rows: list[dict[str, Any]] = []
    for prompt in prompts:
        print(f"Generating {prompt['id']} base", flush=True)
        base_response = generate(prompt["instruction"], config.base_model, None, config)
        print(f"Generating {prompt['id']} fine-tuned", flush=True)
        ft_response = generate(prompt["instruction"], config.base_model, eval_adapter, config)
        raw_rows.append({**prompt, "model_variant": "base", "response": base_response})
        raw_rows.append({**prompt, "model_variant": "fine_tuned", "response": ft_response})

    base_scored = score_rows([row for row in raw_rows if row["model_variant"] == "base"], scorer)
    ft_scored = score_rows([row for row in raw_rows if row["model_variant"] == "fine_tuned"], scorer)
    summary = {"base": summarize(base_scored), "fine_tuned": summarize(ft_scored)}

    write_jsonl(config.output_dir / "generations_scored.jsonl", base_scored + ft_scored)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (config.output_dir / "report.md").write_text(render_report(base_scored, ft_scored, summary))
    print(json.dumps(summary, indent=2))
    print(f"Report: {config.output_dir / 'report.md'}")


if __name__ == "__main__":
    main()
