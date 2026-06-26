"""Offline difficulty scoring \u2014 Slice 4 of the stable training plan.

Plan: docs/plans/gemma4_rl_modal_stable_training_continuation.md Slice 4.

For each task in the input mix, generate K completions from the SFT model
with the same prompt format used by GRPO, score each completion with the
production reward stack, then bucket the task as ``useful`` / ``too_easy``
/ ``too_hard`` / ``dead`` / ``clipped`` / ``same_pattern``.

Writes (under ``--output-dir``):
  * ``rollouts.jsonl``        \u2014 per-completion records
  * ``task_difficulty.jsonl`` \u2014 per-task summary + bucket
  * ``difficulty_summary.json`` \u2014 distribution + kept counts

And (when ``--filtered-output`` is given) the filtered mix, which is the
input mix with ``difficulty_bucket`` overwritten and a ``difficulty``
block added.

Local run (requires vLLM-capable env)::

    uvx modal run scripts/rl/score_task_difficulty_modal.py \\
      --task-path data/rl/humanize_tasks_rl_mix_v1.jsonl \\
      --output-dir outputs/rl_difficulty/mix_v1 \\
      --filtered-output data/rl/humanize_tasks_rl_mix_v1_filtered.jsonl \\
      --rollouts-per-task 8
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

try:
    import modal
except ModuleNotFoundError:  # pragma: no cover - local pytest path
    modal = None  # type: ignore[assignment]

GPU_TYPE = "A100-40GB"
TIMEOUT_HOURS = 2

if modal is not None:
    app = modal.App("humanize-rl-difficulty")
    model_cache_volume = modal.Volume.from_name(
        "humanize-rl-model-cache", create_if_missing=True
    )
    output_volume = modal.Volume.from_name(
        "humanize-rl-checkpoints", create_if_missing=True
    )
else:  # pragma: no cover

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
    output_volume = None  # type: ignore[assignment]


def _build_image() -> Any:
    if modal is None:  # pragma: no cover
        return None
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .uv_pip_install(
            "torch>=2.8.0",
            "torchvision",
            "datasets>=4.0.0",
            "hf-transfer>=0.1.9",
            "huggingface_hub>=0.34.0",
            "peft>=0.13.0",
            "pydantic>=2.0.0",
            "pyyaml>=6.0.0",
            "sentencepiece>=0.2.0",
            "tokenizers>=0.22.0,<=0.23.0",
            "transformers>=5.5.0",
            "vllm>=0.19.1,<0.21.0",
            "scikit-learn>=1.3.0",
            "fasttext-wheel>=0.9.2",
            "click>=8.1.0",
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
            "models/track_a_10k/ridge.pkl",
            remote_path="/workspace/models/track_a_10k/ridge.pkl",
        )
        .add_local_file(
            "models/distilled/baseline_ridge.pkl",
            remote_path="/workspace/models/distilled/baseline_ridge.pkl",
        )
    )


image = _build_image()


# ---------------------------------------------------------------------------
# Remote function
# ---------------------------------------------------------------------------


def _function_decorator() -> Any:
    if modal is None:  # pragma: no cover

        def passthrough(fn):  # type: ignore[no-untyped-def]
            return fn

        return passthrough
    return app.function(
        image=image,
        gpu=GPU_TYPE,
        volumes={
            "/model_cache": model_cache_volume,
            "/checkpoints": output_volume,
        },
        secrets=[modal.Secret.from_name("huggingface")],
        timeout=TIMEOUT_HOURS * 60 * 60,
        retries=modal.Retries(initial_delay=0.0, max_retries=0),
        single_use_containers=True,
    )


@_function_decorator()
def score_difficulty(
    task_path_remote: str,
    output_dir_remote: str,
    filtered_output_remote: str | None,
    model_name: str,
    rollouts_per_task: int,
    temperature: float,
    top_p: float,
    max_completion_length: int,
    max_tasks: int,
    seed: int,
    min_reward: float,
    max_reward: float,
    min_reward_std: float,
    max_clipped_rate: float,
) -> dict[str, Any]:
    import os
    import sys

    sys.path.insert(0, "/workspace/src")
    os.chdir("/workspace")

    # Bug G guard must run BEFORE the first vLLM import that builds
    # Gemma4Attention. Same patch used by
    # src/humanize_rl/training/rl_gemma4_trl_vllm_modal.py.
    from humanize_rl.training.rl_gemma4_trl_vllm_modal import (
        patch_vllm_gemma4_kv_shared_k_norm,
    )

    kv_shared_patched = patch_vllm_gemma4_kv_shared_k_norm()
    print(f"[vllm-gemma4-kv-shared-k-norm-patched] {kv_shared_patched}", flush=True)

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    from humanize_rl.reward.env import render_prompt
    from humanize_rl.reward.reward import load_ridge_scorer, score_response
    from humanize_rl.reward.tasks import load_tasks
    from humanize_rl.rl.difficulty import (
        DifficultyThresholds,
        Rollout,
        filter_task_mix,
        summarize_rollouts,
    )

    started = perf_counter()
    scorer = load_ridge_scorer()
    if scorer is None:
        raise RuntimeError("Ridge scorer pkl missing in image")

    # Processor download is left to vLLM; we only need the task loader and
    # the chat-template rendering that vLLM applies internally via the
    # registered tokenizer for ``model_name``.
    _ = AutoProcessor.from_pretrained(model_name, token=os.environ.get("HF_TOKEN"))

    tasks = load_tasks(Path(task_path_remote))
    if max_tasks > 0:
        tasks = tasks[:max_tasks]
    print(
        f"[difficulty] scoring {len(tasks)} tasks with K={rollouts_per_task}",
        flush=True,
    )

    # Render prompts once. The actual chat-template rendering is delegated
    # to vLLM's tokenizer/chat_template via ``chat`` API \u2014 keeps parity with
    # what TRL feeds the policy during training.
    prompts: list[list[dict[str, str]]] = []
    for task in tasks:
        prompts.append([{"role": "user", "content": render_prompt(task)}])

    llm = LLM(
        model=model_name,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=seed,
        enforce_eager=False,
    )
    sampling = SamplingParams(
        n=rollouts_per_task,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_completion_length,
        seed=seed,
    )

    gen_started = perf_counter()
    outputs = llm.chat(prompts, sampling, use_tqdm=True)
    gen_seconds = perf_counter() - gen_started
    print(f"[difficulty] generation took {gen_seconds:.1f}s", flush=True)

    thresholds = DifficultyThresholds(
        min_reward=min_reward,
        max_reward=max_reward,
        min_reward_std=min_reward_std,
        max_clipped_rate=max_clipped_rate,
    )

    output_dir = Path(output_dir_remote)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = output_dir / "rollouts.jsonl"
    difficulty_path = output_dir / "task_difficulty.jsonl"
    summary_path = output_dir / "difficulty_summary.json"

    difficulties = {}
    rollouts_fh = rollouts_path.open("w", encoding="utf-8")
    difficulty_fh = difficulty_path.open("w", encoding="utf-8")
    try:
        for task, output in zip(tasks, outputs, strict=True):
            rollouts: list[Rollout] = []
            for idx, completion in enumerate(output.outputs):
                response = completion.text.strip()
                clipped = completion.finish_reason == "length"
                result = score_response(task, response, scorer)
                ro = Rollout.from_result(response, result, clipped=clipped)
                rollouts.append(ro)
                rollouts_fh.write(
                    json.dumps(
                        {
                            "task_id": task.id,
                            "completion_idx": idx,
                            "response": response,
                            "response_length": ro.response_length,
                            "reward": ro.reward,
                            "ridge_rubric": ro.ridge_rubric,
                            "deterministic": ro.deterministic,
                            "penalty_sum": ro.penalty_sum,
                            "penalty_names": list(ro.penalty_names),
                            "clipped": clipped,
                        }
                    )
                    + "\n"
                )
            diff = summarize_rollouts(task.id, rollouts, thresholds)
            difficulties[task.id] = diff
            difficulty_fh.write(
                json.dumps(
                    {
                        "task_id": diff.task_id,
                        "family": task.family,
                        "reward_profile": task.reward_profile,
                        "n_completions": diff.n_completions,
                        "reward_mean": diff.reward_mean,
                        "reward_std": diff.reward_std,
                        "ridge_mean": diff.ridge_mean,
                        "ridge_std": diff.ridge_std,
                        "det_mean": diff.det_mean,
                        "det_std": diff.det_std,
                        "penalty_rate": diff.penalty_rate,
                        "penalty_pattern_diversity": diff.penalty_pattern_diversity,
                        "response_length_mean": diff.response_length_mean,
                        "response_length_p95": diff.response_length_p95,
                        "clipped_rate": diff.clipped_rate,
                        "bucket": diff.bucket,
                        "kept": diff.kept,
                        "reasons": list(diff.reasons),
                    }
                )
                + "\n"
            )
    finally:
        rollouts_fh.close()
        difficulty_fh.close()

    bucket_counts = Counter(d.bucket for d in difficulties.values())
    kept_counts = Counter(
        (d.bucket, "kept" if d.kept else "dropped") for d in difficulties.values()
    )

    if filtered_output_remote:
        mix_rows = [
            json.loads(line)
            for line in Path(task_path_remote).read_text().splitlines()
            if line.strip()
        ]
        filtered, drop_counts = filter_task_mix(mix_rows, difficulties)
        Path(filtered_output_remote).parent.mkdir(parents=True, exist_ok=True)
        with Path(filtered_output_remote).open("w", encoding="utf-8") as fh:
            for row in filtered:
                fh.write(json.dumps(row) + "\n")
        filter_stats = {"kept": len(filtered), "dropped": dict(drop_counts)}
    else:
        filter_stats = None

    summary = {
        "model_name": model_name,
        "n_tasks": len(tasks),
        "rollouts_per_task": rollouts_per_task,
        "thresholds": thresholds.__dict__,
        "bucket_counts": dict(bucket_counts),
        "kept_counts": {f"{k[0]}:{k[1]}": v for k, v in kept_counts.items()},
        "filter_stats": filter_stats,
        "generation_seconds": gen_seconds,
        "total_seconds": perf_counter() - started,
        "rollouts_path": str(rollouts_path),
        "difficulty_path": str(difficulty_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    output_volume.commit() if modal is not None and output_volume is not None else None
    print(json.dumps({"bucket_counts": summary["bucket_counts"]}, indent=2), flush=True)
    return summary


# ---------------------------------------------------------------------------
# Local entrypoint
# ---------------------------------------------------------------------------


if modal is not None:

    @app.local_entrypoint()
    def main(
        task_path: str = "data/rl/humanize_tasks_rl_mix_v1.jsonl",
        output_dir: str = "/checkpoints/rl_difficulty/mix_v1",
        filtered_output: str = "/checkpoints/rl_difficulty/mix_v1/humanize_tasks_rl_mix_v1_filtered.jsonl",
        model_name: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged",
        rollouts_per_task: int = 8,
        temperature: float = 1.0,
        top_p: float = 1.0,
        max_completion_length: int = 768,
        max_tasks: int = 0,
        seed: int = 3407,
        min_reward: float = 0.20,
        max_reward: float = 0.80,
        min_reward_std: float = 0.03,
        max_clipped_rate: float = 0.10,
    ) -> None:
        # The Modal image only has v01_smoke baked in by default. Stream the
        # actual task file by uploading to the volume via the script's CWD.
        # Easier path: assume the caller has already placed the file inside
        # the checkpoints volume at /checkpoints/<basename>.
        if not task_path.startswith("/"):
            uploaded = f"/checkpoints/{Path(task_path).name}"
            print(
                f"[difficulty] expecting {task_path} to already be uploaded to {uploaded} "
                "(use `modal volume put humanize-rl-checkpoints <local> /` first)"
            )
            task_path_remote = uploaded
        else:
            task_path_remote = task_path

        print(
            f"Launching difficulty scoring: tasks={task_path_remote}, K={rollouts_per_task}"
        )
        call = score_difficulty.spawn(
            task_path_remote=task_path_remote,
            output_dir_remote=output_dir,
            filtered_output_remote=filtered_output,
            model_name=model_name,
            rollouts_per_task=rollouts_per_task,
            temperature=temperature,
            top_p=top_p,
            max_completion_length=max_completion_length,
            max_tasks=max_tasks,
            seed=seed,
            min_reward=min_reward,
            max_reward=max_reward,
            min_reward_std=min_reward_std,
            max_clipped_rate=max_clipped_rate,
        )
        print(f"Spawned Modal call: {call.object_id}")
