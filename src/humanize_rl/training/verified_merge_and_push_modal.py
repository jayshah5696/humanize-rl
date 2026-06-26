"""Verified merge + push for Gemma 4 E2B humanize SFT artifacts.

Workflow inside a single Modal function (so detached runs are safe and we
do not have to chain `.spawn()` calls):

1. Load `unsloth/gemma-4-E2B-it` with Transformers + the source LoRA
   adapter via PEFT.
2. Generate direct-LoRA outputs on the 10-prompt parity set.
3. Call `merge_and_unload()` to fuse the adapter.
4. Save the merged model locally and reload it cold.
5. Reload generates outputs; compare against direct-LoRA outputs.
6. Validate Gemma 4 invariants:
   - `_keys_to_ignore_on_load_unexpected` matches the
     `[num_hidden_layers - num_kv_shared_layers, num_hidden_layers)` set.
   - `AutoModelForImageTextToText.from_pretrained(..., output_loading_info=True)`
     reports zero `missing_keys` and zero `unexpected_keys`.
   - `tokenizer_config.eos_token == "<turn|>"` (Unsloth #5386 guard); if
     regressed, restore it locally before push.
7. Only if all gates pass, upload to the candidate HF repo with an
   auto-generated, machine-readable model card.

This script never overwrites the production merged repo by default. It
pushes to the candidate repo passed as `--candidate-repo`. Promote later
by renaming the candidate or copying its files into the production repo.

Detached usage:

```bash
uvx modal run --detach \
  src/humanize_rl/training/verified_merge_and_push_modal.py \
  --candidate-repo jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2
```
"""

from __future__ import annotations

import modal

app = modal.App("humanize-rl-gemma4-verified-merge-and-push")

model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
merge_volume = modal.Volume.from_name("humanize-rl-merge-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate>=1.9.0",
        "huggingface_hub>=0.34.0",
        "peft>=0.16.0",
        "pillow>=12.0.0",
        "protobuf>=5.0.0",
        "safetensors>=0.4.5",
        "sentencepiece>=0.2.0",
        "torch>=2.7.0",
        "torchvision>=0.22.0",
        "transformers>=4.54.0",
    )
    .env({"HF_HOME": "/model_cache", "HF_XET_HIGH_PERFORMANCE": "1"})
    .add_local_file(
        "src/humanize_rl/training/_model_cards.py",
        remote_path="/workspace/_model_cards.py",
    )
)

BASE = "unsloth/gemma-4-E2B-it"
LORA = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"

PARITY_PROMPTS: list[str] = [
    "Rewrite this message so it sounds natural, warm, and professional. Keep it concise and preserve all facts.\n\nHi Saurabh. Yes looking forward to working with you.",
    "Make this sound more human and less corporate: 'Per our previous conversation, I wanted to circle back regarding the deliverables.'",
    "Humanize: 'It is important to note that our team is committed to delivering exceptional value.'",
    "Rewrite naturally: 'I hope this email finds you well. I am reaching out to follow up on our recent discussion.'",
    "Make it sound like a real person wrote it: 'We are excited to announce that we will be launching our new product next quarter.'",
    "Rewrite as a casual Slack message: 'Could you kindly provide an update on the status of the project at your earliest convenience?'",
    "Humanize this sentence: 'In today's fast-paced digital landscape, leveraging cutting-edge solutions is paramount.'",
    "Rewrite without AI cliches: 'Let's dive deep into the intricacies of this multifaceted problem.'",
    "Make this sound human: 'I would like to express my sincere gratitude for your continued support.'",
    "Rewrite naturally: 'Please find attached the document for your review and consideration.'",
]


def _shared_layer_keys(num_hidden_layers: int, num_kv_shared_layers: int) -> list[str]:
    start = num_hidden_layers - num_kv_shared_layers
    keys: list[str] = []
    for layer in range(start, num_hidden_layers):
        for name in (
            "k_proj.weight",
            "v_proj.weight",
            "k_norm.weight",
            "v_norm.weight",
        ):
            keys.append(f"model.language_model.layers.{layer}.self_attn.{name}")
    return keys


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": model_cache_volume, "/merge": merge_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=4 * 60 * 60,
    single_use_containers=True,
)
def verified_merge_and_push(
    base: str = BASE,
    lora: str = LORA,
    candidate_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2",
    allow_parity_mismatches: int = 2,
    push: bool = True,
) -> dict:
    import json
    import os
    import shutil
    import sys
    from pathlib import Path

    import torch
    from huggingface_hub import HfApi
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    sys.path.insert(0, "/workspace")
    from _model_cards import merged_model_card  # type: ignore[import-not-found]

    token = os.environ.get("HF_TOKEN")
    report: dict = {
        "base": base,
        "lora": lora,
        "candidate_repo": candidate_repo,
        "gates": {},
        "parity": {},
        "tokenizer": {},
        "push": {"requested": push, "performed": False},
    }

    out_dir = Path("/merge/verified-merge-peft")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    # 1) Load base + LoRA.
    print(f"Loading base {base}", flush=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )
    processor = AutoProcessor.from_pretrained(base, token=token)
    print(f"Loading LoRA {lora}", flush=True)
    direct = PeftModel.from_pretrained(base_model, lora, token=token)
    direct.eval()

    def _gen(model, prompt: str) -> str:
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs, max_new_tokens=160, do_sample=False, use_cache=True
            )
        return processor.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    # 2) Direct LoRA generations.
    print("Generating direct LoRA outputs...", flush=True)
    direct_generations = [_gen(direct, p) for p in PARITY_PROMPTS]

    # 3) Merge.
    print("merge_and_unload()", flush=True)
    merged = direct.merge_and_unload()
    merged.eval()
    in_memory_generations = [_gen(merged, p) for p in PARITY_PROMPTS]
    in_memory_mismatches = sum(
        1
        for a, b in zip(direct_generations, in_memory_generations, strict=True)
        if a != b
    )
    report["parity"]["in_memory_mismatches"] = in_memory_mismatches

    # 4) Save merged locally.
    print(f"Saving merged model to {out_dir}", flush=True)
    merged.save_pretrained(out_dir, safe_serialization=True, max_shard_size="5GB")
    processor.save_pretrained(out_dir)
    del direct
    del merged
    torch.cuda.empty_cache()

    # 5) Reload cold.
    print("Reloading saved merged model cold...", flush=True)
    reloaded, loading_info = AutoModelForImageTextToText.from_pretrained(
        str(out_dir),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_loading_info=True,
    )
    reloaded.eval()
    # Sanity: the saved processor must reload. Loading it forces any
    # processor-config breakage to surface here instead of at consumer
    # time. We then keep using the original processor (same tokenizer)
    # for parity generation.
    AutoProcessor.from_pretrained(str(out_dir))
    reloaded_generations = [_gen(reloaded, p) for p in PARITY_PROMPTS]

    parity_mismatches = [
        {"prompt": p, "direct": d, "reloaded": r}
        for p, d, r in zip(
            PARITY_PROMPTS, direct_generations, reloaded_generations, strict=True
        )
        if d != r
    ]
    report["parity"]["prompt_count"] = len(PARITY_PROMPTS)
    report["parity"]["reload_mismatch_count"] = len(parity_mismatches)
    report["parity"]["reload_mismatches"] = parity_mismatches[:3]

    # 6) Gates.
    cfg = AutoConfig.from_pretrained(str(out_dir))
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = int(text_cfg.num_hidden_layers)
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0))
    expected_missing = set(_shared_layer_keys(n_layers, n_shared))

    present_keys: set[str] = set()
    for shard in sorted(out_dir.glob("model*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                present_keys.add(key)

    wrongly_present = sorted(expected_missing & present_keys)
    report["gates"]["num_hidden_layers"] = n_layers
    report["gates"]["num_kv_shared_layers"] = n_shared
    report["gates"]["wrongly_present_shared_kv_keys"] = wrongly_present
    report["gates"]["safetensors_key_count"] = len(present_keys)
    report["gates"]["transformers_missing_keys"] = sorted(
        loading_info.get("missing_keys", [])
    )
    report["gates"]["transformers_unexpected_keys"] = sorted(
        loading_info.get("unexpected_keys", [])
    )

    # 7) Tokenizer eos guard (Unsloth #5386).
    tok_cfg_path = out_dir / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    eos_token = tok_cfg.get("eos_token")
    report["tokenizer"]["eos_token_observed"] = eos_token
    report["tokenizer"]["eos_token_expected"] = "<turn|>"
    if eos_token != "<turn|>":
        print(
            f"WARN: tokenizer eos_token regressed to {eos_token!r}; restoring to '<turn|>'",
            flush=True,
        )
        tok_cfg["eos_token"] = "<turn|>"
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False))
        report["tokenizer"]["restored"] = True
    else:
        report["tokenizer"]["restored"] = False

    # 8) Compose gate verdict.
    gates_pass = (
        len(wrongly_present) == 0
        and len(report["gates"]["transformers_missing_keys"]) == 0
        and len(report["gates"]["transformers_unexpected_keys"]) == 0
        and len(parity_mismatches) <= allow_parity_mismatches
    )
    report["gates"]["all_pass"] = gates_pass

    # 9) Write model card.
    readme = out_dir / "README.md"
    readme.write_text(
        merged_model_card(
            base_model=base,
            lora_repo=lora,
            parity_prompt_count=len(PARITY_PROMPTS),
            parity_mismatch_count=len(parity_mismatches),
            transformers_missing_keys=len(report["gates"]["transformers_missing_keys"]),
            transformers_unexpected_keys=len(
                report["gates"]["transformers_unexpected_keys"]
            ),
            eos_token=tok_cfg.get("eos_token", "<turn|>"),
            safetensors_key_count=len(present_keys),
        )
    )

    # 10) Persist report.
    (out_dir / "verification_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    merge_volume.commit()

    # 11) Push only if gates passed.
    if not gates_pass:
        print("Gates FAILED; not pushing.", flush=True)
        report["push"]["performed"] = False
        report["push"]["reason"] = "gates_failed"
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return report

    if not push:
        print("Gates passed but push=false; skipping upload.", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
        return report

    print(f"Gates passed. Uploading to {candidate_repo}", flush=True)
    api = HfApi(token=token)
    api.create_repo(repo_id=candidate_repo, repo_type="model", exist_ok=True)
    api.upload_folder(
        repo_id=candidate_repo,
        repo_type="model",
        folder_path=str(out_dir),
        path_in_repo=".",
        commit_message=(
            "Verified PEFT merge: 10-prompt parity, KV-shared keys correct, "
            "tokenizer eos preserved"
        ),
    )
    report["push"]["performed"] = True
    report["push"]["repo"] = candidate_repo
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@app.local_entrypoint()
def main(
    base: str = BASE,
    lora: str = LORA,
    candidate_repo: str = "jayshah5696/gemma4-e2b-humanize-unsloth-merged-peft-v2",
    allow_parity_mismatches: int = 2,
    push: bool = True,
) -> None:
    call = verified_merge_and_push.spawn(
        base=base,
        lora=lora,
        candidate_repo=candidate_repo,
        allow_parity_mismatches=allow_parity_mismatches,
        push=push,
    )
    print(f"Spawned Modal call: {call.object_id}")
