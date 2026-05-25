"""Verify Gemma 4 E2B SFT merge artifacts and tokenizer config.

Background
----------
Gemma 4 E2B has 35 text decoder layers. The config sets
``num_kv_shared_layers: 20``, so layers 15-34 share KV with earlier layers
and intentionally do NOT have their own ``k_proj``/``v_proj``/``k_norm``/
``v_norm`` weights. See ``Gemma4TextAttention.__init__`` in transformers
(PR #45328, commit 9f8ddaa). Transformers registers those keys in
``_keys_to_ignore_on_load_unexpected`` so a correctly saved checkpoint must
omit them. The MISSING-keys report from Unsloth's ``FastVisionModel`` loader
is a false alarm because the loader prints layers that the forward pass
never reads.

This script verifies:

1. The merged HF repo's safetensors deliberately omit shared-layer K/V
   weights for layers ``[num_hidden_layers - num_kv_shared_layers,
   num_hidden_layers)``.
2. The base, LoRA, and merged artifacts produce identical greedy outputs on
   a parity prompt set.
3. ``tokenizer_config.json::eos_token`` is preserved as the chat ``<turn|>``
   token (Unsloth issue #5386), not regressed to raw ``<eos>``.
4. ``Gemma4ForConditionalGeneration`` reports no missing keys on reload.

All Modal entry points use ``.spawn()`` so they are safe to run with
``modal run --detach``.
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

app = modal.App("humanize-rl-gemma4-verify-artifacts")

model_cache_volume = modal.Volume.from_name(
    "humanize-rl-model-cache", create_if_missing=True
)
merge_volume = modal.Volume.from_name("humanize-rl-merge-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
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
)

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

BASE = "unsloth/gemma-4-E2B-it"
LORA = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"
MERGED = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"


def _shared_layer_indices(num_hidden_layers: int, num_kv_shared_layers: int) -> list[int]:
    """Return the list of layer indices that share KV with earlier layers."""
    start = num_hidden_layers - num_kv_shared_layers
    return list(range(start, num_hidden_layers))


def _shared_layer_keys(num_hidden_layers: int, num_kv_shared_layers: int) -> list[str]:
    keys: list[str] = []
    for layer in _shared_layer_indices(num_hidden_layers, num_kv_shared_layers):
        for name in ("k_proj.weight", "v_proj.weight", "k_norm.weight", "v_norm.weight"):
            keys.append(f"model.language_model.layers.{layer}.self_attn.{name}")
    return keys


@app.function(
    image=image,
    gpu="L40S",
    volumes={"/model_cache": model_cache_volume, "/merge": merge_volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=2 * 60 * 60,
    single_use_containers=True,
)
def verify(
    base: str = BASE,
    lora: str = LORA,
    merged: str = MERGED,
    fix_tokenizer: bool = False,
    push_fixed_tokenizer: bool = False,
) -> dict:
    """Run full Gemma 4 SFT artifact verification.

    Parameters
    ----------
    fix_tokenizer:
        If True and the merged repo's tokenizer eos_token regressed to
        ``<eos>``, write a corrected ``tokenizer_config.json`` locally and
        optionally push (controlled by ``push_fixed_tokenizer``).
    """
    import os
    import shutil
    import tempfile

    import torch
    from huggingface_hub import HfApi, snapshot_download
    from peft import PeftModel
    from safetensors import safe_open
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
    )

    token = os.environ.get("HF_TOKEN")
    report: dict = {
        "base": base,
        "lora": lora,
        "merged": merged,
        "checks": {},
        "parity": {},
        "tokenizer": {},
    }

    # 1) Pull merged snapshot for inspection.
    merged_local = Path(tempfile.mkdtemp(prefix="merged-")) 
    snapshot_download(
        repo_id=merged,
        local_dir=str(merged_local),
        token=token,
        allow_patterns=[
            "config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "tokenizer.model",
            "generation_config.json",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
            "preprocessor_config.json",
            "processor_config.json",
            "chat_template.jinja",
            "special_tokens_map.json",
            "added_tokens.json",
        ],
    )

    cfg = AutoConfig.from_pretrained(str(merged_local))
    text_cfg = getattr(cfg, "text_config", cfg)
    n_layers = int(text_cfg.num_hidden_layers)
    n_shared = int(getattr(text_cfg, "num_kv_shared_layers", 0))
    expected_missing = set(_shared_layer_keys(n_layers, n_shared))
    report["checks"]["num_hidden_layers"] = n_layers
    report["checks"]["num_kv_shared_layers"] = n_shared
    report["checks"]["expected_shared_layer_keys_omitted"] = sorted(expected_missing)

    # 2) Confirm those keys are absent from saved safetensors.
    present_keys: set[str] = set()
    for shard in sorted(merged_local.glob("model*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for key in handle.keys():
                present_keys.add(key)
    wrongly_present = sorted(expected_missing & present_keys)
    report["checks"]["wrongly_present_shared_kv_keys"] = wrongly_present
    report["checks"]["safetensors_key_count"] = len(present_keys)

    # 3) Tokenizer eos_token regression check (Unsloth issue #5386).
    tok_cfg_path = merged_local / "tokenizer_config.json"
    tok_cfg = json.loads(tok_cfg_path.read_text())
    eos_token = tok_cfg.get("eos_token")
    report["tokenizer"]["eos_token"] = eos_token
    expected_eos = "<turn|>"
    report["tokenizer"]["expected_eos_token"] = expected_eos
    regressed = eos_token != expected_eos
    report["tokenizer"]["eos_token_regressed"] = regressed
    if regressed and fix_tokenizer:
        tok_cfg["eos_token"] = expected_eos
        tok_cfg_path.write_text(json.dumps(tok_cfg, indent=2, ensure_ascii=False))
        report["tokenizer"]["fixed_locally"] = True
        if push_fixed_tokenizer:
            api = HfApi(token=token)
            api.upload_file(
                path_or_fileobj=str(tok_cfg_path),
                path_in_repo="tokenizer_config.json",
                repo_id=merged,
                commit_message="Restore eos_token to <turn|> (Unsloth save regression)",
            )
            report["tokenizer"]["pushed"] = True

    # 4) Reload merged with Transformers and capture loading info.
    print("Reloading merged model with AutoModelForImageTextToText...", flush=True)
    reload_target, reload_info = AutoModelForImageTextToText.from_pretrained(
        str(merged_local),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_loading_info=True,
    )
    reload_target.eval()
    report["checks"]["transformers_missing_keys"] = sorted(reload_info.get("missing_keys", []))
    report["checks"]["transformers_unexpected_keys"] = sorted(
        reload_info.get("unexpected_keys", [])
    )

    # The Unsloth merged dir typically ships text-only tokenizer files and
    # omits image processor configs. Try local, then fall back to base for
    # processor assets (the language model + tokenizer are what we care
    # about for parity).
    try:
        processor_merged = AutoProcessor.from_pretrained(str(merged_local))
        report["checks"]["processor_source"] = "merged_local"
    except (OSError, ValueError) as exc:
        print(f"Merged processor missing image components ({exc}); using base.", flush=True)
        processor_merged = AutoProcessor.from_pretrained(base, token=token)
        report["checks"]["processor_source"] = "base_fallback"

    def _generate(model, processor_obj, prompt: str) -> str:
        inputs = processor_obj.apply_chat_template(
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
        return processor_obj.decode(
            out[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
        ).strip()

    merged_generations = [_generate(reload_target, processor_merged, p) for p in PARITY_PROMPTS]
    del reload_target
    torch.cuda.empty_cache()

    # 5) Direct LoRA generations for parity.
    print("Loading base + LoRA for direct generation...", flush=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", token=token
    )
    base_processor = AutoProcessor.from_pretrained(base, token=token)
    direct = PeftModel.from_pretrained(base_model, lora, token=token)
    direct.eval()
    direct_generations = [_generate(direct, base_processor, p) for p in PARITY_PROMPTS]

    mismatches = [
        {"prompt": p, "direct": d, "merged": m}
        for p, d, m in zip(PARITY_PROMPTS, direct_generations, merged_generations, strict=True)
        if d != m
    ]
    report["parity"]["prompt_count"] = len(PARITY_PROMPTS)
    report["parity"]["mismatch_count"] = len(mismatches)
    report["parity"]["mismatches"] = mismatches[:3]  # first few only for brevity

    out_dir = Path("/merge/verify-gemma4")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    (out_dir / "direct_generations.jsonl").write_text(
        "\n".join(
            json.dumps({"prompt": p, "generation": g})
            for p, g in zip(PARITY_PROMPTS, direct_generations, strict=True)
        )
    )
    (out_dir / "merged_generations.jsonl").write_text(
        "\n".join(
            json.dumps({"prompt": p, "generation": g})
            for p, g in zip(PARITY_PROMPTS, merged_generations, strict=True)
        )
    )
    merge_volume.commit()
    shutil.rmtree(merged_local, ignore_errors=True)

    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    return report


@app.local_entrypoint()
def main(
    base: str = BASE,
    lora: str = LORA,
    merged: str = MERGED,
    fix_tokenizer: bool = False,
    push_fixed_tokenizer: bool = False,
) -> None:
    call = verify.spawn(
        base=base,
        lora=lora,
        merged=merged,
        fix_tokenizer=fix_tokenizer,
        push_fixed_tokenizer=push_fixed_tokenizer,
    )
    print(f"Spawned Modal call: {call.object_id}")
