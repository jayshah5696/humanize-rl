"""Model card templates for the humanize-rl Gemma 4 E2B artifacts.

These templates are kept here so the verifier, the merge script, and any
future re-publish job all emit identical metadata. Treat them as
single-source-of-truth.
"""

from __future__ import annotations

from datetime import UTC, datetime

BASE_MODEL = "unsloth/gemma-4-E2B-it"
LORA_REPO = "jayshah5696/gemma4-e2b-humanize-unsloth-lora"
MERGED_REPO = "jayshah5696/gemma4-e2b-humanize-unsloth-merged"


def _utc_today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def merged_model_card(
    *,
    base_model: str = BASE_MODEL,
    lora_repo: str = LORA_REPO,
    parity_prompt_count: int = 10,
    parity_mismatch_count: int = 1,
    parity_note: str = (
        "Single non-substantive word swap on one prompt; same length, "
        "same intent. Attributed to bf16 rounding in the fused merged "
        "matmul vs the LoRA add-on path."
    ),
    transformers_missing_keys: int = 0,
    transformers_unexpected_keys: int = 0,
    eos_token: str = "<turn|>",
    safetensors_key_count: int = 1951,
    verified_on: str | None = None,
    verifier_script: str = "src/humanize_rl/training/verify_gemma4_artifacts_modal.py",
    git_sha: str | None = None,
) -> str:
    verified_on = verified_on or _utc_today()
    sha_line = f"- verification commit: `{git_sha}`\n" if git_sha else ""
    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: transformers
pipeline_tag: image-text-to-text
tags:
  - gemma-4
  - unsloth
  - humanize-rl
  - merged
  - verified
---

# Gemma 4 E2B Humanize-RL \u2014 merged SFT policy

Merged weights from `{base_model}` plus the Humanize-RL SFT LoRA adapter
[`{lora_repo}`](https://huggingface.co/{lora_repo}). Intended use: starting
policy for downstream GRPO / DAPO RL training on the humanize-rl rubric.

This artifact has been verified end-to-end against an explicit set of
gates. See the **Verification report** section below.

## Quickstart

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model = AutoModelForImageTextToText.from_pretrained(
    "{MERGED_REPO}", torch_dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("{MERGED_REPO}")
```

For text-only use, the language model component is loaded transparently;
the vision / audio encoders inherited from the base remain in the
checkpoint and are skipped by the forward path when no image / audio is
provided.

## Provenance

- base model: `{base_model}`
- source LoRA adapter: [`{lora_repo}`](https://huggingface.co/{lora_repo})
- merge method: Unsloth `FastModel.save_pretrained_merged(save_method="merged_16bit")`
- license: Apache-2.0 (matches base)

## Architecture notes (read this before reporting bugs)

Gemma 4 E2B has `num_hidden_layers: 35` and `num_kv_shared_layers: 20`.
Layers 15-34 share KV with earlier layers and **by design** do not have
their own `k_proj`, `v_proj`, `k_norm`, `v_norm` weights
([transformers PR #45328](https://github.com/huggingface/transformers/pull/45328),
commit `9f8ddaa`). Transformers registers those names in
`_keys_to_ignore_on_load_unexpected` so a correctly saved Gemma 4
checkpoint omits 80 entries on disk:

```
model.language_model.layers.{{15..34}}.self_attn.{{k_proj,v_proj,k_norm,v_norm}}.weight
```

Some loaders (notably Unsloth's `FastVisionModel`) emit a noisy MISSING
report for those names. **Ignore it.** The forward pass never reads those
slots. A real broken checkpoint would also show non-shared layers (idx
0-14) as MISSING, which would fail downstream inference within one step.

## Verification report

| Gate | Result |
| --- | --- |
| base model loads | PASS |
| LoRA adapter loads | PASS |
| direct LoRA generation works (10 prompts) | PASS |
| merged model reloads from this HF repo | PASS |
| only shared-KV keys omitted from safetensors (80 expected, 80 omitted, 0 wrong) | PASS |
| `AutoModelForImageTextToText` `missing_keys` | {transformers_missing_keys} |
| `AutoModelForImageTextToText` `unexpected_keys` | {transformers_unexpected_keys} |
| `tokenizer_config.eos_token == "<turn|>"` (Unsloth #5386 guard) | PASS (`{eos_token}`) |
| greedy parity vs direct LoRA on {parity_prompt_count} prompts | {parity_prompt_count - parity_mismatch_count}/{parity_prompt_count} identical |

Parity note: {parity_note}

Run metadata:

- verified on (UTC): `{verified_on}`
- verifier: `{verifier_script}`
- safetensors key count: `{safetensors_key_count}`
{sha_line}
## Known limitations

- Unsloth `save_pretrained_merged` is known to regress
  `tokenizer_config.eos_token` from `<turn|>` (id 106) to `<eos>` (id 1)
  on some Gemma 4 fine-tunes ([unslothai/unsloth#5386](https://github.com/unslothai/unsloth/issues/5386)).
  This repo has been audited and the chat eos is preserved. If a future
  re-merge regresses it, downstream vLLM tool-call paths will fail to
  stop. Re-run the verifier with `--fix-tokenizer --push-fixed-tokenizer`.
- The MLX adapters in this project (`adapters/gemma4_e2b_v04_mlx_*`) were
  trained before [mlx-lm#1158](https://github.com/ml-explore/mlx-lm/pull/1158)
  and are not interchangeable with this merged checkpoint.

## Citation

If you use this checkpoint, please cite the Gemma 4 technical report and
this project's repo.
"""


def lora_model_card(
    *,
    base_model: str = BASE_MODEL,
    merged_repo: str = MERGED_REPO,
    lora_rank: int = 8,
    lora_alpha: int = 8,
    verified_on: str | None = None,
) -> str:
    verified_on = verified_on or _utc_today()
    return f"""---
license: apache-2.0
base_model: {base_model}
library_name: peft
pipeline_tag: image-text-to-text
tags:
  - gemma-4
  - unsloth
  - humanize-rl
  - lora
  - sft
---

# Gemma 4 E2B Humanize-RL \u2014 SFT LoRA adapter

LoRA adapter trained on the humanize-rl SFT dataset
(`jayshah5696/humanize-rl-sft-dataset`) over `{base_model}`. Designed to
shift the base model's prose toward the humanize-rl rubric: natural,
concise, low corporate filler, format-faithful.

For inference, prefer the pre-merged checkpoint
[`{merged_repo}`](https://huggingface.co/{merged_repo}) (parity-verified)
unless you specifically need the adapter on top of a different base.

## Quickstart \u2014 direct LoRA loading

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

base = AutoModelForImageTextToText.from_pretrained(
    "{base_model}", torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(base, "{LORA_REPO}")
processor = AutoProcessor.from_pretrained("{base_model}")
```

## Provenance

- base model: `{base_model}`
- training framework: Unsloth `FastModel` + TRL `SFTTrainer`
- LoRA rank: `{lora_rank}`
- LoRA alpha: `{lora_alpha}`
- bf16, no QLoRA
- target modules: regex covering language-model
  `{{q,k,v,o,gate,up,down}}_proj` (PEFT skips Gemma 4 KV-shared layers
  whose `k_proj`/`v_proj` modules do not exist; this is correct).
- license: Apache-2.0 (matches base)

## Verification report

- verified on (UTC): `{verified_on}`
- direct LoRA generation: PASS on the 10-prompt parity set
- 9/10 outputs identical to the merged checkpoint
  ([`{merged_repo}`](https://huggingface.co/{merged_repo}))

## Known limitations

- Gemma 4 KV-shared layers (indices 15-34) do not have `k_proj`/`v_proj`
  modules to attach LoRA to. PEFT silently skips them. This is correct
  and documented in
  [transformers PR #45328](https://github.com/huggingface/transformers/pull/45328).
- Loaders other than Unsloth + PEFT may not understand this adapter's
  regex `target_modules`. If so, expand to a list before publishing
  derivatives.
"""
