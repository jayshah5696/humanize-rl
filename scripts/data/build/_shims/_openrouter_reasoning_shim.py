"""Sitecustomize-style shim that injects OpenRouter `extra_body` into every
`OpenAI.chat.completions.create` call (and `.parse`).

Activated by setting `HUMANIZE_OPENROUTER_EXTRA_BODY` to a JSON string before
running arka. Idempotent.

NOTE: An earlier version also patched `arka.pipeline.generator_stages.
TransformGeneratorStage.run` to swallow per-row errors, but Gemini 3.1 Pro on
OpenRouter currently ignores `reasoning.max_tokens` overrides anyway, so we
dropped that path for slice 4. Revisit if a usable reasoning cap exists.
"""
from __future__ import annotations

import json
import os
from typing import Any

_FLAG = "_humanize_openrouter_patched"


def _merge_extra(kwargs: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    existing = dict(kwargs.get("extra_body") or {})
    for k, v in extra.items():
        if k == "reasoning" and isinstance(existing.get(k), dict) and isinstance(v, dict):
            merged = dict(existing[k])
            merged.update(v)
            existing[k] = merged
        else:
            existing[k] = v
    kwargs["extra_body"] = existing
    return kwargs


def install() -> None:
    raw = os.environ.get("HUMANIZE_OPENROUTER_EXTRA_BODY")
    if not raw:
        return
    try:
        extra = json.loads(raw)
    except Exception:  # noqa: BLE001
        return
    if not isinstance(extra, dict) or not extra:
        return

    try:
        from openai.resources.chat import completions as _comp_mod
    except ImportError:
        return

    Completions = getattr(_comp_mod, "Completions", None)
    if Completions is None or getattr(Completions, _FLAG, False):
        return

    orig_create = Completions.create

    def patched_create(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        kwargs = _merge_extra(kwargs, extra)
        return orig_create(self, *args, **kwargs)

    Completions.create = patched_create  # type: ignore[assignment]
    setattr(Completions, _FLAG, True)

    try:
        from openai.resources.beta.chat import completions as _beta_comp_mod
        BetaCompletions = getattr(_beta_comp_mod, "Completions", None)
        if BetaCompletions is not None and not getattr(BetaCompletions, _FLAG, False):
            orig_parse = BetaCompletions.parse

            def patched_parse(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                kwargs = _merge_extra(kwargs, extra)
                return orig_parse(self, *args, **kwargs)

            BetaCompletions.parse = patched_parse  # type: ignore[assignment]
            setattr(BetaCompletions, _FLAG, True)
    except ImportError:
        pass


install()
