"""Auto-loaded by Python at startup when this directory is on sys.path.

Activates `_openrouter_reasoning_shim.install()` which patches
`OpenAI.chat.completions.create/.parse` to inject `extra_body` (e.g.
`reasoning.exclude=true`) for OpenRouter calls.

Driven by env var `HUMANIZE_OPENROUTER_EXTRA_BODY` (JSON string).
"""
from _openrouter_reasoning_shim import install

install()
