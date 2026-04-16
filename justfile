# humanize-rl justfile

# Run tests
test *args:
    uv run pytest {{args}}

# Lint
lint:
    uv run ruff check .

# Format
format:
    uv run ruff format .

# Check (lint + format check)
check:
    uv run ruff check .
    uv run ruff format --check .

# Score text for humanness (Layer 1)
score text:
    uv run python -m humanize_rl.scoring.cli "{{text}}"
