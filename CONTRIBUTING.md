# Contributing

## Development setup

This project uses `uv` for Python dependency management and `just` for common tasks.

```bash
uv sync
just test
just lint
```

## Arka dependency

`humanize-rl` depends on [`arka`](https://github.com/jayshah5696/arka).

By default, this repository resolves `arka` from GitHub via `uv` using the pinned source in `pyproject.toml`. This keeps fresh clones and CI reproducible.

If you are developing `arka` locally at the same time, you can temporarily switch `tool.uv.sources.arka` to a local editable path in your own branch.

## Common tasks

```bash
just test          # run pytest
just lint          # run ruff check
just format        # run ruff format
just check         # lint + format check
just score "..."   # Layer 1 score inline text
just benchmark     # run Layer 1 benchmark
just aiify         # run AIify Arka pipeline
just humanize      # run humanize Arka pipeline
just pipeline      # score/export pipeline outputs
just score-all     # run Layer 1 + Layer 2 scoring
```

## Environment variables

Layer 2 scoring and Arka pipeline runs require:

```bash
export OPENROUTER_API_KEY=...
```

## Generated artifacts

The following paths are generated and are usually not committed unless they are intentional benchmark artifacts:

- `output/`
- `data/processed/`
- `runs/`
- local caches and virtualenv files

## Quality bar

Before committing:

```bash
just test
just lint
```

Prefer small, reversible changes. Follow `AGENTS.md` for project-specific rules.
