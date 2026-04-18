# Status

## Current state

The project has a working vertical slice for:

- Layer 1 deterministic humanness scoring
- AIify pipeline config
- Humanize pipeline config
- Layer 2 rubric-based scoring wrapper
- End-to-end scoring/export on small benchmark artifacts

## Done

- Slice 1: Layer 1 scorer
- Slice 2: Rubric + seeds + first AIify run
- Slice 3: Full small-scale pipeline (AIify -> humanize -> combined scoring)
- P0 infrastructure pass:
  - CI workflow present
  - `CONTRIBUTING.md` added
  - `STATUS.md` added
  - `arka` dependency changed from broken local path to pinned GitHub source

## Next up

1. Get `uv sync`, `just test`, and `just lint` green with the new dependency source
2. Add missing tests for rubric loading, gate logic, Layer 2 normalization/fallback, and pipeline export behavior
3. Build the real MVP benchmark dataset (200 samples, 3 classes)
4. Tune Layer 1 weights and Layer 2 gate thresholds on the benchmark
5. Scale pair generation only after the benchmark is credible

## Known gaps

- Benchmark is still prototype-scale
- Training package is not implemented yet
- Reward / RL package is not implemented yet
- README needs more operational detail for outputs, env vars, and workflow

## Known risks

- Current small benchmark may overstate generalization because AIify injects patterns Layer 1 already detects
- Layer 2 adds useful signal for human-vs-AI, but human-vs-humanized remains intentionally hard
