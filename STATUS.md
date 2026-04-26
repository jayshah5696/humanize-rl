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
- **V03 V-Slice 0 (walking skeleton):** seed schema, 10 hand-pasted
  instruction_technical seeds, AIify v02 + humanize v02 prompts and
  configs, v03 pair gate (subset), end-to-end driver `just v03-ws`,
  12 new tests. See `docs/solutions/v03-walking-skeleton.md`.
- **V-Slice 0.5 (Layer 1 detector tuning):** widened hedge / opener /
  padding regexes to match uncontracted forms (`it is worth noting`,
  `I would be happy to`) and the openers the LLM actually emits
  (`when working with`, `in modern <noun> workflows`). Fixed an
  occurrence-counting bug in `score_hedging`. Added 9 regression
  tests pinned to live AIify outputs. Walking-skeleton acceptance
  went 0/10 → 1/10 with no LLM re-run. 112 tests total.
- **V-Slice 2 (best-of-N candidate selection):** added
  `src/humanize_rl/data/selector.py` (group by original-seed,
  Layer-1 score each candidate, pick by band-distance / overshoot-aware
  tier ordering). Two-tier selector enforces 1.25× length cap above
  band-distance. `scripts/duplicate_seeds.py` works around arka's
  TransformGeneratorStage ignoring `generation_multiplier`. 7 selector
  tests. Best-of-4 + length-aware selector hit 4/10 acceptance.
- **V-Slice 1 (second domain):** added 10 email/professional seeds;
  domain-aware AIify and humanize prompts; per-domain reporting in
  `walking_skeleton.py`. Result: instruction_technical 4/10, email 0/10
  — surfaced that email AIify is fundamentally weaker than tech AIify.
- **V-Slice 3 (preservation gate):**
  `src/humanize_rl/data/preservation.py` — entity/number/role
  preservation diffs wired into `pair_gate_v03`. Singleton-skip
  heuristic learned from V-Slice 1 false positives. 15 preservation
  tests + 3 gate-integration tests.
- **V-Slice 4 (full report shape):**
  `src/humanize_rl/data/report_v03.py` emits the spec's three splits
  (`v03_core`, `v03_ood_ai`, `v03_diagnostics`) plus JSON/MD report
  with per-domain, per-length-band, AUROC, manual-review queue.
  8 report tests. Final state: AUROC human_vs_ai = 1.000,
  humanized_vs_ai = 1.000, human_vs_humanized = 0.206 (overshoot is
  the headline known issue). 147 tests total.

## Next up

Walking skeleton (V-Slices 0–4) is complete. See
`docs/solutions/v03-walking-skeleton-final.md` for the full retrospective
and horizontal scale-up plan. Two open priorities:

1. **Email AIify weakness.** AIify can't drag email seeds below 0.61 —
   the email-specific AI tells ("I wanted to circle back", over-formal
   closing inflation) aren't in the Layer 1 pattern set. Either expand
   `src/humanize_rl/scoring/patterns.py` for email-genre AI tells, OR
   accept that the same AIify prompt won't work cross-domain and ship
   one prompt per domain.
2. **Humanize overshoot** (AUROC human_vs_humanized = 0.206). Selector
   enforces `humanized ≤ original + 0.05` per-row but at the aggregate
   level humanized still ends up systematically higher than human. The
   underlying cause is the AIify-then-humanize loop training the
   humanize prompt to over-correct. Worth either (a) tightening the
   humanize prompt's overshoot ceiling, or (b) adding an L2 judge
   penalty when humanized > original.
3. **H1.5 — real corpus loaders.** Decision needed: add `datasets`
   as a runtime dep, or as a dev extras group used by a one-time
   data-prep script. See `docs/solutions/v03-walking-skeleton-final.md`.

## Known gaps

- Benchmark is still prototype-scale
- Training package is not implemented yet
- Reward / RL package is not implemented yet
- README needs more operational detail for outputs, env vars, and workflow

## Known risks

- Current small benchmark may overstate generalization because AIify injects patterns Layer 1 already detects
- Layer 2 adds useful signal for human-vs-AI, but human-vs-humanized remains intentionally hard
