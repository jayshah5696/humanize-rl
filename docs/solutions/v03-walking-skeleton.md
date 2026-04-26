# V03 Walking Skeleton (V-Slice 0) — done

**Date:** 2026-04-25
**Tag:** `v03-walking-skeleton-done`
**Spec:** `docs/plans/v03-seed-benchmark-spec.md`
**Vertical-slice plan:** see chat dated 2026-04-25 ("vertical first, horizontal later")

## Goal

Prove the **whole v03 pipeline shape** end-to-end on a tiny single-domain
slice, before sourcing 200 corpus seeds or building dimension-aware AIify
sampling. Walking-skeleton domain: `instruction_technical` only, 10 seeds,
`source_dataset: curated_paste`.

## What this slice contains

| Concern | Where | Notes |
|---|---|---|
| Seed schema (Pydantic) | `src/humanize_rl/data/seed.py` | 5 domains, 14 discourse roles, length-band, anchors |
| Walking-skeleton seeds | `seeds/v03/walking_skeleton.jsonl` (10 rows) | hand-pasted; defers HF `datasets` dep to V-Slice 1 |
| Seed builder | `scripts/build_walking_skeleton_seeds.py` | computes word_count + anchor_count |
| AIify v02 prompt | `prompts/aiify_v02.txt` + `configs/v03/01-aiify-walking-skeleton.yaml` | subset of 3 patterns, intensity=medium, hardcoded |
| Humanize v02 prompt | `prompts/humanize_v02.txt` + `configs/v03/02-humanize-walking-skeleton.yaml` | preserves discourse role |
| Pair gate v03 (subset) | `src/humanize_rl/data/pair_gate_v03.py` | thresholds + length ratios + dim-improvement count |
| End-to-end driver | `src/humanize_rl/data/walking_skeleton.py` | `just v03-ws` |
| Tests | `tests/data/test_seed_schema.py`, `tests/data/test_pair_gate_v03.py` | 12 new tests, all green |
| One-shot runner | `just v03-ws` | seeds → AIify → humanize → score → gate → export |

## Run command

```bash
just v03-ws            # full slice, ~$0.55, ~3 min
# or step by step:
just v03-ws-seeds
just v03-ws-aiify      # ~$0.05, Gemini 3.1 Flash Lite
just v03-ws-humanize   # ~$0.50, Gemini 3.1 Pro
just v03-ws-score      # free, Layer 1 only
```

## First-run results (2026-04-25)

```
triples:                10
accepted by gate:       0/10 (0%)
mean original score:    0.805
mean aiified score:     0.711
mean humanized score:   0.826
mean aiify delta:       +0.094  (target ≥ +0.25)
mean humanize delta:    +0.115  (target ≥ +0.30)
```

Pipeline runs end-to-end. Gate works. Acceptance rate is 0% because the
**signal-to-noise ratio is too low** at this slice: AIify is asked to add
only 3 mild patterns, so it can't drag the score down to ≤ 0.45, and
humanize has very little to recover.

This is exactly the diagnostic V-Slice 0 was meant to expose.

## Findings to feed into next slices

1. **AIify is too gentle.** With intensity=`medium` and only 3 patterns,
   mean aiified score is 0.71. The spec target band is 0.20-0.40. V-Slice 2
   (selection) should help by trying 2 candidates, but V-Slice 1 should also
   widen the pattern subset to 4 dims and bias toward `medium`/`heavy`.
2. **Length blowup is the dominant rejection.** AIify routinely adds 25-46%
   length even when the prompt says "within 20%". 6/10 triples fail
   `length_ratio_aiify/original > 1.25`. Two fixes to try:
   - tighten the prompt ("hard cap: ±15%, do not add new sentences")
   - add a length-aware regenerate in the V-Slice 2 selector
3. **Walking-skeleton seeds are short** (median 80 words) — below the
   instruction_technical preferred band (120-240). Expected, since they're
   hand-pasted to keep this slice trivial. V-Slice 1's GoodWiki / FineWeb-Edu
   loader will hit the right band naturally.
4. **Anchor regex is too narrow.** Several seeds with obvious anchors
   (UnicodeDecodeError, pg_stat_activity, ~/.local/share) score
   `anchors_count = 0`. Improve in V-Slice 1 alongside the real loaders.
5. **Recovery ratio > 1 on most rows** — humanize already pushes scores
   above the original. This is the v03 spec's biggest concern (§10.4).
   V-Slice 3's preservation gate (entities, role, stance) needs to be in
   place before we trust any SFT exports.

## Out of scope (deliberately)

- Layer 2 LLM judge — V-Slice 0 uses Layer 1 only to keep the loop free.
  Re-introduced in V-Slice 2.
- Multi-domain sampling — V-Slice 1.
- Candidate selection (best of N) — V-Slice 2.
- Entity / number / discourse-role preservation checks — V-Slice 3.
- Real corpus loaders (Enron / GoodWiki / FineWeb-Edu / RAID / peS2o /
  WritingPrompts / PG-19) — V-Slice 1 onwards.

## Next: V-Slice 1

Add `email/professional` as the second domain (10 seeds), introduce the
first real corpus loader, branch AIify dim sampling per domain, and add
per-domain reporting.

---

## Iteration 1 (2026-04-25, tag `v03-ws-iter1-done`)

### What changed

- AIify v02 prompt tightened in `configs/v03/01-aiify-walking-skeleton.yaml`:
  - widened pattern set 3 → 5 (added HEDGING and OPENER, kept PADDING /
    TRANSITIONS / STRUCTURAL_SYMMETRY)
  - added a hard "do not add new sentences" rule (more enforceable than a %
    cap)
  - tightened length cap to 1.05-1.15× ideal, 1.20× hard max
- One small Layer 1 bugfix in `score_hedging`: count *occurrences* across
  all patterns, not the number of distinct patterns matched. The old logic
  scored "it's worth noting it's worth noting it's worth noting" the same
  as one occurrence. All 55 scoring tests still pass.
- The standalone `prompts/aiify_v02.txt` was rewritten to take
  `{target_dimensions}` as a placeholder (used by V-Slice 1+ programmatic
  prompt construction). The arka config keeps a flattened version inline
  because arka's prompt template only interpolates `{input_text}`.

### Results

```
triples:                10
accepted by gate:       0/10 (0%)
mean original score:    0.805  (unchanged — same seeds)
mean aiified score:     0.711 → 0.662   (↓ 0.05, AIify is harsher)
mean humanized score:   0.826 → 0.818   (basically unchanged)
mean aiify delta:       +0.094 → +0.143 (↑ 50%, but target is +0.25)
mean humanize delta:    +0.115 → +0.155 (↑ 35%, but target is +0.30)
length_ratio rejections: 6/10 → 4/10
```

Progress on every metric, but still 0/10 accepted.

### Root cause exposed by iteration 1

The AIified outputs are **clearly AI-flavored** to a human reader ("When
working with pytest, it is worth noting that... Furthermore... Moreover...
Additionally... In general..."). Layer 1 only catches `transition_overuse`
and half-credits `sentence_variance`; the hedging / opener / padding
patterns slip through.

Direct cause: most of the regexes in `src/humanize_rl/scoring/patterns.py`
require contracted forms (`it'?s worth noting`, `i'?d be happy to`). The
LLM consistently writes the *uncontracted* AI-flavored variant ("it is
worth noting", "I would be happy to"), so the patterns never fire.

### Decision

This is **not** a v03 walking-skeleton issue — it's a Layer 1 detector gap
that would block every downstream slice. Promote it to its own slice
before V-Slice 1.

---

## Iteration 2: V-Slice 0.5 — Layer 1 detector tuning (2026-04-25, tag `v03-ws-iter2-layer1-fix`)

### What changed (no LLM re-run; pure scoring fix)

In `src/humanize_rl/scoring/patterns.py`:
- Hedge regexes now accept both contracted and uncontracted forms
  (`(?:it'?s|it is) worth noting`, etc.).
- Added the openers the live LLM actually emitted on every walking-skeleton
  row: `^when working with\b`, `^in (?:modern|today's|the modern|the current)
  [a-z]+ (?:workflows?|systems?|environments?|practice)`,
  `^in the (?:context|world|realm) of\b`. Also the uncontracted
  `^(?:i'?d|i would) be happy to`.
- Added padding/hedge filler the LLM repeats every paragraph:
  `\bone common pitfall\b`, `\bone of the (?:most |key )?(?:common|key|...)`,
  `\bin general,`, `\bgenerally speaking,`, `\bin many cases,`,
  `\bas a general rule,`, `\bwhat'?s interesting (?:here )?is\b`.
- Same uncontracted-form widening on the transition pattern that
  duplicated the hedge phrase.

9 regression tests pinned to actual AIified strings live in
`tests/scoring/test_layer1_v03_regression.py`. They lock the fix.

### Results (same arka outputs, re-scored)

```
triples:                10
accepted by gate:       0/10 → 1/10  ✅ first green pair end-to-end
mean original score:    0.805            (unchanged — same seeds)
mean aiified score:     0.662 → 0.465   (↓ 0.20)
mean humanized score:   0.818 → 0.786   (↓ 0.03)
mean aiify delta:       +0.143 → +0.340 ✅ target was ≥ +0.25
mean humanize delta:    +0.155 → +0.321 ✅ target was ≥ +0.30
```

### Per-row diagnosis after the fix

| Triples | Reject reason | Likely fix |
|---|---|---|
| 6, 7, 8 | only `length_ratio_aiify/original > 1.25` | V-Slice 2 selector (best-of-2) or stricter prompt cap |
| 0, 5, 9 | only `aiified_score > 0.45` (narrow miss: 0.46-0.49) | Slightly more aggressive AIify prompt or selector |
| 2 | length + narrow humanize delta | combo of above |
| 1 | humanize did not recover (humanized=0.65) | Humanize prompt / selector |
| 3 | humanize delta short by 0.025 | Selector |
| 4 | accepted | — |

Four of nine remaining rejects are pure length-cap issues. Three are narrow
AIify misses. The pipeline shape and gate logic are now demonstrably correct.

### Total test count

103 → 112 (+9 regression tests). All green. Lint clean.

## Next: V-Slice 1

Add `email/professional` as the second domain (10 seeds), introduce the
first real corpus loader (Enron via HF `datasets`), branch AIify dim
sampling per domain, and add per-domain reporting. Layer 1 is now trusted
enough to make the per-domain numbers meaningful.

---

## Iteration 3: V-Slice 2 — best-of-2 candidate selection (2026-04-25, tag `v03-vslice2-done`)

### What changed

- **Selector** (`src/humanize_rl/data/selector.py`): groups arka transform
  output rows by `system.transform_original.text`, scores each candidate
  with Layer 1, picks the best per group.
  - AIify mode: pick by closeness to band midpoint 0.30; tie-break on
    shorter candidate (combats length blowup).
  - Humanize mode: pick by largest `humanize_delta` while respecting the
    `humanized ≤ original + 0.05` overshoot ceiling. Two-hop lookup bridges
    AIified-text → human-original-text → score.
- **Seed-duplication trick** (`scripts/duplicate_seeds.py`): arka's
  `TransformGeneratorStage` ignores `generation_multiplier` (it's only used
  by prompt-generation stages). To get N candidates per seed without
  modifying arka, the seed file is duplicated upstream. AGENTS.md said no
  custom arka stages — this stays in our repo.
- **Tests** (`tests/data/test_selector.py`): 6 tests covering grouping,
  AIify band selection, length tie-break, humanize tier-0 preference,
  overshoot avoidance.
- **`just v03-ws` chain extended**:
  ```
  v03-ws-seeds  (writes _x2.jsonl)
  v03-ws-aiify  → 20 candidates
  v03-ws-aiify-select  → 10 best (writes _x2.jsonl too)
  v03-ws-humanize  → 20 candidates
  v03-ws-humanize-select  → 10 best
  v03-ws-score  → gate, export, report
  ```

### Results

```
triples:                10
accepted by gate:       1/10 → 2/10  (↑ 100% relative)
mean original score:    0.805            (unchanged)
mean aiified score:     0.465 → 0.463   (selector held the line)
mean humanized score:   0.786 → 0.765   (slight drop: selector chose less
                                          aggressive humanizations to
                                          avoid overshoot)
mean aiify delta:       +0.340 → +0.343 ✅
mean humanize delta:    +0.321 → +0.303 ✅ (still over target)
```

Cost: ~$1.20 for the full V-Slice 2 run (vs $0.55 for V-Slice 0/0.5).

### Per-row diagnosis after best-of-2

| Reject reason | Triples | Notes |
|---|---|---|
| `aiified_score > 0.45` (narrow miss 0.46-0.50) | 0, 1, 2, 7, 8 | Both candidates landed above the band; selector picked the lower one but neither hit ≤ 0.45 |
| `length_ratio_aiify/original > 1.25` | 2, 7, 8 | AIify still adding 25-37% length on long technical seeds; selector preferred the in-band candidate even if longer |
| `humanize_delta < 0.3` | 1, 2, 3, 5, 9 | Several humanize candidates were too conservative |
| `humanized_score < 0.75` | 3, 5, 9 | Humanize got ~0.65-0.74 |
| Accepted | 4, 6 | both deltas ≥ 0.375; lengths within band |

### Findings

1. **Best-of-2 is not enough on its own.** Many seeds had both candidates
   above the AIify band (0.46-0.55). The selector can only pick from what
   it's given. To reliably hit `aiified ≤ 0.45`, we need either:
   (a) more candidates (best-of-3 or best-of-4), or
   (b) a stronger AIify prompt, or
   (c) regenerate-on-failure: if best candidate misses the band, re-prompt.
2. **Length blowup is the persistent bug.** Even after iter 1 prompt
   tightening, AIify routinely adds 20-37% length. The "do not add new
   sentences" instruction isn't being honored. Worth trying:
   `temperature=0.4` (down from 0.7) or a different model.
3. **Humanize needs more help than AIify.** 5 of 8 rejects involve
   humanize. The Pro model is producing safe-but-bland rewrites. Best-of-2
   helps marginally; best-of-3 with explicit "you must achieve
   humanized_score ≥ 0.85" framing might help more.
4. **The pipeline is now a tunable surface, not a question mark.** Each
   reject reason maps cleanly to a knob: AIify candidate count, AIify
   temperature, humanize candidate count, humanize prompt sharpness.

## Recommendations before V-Slice 1

- Bump AIify candidate count from 2 → 4 (cost: +$0.20). Easy mechanical change.
- Drop AIify temperature 0.7 → 0.4. Free.
- Bump humanize candidate count from 2 → 3. Cost: +$0.50.
- Then V-Slice 1 lands on a pipeline that produces ≥5/10 accepted pairs
  per 10 seeds, which is enough signal for per-domain comparisons.
```
