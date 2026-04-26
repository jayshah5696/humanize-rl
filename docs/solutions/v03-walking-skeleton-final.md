# V03 Walking Skeleton — Vertical Slices 0 → 4 Complete

**Date:** 2026-04-25
**Final tag:** `v03-vslice4-full-report`
**Spec:** `docs/plans/v03-seed-benchmark-spec.md`
**Earlier retro:** `docs/solutions/v03-walking-skeleton.md`

## Summary

Walked the entire v03 vertical end-to-end at 20-seed scale, across two
domains, with every piece of pipeline machinery the spec requires. Each
slice was its own commit-tagged checkpoint so failures were caught at
$0.50, not $25.

## Slice-by-slice

| Tag | Slice | What it added | Cost | Acceptance |
|---|---|---|---|---|
| `v03-walking-skeleton-done` | 0 | Schema + 10 hand-pasted instruction_technical seeds + AIify v02 + humanize v02 + minimal gate + driver + 12 tests. End-to-end works. | ~$0.55 | 0/10 |
| `v03-ws-iter1-done` | 0.iter1 | Tightened AIify prompt (5 patterns + no-new-sentences cap) | ~$0.55 | 0/10 |
| `v03-ws-iter2-layer1-fix` | 0.5 | **Critical bugfix:** widened hedge/opener/transition regexes to match uncontracted forms (`it is worth noting`) and openers the LLM actually emits (`when working with`). 9 regression tests pinned to live AIify outputs. | $0 | 1/10 |
| `v03-vslice2-done` | 2 (initial) | Best-of-2 candidate selection + `scripts/duplicate_seeds.py` (works around arka transform-stage ignoring `generation_multiplier`). 6 selector tests. | ~$1.20 | 2/10 |
| `v03-vslice2-tuned-4-of-10` | 2.tuned | Best-of-4 AIify; two-tier selector that prefers in-cap candidates over slightly-better-scored over-cap candidates. | ~$1.40 | **4/10** |
| `v03-vslice1-two-domains` | 1 | Added 10 email/professional seeds; per-domain AIify prompt with discourse-role list; per-domain reporting in `walking_skeleton.py`. | ~$2.50 (20 seeds × 4 + 20 × 3) | 5/20 |
| `v03-vslice3-preservation` | 3 | `preservation.py` — entity / number / role preservation diffs. Singleton skip heuristic learned from V-Slice 1 false positives. Wired into pair gate v03. 15 tests. | $0 | 4/20 |
| `v03-vslice4-full-report` | 4 | `report_v03.py` — emits `v03_core`, `v03_ood_ai`, `v03_diagnostics` splits + JSON/MD report with per-domain, per-length-band, AUROC, manual-review queue. 8 tests. | $0 | 4/20 + report |

## Final numbers (V-Slice 4 report, `runs/v03/v03_report.md`)

```
core rows:            60   (20 triples × 3 classes)
ood-ai rows:          75   (legacy long-form ChatGPT/Claude explainers)
diagnostics rows:     48

Per-class mean (core):
  ai           0.528
  human        0.808
  humanized    0.872

AUROC (core):
  human_vs_ai           1.000   ← Layer 1 perfectly splits the matched set
  humanized_vs_ai       1.000   ← humanize works
  human_vs_humanized    0.206   ← humanize OVERSHOOTS — humanized > human

Per-domain mean score:
  email                  ai=0.607  human=0.811  humanized=0.922  ← AIify weak
  instruction_technical  ai=0.449  human=0.805  humanized=0.821  ← clean

OOD AI summary (legacy long-form, separate split per spec §1.2):
  n=75  mean=0.446  frac<0.5: 88%  frac<0.7: 100%   ← Layer 1 catches them
```

## Test count over time

`pytest -q`: 67 → 103 → 112 → 118 → 119 → 134 → 137 → 139 → 147 (final).
Lint: clean. Coverage on touched modules: 60%.

## Pipeline now has every spec piece

- [x] Seed schema (5 domains, 14 discourse roles, length bands, anchors)
- [x] Domain-aware AIify prompt (subset patterns; preserves discourse role)
- [x] Domain-aware humanize prompt (preserves entities, numbers, role)
- [x] Best-of-N candidate selection (band-distance for AIify; tier-ordered overshoot-aware for humanize)
- [x] Pair gate v03: thresholds, deltas, length ratios, dim-improvement count, **preservation checks (entities/numbers/role)**, suspicion flags, manual-review queue
- [x] Three benchmark splits: matched core, OOD AI, diagnostics
- [x] Per-domain, per-length-band, per-dimension reporting + 3 AUROC pairs
- [x] One-command runner: `just v03-ws`

## What's NOT done (deliberate scope cuts)

- **Real corpus loaders** (Enron, GoodWiki, FineWeb-Edu, RAID human, peS2o, WritingPrompts, PG-19). Walking-skeleton uses `curated_paste` for both domains. Adding `datasets` HF dep is V-Slice 1.5.
- **Layer 2 LLM judge** in the walking skeleton. Currently L1-only to keep the loop free. `score_all.py` still works for combined scoring on the legacy benchmark.
- **3 of 5 spec domains** (academic, blog/opinion, creative). The pipeline machinery handles them; just need seeds.
- **Email AIify weakness.** AIify can't drag email seeds below 0.61 because the email-specific AI tells (over-formal closing, soft CTA inflation, "I wanted to circle back") aren't in the Layer 1 pattern set. Needs Layer 1 expansion specifically for emails.
- **Humanize overshoot.** AUROC human_vs_humanized = 0.206 confirms the spec's #1 concern. The selector enforces `humanized ≤ original + 0.05` but at scale it can't always avoid overshoot if no candidate satisfies. Best-of-3 helps but isn't enough.

## Recommended horizontal scale-up (mechanical, no new code)

The pipeline is now a tunable surface. Each scale-up is a config change + rerun:

| Scale-up | Action | Cost est. |
|---|---|---|
| H1: 20 → 50 seeds (5 domains × 10) | hand-paste 30 more seeds | ~$5 |
| H1.5: replace `curated_paste` | add `datasets` dep + 7 small loaders | $0 dev + ~$5 first run |
| H2: 50 → 200 seeds | run loaders with `--accept-target 200` | ~$15 |
| H3: 200 → full benchmark v03_core | re-run full pipeline | ~$25 |

### Open question for H1.5

Per AGENTS.md "Ask before adding major frameworks." The HF `datasets`
package is large (~50 MB + transitive). Two options:
- **(a)** add `datasets` as a runtime dep (cleanest, supports streaming).
- **(b)** add `datasets` only in a `dev` extras group; loaders go behind
  a `python -m humanize_rl.data.loaders` entry point that's a one-time
  data prep step.

I'd default to (b) so the runtime install stays small — but this is a
question for V-Slice 1.5.
