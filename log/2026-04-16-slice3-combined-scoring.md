# Experiment Log: Slice 3 — Combined L1+L2 Scoring

**Date:** 2026-04-16
**Run:** score_all with gate logic
**Model:** google/gemini-3.1-pro-preview (Layer 2 judge)
**Cost:** ~$8.50 (122 L2 calls × Gemini 3.1 Pro, 28 skipped by gate)

## What Was Done

First end-to-end run of BOTH scoring layers on all 3 text classes:

1. **Layer 1** (deterministic): scored all 150 texts (free, <10ms each)
2. **Gate logic**: 28/150 texts skipped Layer 2 (19% savings)
   - 14 human texts scored > 0.85 (clearly human, skip)
   - 14 AI texts scored < 0.30 (clearly AI, skip)
   - All 50 humanized texts sent to Layer 2 (always score final output)
3. **Layer 2** (LLM judge): 122 texts scored with Gemini 3.1 Pro
   - 2/122 failed (0.5 neutral default used)
   - 8 dimensions per text: structural_symmetry, specificity, formality_gradient,
     voice_consistency, rhetorical_sophistication, padding_density,
     personality_presence, copula_avoidance
4. **Combined score**: 0.4 × L1 + 0.6 × L2

## Results

### Mean Scores by Class
```
              L1      L2      Combined
human         0.865   0.953   0.903
ai            0.427   0.167   0.271
humanized     0.907   0.925   0.918
```

### AUROC
```
                              L1       Combined
human vs AI                   1.0000   1.0000
humanized vs AI               1.0000   1.0000
human vs humanized            0.3094   0.3444
```

## Key Insights

1. **Layer 2 confirms Layer 1** — AI text scores even lower on L2 (0.167)
   than L1 (0.427). The LLM judge catches deeper patterns that regex misses.

2. **Human vs humanized: still indistinguishable** — Combined AUROC 0.34
   (below chance). Neither L1 nor L2 can separate original human text from
   well-humanized text. This is GOOD — it means Gemini 3.1 Pro's humanization
   actually works. But it also means our current scorer can't detect it.

3. **Humanized scores slightly higher than human** — Combined 0.918 vs 0.903.
   The humanizer over-corrects: it removes ALL AI patterns, including ones
   that humans sometimes use naturally (occasional em-dashes, etc.).

4. **Gate saved 19%** — 28 obvious cases skipped. At scale this saves
   real API cost. Expected to increase to 30-50% with more diverse data
   where clear human/AI texts are more common.

5. **2/122 L2 failures (1.6%)** — Gemini 3.1 Pro occasionally returns
   empty responses for structured output. Error handling with neutral
   defaults kept the pipeline running.

## What This Means for the Project

- **Scoring works end-to-end** — both layers, gate, combination, export
- **The humanizer is good enough** — indistinguishable from human by our scorer
- **SFT training data is ready** — 50 pairs with clear before/after delta
- **Next bottleneck is data scale** — need 500+ samples from diverse sources
  to build a proper benchmark and find scorer weaknesses

## Files Created
- `data/benchmark/scored_combined_v01.jsonl` — 150 texts with L1+L2+combined scores
- `src/humanize_rl/score_all.py` — Full L1+L2 scoring runner with gate
