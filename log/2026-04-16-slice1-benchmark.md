# Experiment Log: Slice 1 — Layer 1 Scorer + AIify Pipeline + Benchmark

**Date:** 2026-04-16
**Run ID:** aiify-v01
**Cost:** ~$0.15 (50 seeds × Gemini 3.1 Flash Lite via OpenRouter)

## What Was Done

### End-to-end vertical slice:
1. **Dataset**: 50 curated human-written text samples across 5 domains
   - email/professional: 12 samples (24%)
   - blog/opinion: 13 samples (26%)
   - essay/reflective: 8 samples (16%)
   - technical/docs: 8 samples (16%)
   - academic: 4 samples (8%)
   - social/casual: 5 samples (10%)

2. **AIify Pipeline**: Arka TransformGeneratorStage with Gemini 3.1 Flash Lite
   - All 50 seeds transformed successfully (100% yield)
   - Prompt instructs 8 specific AI pattern injections
   - Original text preserved in record system field

3. **Layer 1 Scorer**: 8 deterministic dimensions (regex + heuristics)
   - opener_pattern, hedging_density, list_overuse, sentence_variance
   - contractions, closing_pattern, em_dash_density, transition_overuse

4. **Benchmark**: AUROC + accuracy on scored pairs

## Results

### Pair Scoring
```
Pairs:           50
Original mean:   0.865
AI-ified mean:   0.427
Mean delta:      0.438
Min delta:       0.225
Max delta:       0.550
Effective AIify: 50/50 (100%)
```

### AUROC (Layer 1 on real LLM-generated AI text)
```
Overall AUROC:   1.0000
Accuracy:        100% (threshold=0.71)
```

### Per-Dimension AUROC (ranked)
```
opener_pattern         1.0000  ← perfect (AIify always adds sycophantic opener)
closing_pattern        1.0000  ← perfect (AIify always adds sign-off)
transition_overuse     1.0000  ← perfect (AIify always adds Furthermore/Moreover)
hedging_density        0.8900
contractions           0.8500
sentence_variance      0.6276
list_overuse           0.5000  ← weak (AIify doesn't always add lists)
em_dash_density        0.3700  ← inverted! AIify removes em-dashes
```

## Key Insights

1. **opener + closing + transitions are the killer trio** — these 3 dimensions alone achieve perfect separation on AIified text. This confirms the humanizer skill's emphasis on these patterns.

2. **em_dash_density is inverted** — the AIify prompt removes em-dashes (which are an AI tell themselves). Human text has more em-dashes than our specifically-AIified text. This is expected and actually correct: em-dashes are an AI writing tell, but our AIify prompt instructs removing personality markers, and em-dashes are used for dramatic effect.

3. **list_overuse at 0.5 (chance)** — many AI-ified samples kept prose structure. The AIify prompt says to "convert prose into bullet points" but Flash Lite doesn't always follow this instruction aggressively enough.

4. **100% AIify success rate** — all 50 seeds transformed with meaningful score delta. The AIify prompt is effective.

5. **Mean delta of 0.438** — substantial. Every human sample dropped by at least 0.225 points after AIification.

## Known Limitations

- Dataset is curated (hand-written), not sourced from real corpora
- AIify uses one model (Flash Lite) — real AI text comes from many models
- Layer 1 alone is likely overfit to heavy-pattern AI text
- Need Layer 2 (LLM judge) for subtle pattern detection
- em_dash_density dimension needs recalibration for this use case

## What's Next

1. **Layer 2 scoring** — Rubric-based LLM judge via Arka LabelingEngine (Gemini 3.1 Pro)
2. **Humanize pipeline** — Transform AI-ified text back to human-sounding
3. **Harder benchmark** — 3-class (human vs AI vs humanized), more diverse sources
4. **Weight tuning** — Use AUROC per-dim to tune aggregator weights

## Files Created
- `seeds/human_seeds_v01.jsonl` — 50 curated human samples
- `output/01-aiify-dataset.jsonl` — 50 AIified samples (Arka output)
- `data/benchmark/scored_pairs_v01.jsonl` — 100 scored records (50 pairs)
- `configs/01-aiify.yaml` — Arka pipeline config
- `prompts/aiify.txt` — AIification prompt template
- `rubrics/humanness_v01.yaml` — Layer 2 rubric (8 dims, ready for use)
