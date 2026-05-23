# v03 walking-skeleton report

- core rows: **60**
- ood-ai rows: **75**
- diagnostics rows: **39**

## Per-class mean (core)
- **ai**: 0.522
- **human**: 0.808
- **humanized**: 0.866

## AUROC (core)
- **human_vs_ai**: 1.000
- **humanized_vs_ai**: 1.000
- **human_vs_humanized**: 0.236

## Per-domain mean score (core)
- **email**
  - ai: 0.595
  - human: 0.811
  - humanized: 0.910
- **instruction_technical**
  - ai: 0.449
  - human: 0.805
  - humanized: 0.821

## Per-length-band (core)
- **medium**
  - ai: 0.506
- **short**
  - ai: 0.526
  - human: 0.808
  - humanized: 0.866

## OOD AI summary (legacy long-form, separate)
- **n**: 75
- **mean_score**: 0.446
- **frac_below_0.5**: 0.880
- **frac_below_0.7**: 1.000

## Manual review queue (top 10 failed triples)
- `triple_003_human` (instruction_technical, label=human, score=0.75)
- `triple_003_ai` (instruction_technical, label=ai, score=0.375)
- `triple_003_humanized` (instruction_technical, label=humanized, score=0.775)
- `triple_004_human` (instruction_technical, label=human, score=0.775)
- `triple_004_ai` (instruction_technical, label=ai, score=0.4)
- `triple_004_humanized` (instruction_technical, label=humanized, score=0.725)
- `triple_007_human` (instruction_technical, label=human, score=0.7375)
- `triple_007_ai` (instruction_technical, label=ai, score=0.45)
- `triple_007_humanized` (instruction_technical, label=humanized, score=0.8375)
- `triple_010_human` (email, label=human, score=0.8)