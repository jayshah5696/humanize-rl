# v03 walking-skeleton report

- core rows: **60**
- ood-ai rows: **75**
- diagnostics rows: **48**

## Per-class mean (core)
- **ai**: 0.528
- **human**: 0.808
- **humanized**: 0.872

## AUROC (core)
- **human_vs_ai**: 1.000
- **humanized_vs_ai**: 1.000
- **human_vs_humanized**: 0.206

## Per-domain mean score (core)
- **email**
  - ai: 0.607
  - human: 0.811
  - humanized: 0.922
- **instruction_technical**
  - ai: 0.449
  - human: 0.805
  - humanized: 0.821

## Per-length-band (core)
- **medium**
  - ai: 0.506
- **short**
  - ai: 0.534
  - human: 0.808
  - humanized: 0.872

## OOD AI summary (legacy long-form, separate)
- **n**: 75
- **mean_score**: 0.446
- **frac_below_0.5**: 0.880
- **frac_below_0.7**: 1.000

## Manual review queue (top 10 failed triples)
- `triple_000_human` (instruction_technical, label=human, score=0.8125)
- `triple_000_ai` (instruction_technical, label=ai, score=0.5375)
- `triple_000_humanized` (instruction_technical, label=humanized, score=0.8125)
- `triple_002_human` (instruction_technical, label=human, score=0.8125)
- `triple_002_ai` (instruction_technical, label=ai, score=0.475)
- `triple_002_humanized` (instruction_technical, label=humanized, score=0.8125)
- `triple_003_human` (instruction_technical, label=human, score=0.75)
- `triple_003_ai` (instruction_technical, label=ai, score=0.375)
- `triple_003_humanized` (instruction_technical, label=humanized, score=0.775)
- `triple_004_human` (instruction_technical, label=human, score=0.775)