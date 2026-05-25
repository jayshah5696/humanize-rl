# Humanize-RL Track A Distilled Scorer

Local surrogate scorers for humanness / AI-writing-pattern detection.

## Artifacts

- `ridge.pkl`: TF-IDF 1-4 gram + Logistic/Ridge heads. Recommended default.
- `fasttext.pkl`: fastText supervised classifier + vector regression heads. Secondary comparator.
- `metadata.json`: training summary.

## Training data

Trained on `data/scorer/l2_labeled_scorer_v02_10k.jsonl`: 10,000 L2-rubric-labeled rows.
Labels include AI-generated, human-authored, and humanized-synthetic examples.

## Evaluation summary

Ridge achieved ~0.997-0.999 AUROC across capped random/group splits, with the best rubric MSE and ~1ms/row latency.
False-positive challenge set showed 0.0% false positives at threshold 0.5 on 400 human-authored challenge rows.

See the project repository for full reports and figures.
