# Development Log

## [2026-05-23] Track B: SFT Data Preparation (Vertical Slice)

**Author:** Antigravity AI Agent  
**Focus:** Engineering a high-quality, diverse Supervised Fine-Tuning (SFT) dataset for Gemma 4 E2B-it alignment under the Direct Generative Humanness paradigm.

### Accomplishments

1. **Heuristic Quality Filters:**
   - Implemented [src/humanize_rl/data/filters.py](file:///Users/jshah/Documents/GitHub/humanize-rl/src/humanize_rl/data/filters.py) containing:
     - `LangDetectFilter`: Restricts outputs to English text using pure Python fallback language detection.
     - `TokenPriorPerplexityFilter`: Model-free Zipf-law and token entropy estimation to prune anomalous repetitions or gibberish.
     - `SpaCyActionFilter`: Syntactic dependency parsing to isolate commands whose root verbs are writing actions (e.g., *write*, *draft*, *summarize*) targeting suitable objects (e.g., *email*, *report*, *blog*).
   - Validated filters using Red/Green TDD in [tests/data/test_filters.py](file:///Users/jshah/Documents/GitHub/humanize-rl/tests/data/test_filters.py). All 3 test groups pass cleanly.

2. **Pre-processing & Funnel Tracking Orchestrator:**
   - Created [scripts/prep_dataset.py](file:///Users/jshah/Documents/GitHub/humanize-rl/scripts/prep_dataset.py) to stream and filter bulk base datasets (LMSYS-Chat-1M and Alpaca).
   - Logged pipeline funnel metrics and generated drop-off charts (`funnel_dropoff.png`) directly to MLflow.

3. **Stream Merger & Format Normalization:**
   - Modified [scripts/publish_dataset.py](file:///Users/jshah/Documents/GitHub/humanize-rl/scripts/publish_dataset.py) to aggregate legacy gold pairs, corpus SFT scaleup data, walking skeleton outputs, and expansion SFT sets.
   - Solved a pipeline `KeyError` by adding a domain-to-instruction fallback mapping when processing files missing explicit instructions (specifically `expansion_sft.jsonl`).
   - Wrote both `messages` (ShareGPT format) and flat `instruction`/`response` keys to `v03_combined_sharegpt.jsonl` for dual compatibility.
   - Auto-generated a pie chart showing domain distribution inside `runs/v03/domain_distribution.png`.

4. **Arka Ingestion Configuration:**
   - Configured [configs/v03/03_sft_data_prep.yaml](file:///Users/jshah/Documents/GitHub/humanize-rl/configs/v03/03_sft_data_prep.yaml) to ingest the combined tasks, normalize conversations, and output the final training-ready `v03_combined_sft.jsonl` dataset.

5. **Just Command Automation:**
   - Updated [justfile](file:///Users/jshah/Documents/GitHub/humanize-rl/justfile) with new developer commands:
     - `just prep-data`: Runs streaming, pre-processing, and MLflow logging.
     - `just merge-data`: Aggregates all streams and runs the Arka data preparation pipeline.
     - `just publish-dataset`: Formats and uploads the dataset and visual metadata to Hugging Face Hub.

### Verification Metrics

- **Total Merged Rows:** 1,269 unified SFT entries.
- **Pipeline Execution:** Successfully ran `just merge-data` end-to-end in 4.1s.
- **Hugging Face Publication:** Successfully uploaded the unified SFT dataset and domain distribution plot metadata to Hugging Face Hub under `jayshah5696/humanize-rl-sft-dataset`.
- **Test Integrity:** Passed all 5 general pytest checks and 3 filter-specific checks without regressions.


## [2026-05-23] Track B: SFT Data Prep v2 — Slices 1 to 5 Implementation and Validation

**Author:** Antigravity AI Agent  
**Focus:** Implementing and validating Slices 1 to 5 for SFT Dataset v2, focusing on Stream A pilot execution and prompt calibration.

### Accomplishments

1. **Structured Output Calibration:**
   - Diagnosed an issue in Arka where `supports_json_schema: false` forced the LLM client to use OpenAI's native `.parsed` choice extraction on OpenRouter, resulting in parsing failures (`invalid_structured_response: provider returned no parsed object`).
   - Enabled `supports_json_schema: true` in [configs/v04/stream_a_evol_pilot.yaml](file:///Users/jshah/Documents/GitHub/humanize-rl/configs/v04/stream_a_evol_pilot.yaml) to ensure structured JSON outputs are validated via the robust compatible JSON schema strategy.

2. **Pipeline Checkpoint Synchronization:**
   - Identified a checkpoint propagation bug where re-executing an intermediate failed stage did not invalidate downstream completed checkpoints, causing Arka to reuse outdated results from the file system.
   - Cleared the stale stage runs and Parquet folders for downstream dedup and filter stages (`02c_exact_dedup`, `02d_near_dedup`, `02a_length_filter`, `02b_language_filter`) inside [configs/v04/state.db](file:///Users/jshah/Documents/GitHub/humanize-rl/configs/v04/state.db) and run stages.

3. **Stream A Pilot Execution:**
   - Successfully executed the Stream A pilot run using `--resume`, yielding 59 high-quality records from 30 initial pilot seeds.
   - Outputs are successfully processed, filtered, and saved to [data/processed/v04_stream_a_pilot.jsonl](file:///Users/jshah/Documents/GitHub/humanize-rl/data/processed/v04_stream_a_pilot.jsonl).

### Verification & Manual Review

- **Manual Spot-check:** Reviewed the output JSONL file and confirmed that the model generates extremely natural, human-sounding Slack messages and emails.
- **Tone & Style:** The output matches requested registers (casual vs. formal), uses natural contractions (e.g., *I've*, *We'll*, *I'm*), eliminates all AI hedging/boilerplate phrases (e.g., *Certainly!*, *Please do not hesitate to reach out*), and preserves the facts/context accurately.
- **Run Success:** The pipeline executed successfully to completion in 0.4s on the resumed run.



## [2026-05-23] Track A: 10K Layer-2 Scorer Calibration and HF-Ready Model Artifacts

**Author:** Pi Coding Agent  
**Focus:** Building a high-quality distilled Layer 2 humanness scorer from rubric-labeled data, validating false positives, and preparing model artifacts for Hugging Face release.

### Accomplishments

1. **Built a 10K Layer-2 labeled scorer dataset:**
   - Created `data/scorer/l2_labeling_pool_v02_10k.jsonl` from balanced AI-generated, human-authored, and humanized-synthetic examples.
   - Labeled all 10,000 rows with the humanness Layer 2 rubric using Gemini judge calls.
   - Produced `data/scorer/l2_labeled_v02_10k.jsonl` and scorer-ready `data/scorer/l2_labeled_scorer_v02_10k.jsonl`.
   - Verified 10,000 unique rows, 0 duplicates, and only 8 rows with empty `l2_per_dim`.

2. **Improved the labeling infrastructure:**
   - Added resumable L2 labeling with duplicate-safe appends in `scripts/label_l2_pool.py`.
   - Added `scripts/dedupe_l2_labels.py` to clean duplicate IDs.
   - Added `scripts/build_l2_labeling_pool.py` and `scripts/merge_l2_labels_into_scorer_data.py`.
   - Fixed the previous sequential labeling bottleneck by adding real concurrent scoring with `ThreadPoolExecutor`.

3. **Ran scorer model comparison:**
   - Evaluated Ridge, fastText, dense Tiny, and dense Luxical scorer variants.
   - Generated final reports and Tufte-style compact comparison figures in `runs/track_a_capped/`.
   - Ridge achieved the best practical tradeoff: strong AUROC, lowest rubric MSE, small artifact size, and ~1ms/row latency.

4. **Built and evaluated a human false-positive challenge set:**
   - Created `data/scorer/false_positive_human_v01.jsonl` with 400 human-authored hard negatives from formal email, arXiv abstracts, StackExchange markdown, and local formal examples.
   - Generated false-positive report in `runs/false_positive_eval/`.
   - Ridge produced 0.0% false positives at threshold 0.5; fastText produced 0.25%; dense Luxical produced 18.5%.

5. **Trained final local scorer artifacts:**
   - Trained final Ridge and fastText models on the 10K L2-labeled scorer dataset.
   - Saved artifacts under `models/track_a_10k/`:
     - `ridge.pkl`
     - `fasttext.pkl`
     - `metadata.json`
   - Dense Luxical was evaluated but not packaged because its SentenceTransformer wrapper is not pickle-safe on the current stack.

6. **Paper/report integration:**
   - Added `paper/track_a_scorer_report.md` with data construction, architecture, metric definitions, figures, false-positive analysis, shortcut analysis, limitations, and model-selection rationale.

### Verification Metrics

- **L2 scorer dataset:** 10,000 rows, 4,946 AI-generated, 3,308 human-authored, 1,746 humanized-synthetic.
- **Mean L2 scores:** AI = 0.283, human-authored = 0.866, humanized-synthetic = 0.828.
- **Selected scorer:** Ridge.
- **Ridge capped benchmark:** AUROC ≈ 0.997-0.999, rubric MSE ≈ 0.036-0.046 depending split, latency ≈ 1ms/row.
- **False-positive challenge:** Ridge false-positive rate = 0.0% on 400 human-authored hard negatives.

### Artifacts

- Dataset: `data/scorer/l2_labeled_scorer_v02_10k.jsonl`
- Final models: `models/track_a_10k/`
- Scorer report: `runs/track_a_capped/REPORT.md`
- False-positive report: `runs/false_positive_eval/REPORT.md`
- Paper section: `paper/track_a_scorer_report.md`
