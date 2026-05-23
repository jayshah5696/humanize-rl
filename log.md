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

