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


