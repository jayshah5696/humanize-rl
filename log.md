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


## [2026-06-19] Prime p50 RL Sweep: Research Log and Training Gate

**Author:** Codex  
**Focus:** Build a simple, inspectable Prime-first RL path for a sub-3B humanizing rewrite model. Keep the optimizer reward easy to reason about:

```
p50_50_no_penalty = 0.50 * ridge_rubric_mean + 0.50 * deterministic_mean
```

Penalties remain diagnostics and strict-eval gates. They are not directly added to the p50 optimizer reward, so the deterministic half must contain the hard caps for behaviors we never want to reinforce.

### Research Question

Can a small hosted RL run improve direct, natural rewriting without learning reward hacks such as:

- option menus or wrapper prose;
- formal letter shells when the task asks for only the message;
- emoji, hashtags, or shouty all-caps unless explicitly requested;
- invented numbers, dates, threats, bribes, or unsupported narrative details;
- polished AI cadence such as `We are writing to...`;
- high ridge/humanness score while dropping required facts.

### Data State

- Base mixed task file: `data/rl/humanize_tasks_rl_mix_v2.jsonl`.
- Filtered p50 task file: `data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`.
- Published-env bundle:
  `environments/humanize_rl_env/humanize_rl_env/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`.
- Count: 972 tasks.
- Splits: 777 train / 92 validation / 103 test.
- Source: v01/v02/v03 RL tasks plus Modal saved rollouts.
- Row shape for Prime: `prompt`, `answer`, `info`, `example_id`.
- Full task payload is preserved under `info.task`.
- Ridge scorer bundled at:
  `environments/humanize_rl_env/humanize_rl_env/ridge_state.pkl`.
- p50 mode fails loudly if the ridge scorer is unavailable.

Interpretation: the dataset is small enough for fast sweeps and debugging. It is not a final broad benchmark. It is currently best used as a reward-shaping and model-selection taskset before scaling SFT/RL data.

### Config State

- Primary smoke model: `Qwen/Qwen3.5-0.8B`.
- Primary target model: `Qwen/Qwen3.5-2B`.
- Boundary target: `Qwen/Qwen3.5-4B`.
- Larger MoE candidate: `Qwen/Qwen3.6-35B-A3B`.
- Fallback: `meta-llama/Llama-3.2-3B-Instruct`.
- Active Prime configs live in `configs/prime/`.
- All active Prime configs now pin env `0.3.14`.
- Env `0.3.14` is published and local Prime eval smoke passed. Hosted
  training is still gated on verifier approval for docs-debug `r9`.
- All active train/eval sampling blocks use `max_tokens = 4096`.
- Qwen configs set `enable_thinking = false`.
- W&B project: `humanize-rl`.
- No local command-wrapper config is present.
- W&B token is not committed; it is supplied from `/private/tmp/humanize_rl_prime_wandb.env`.

### Experiment Ledger

| Env | Run / Artifact | Result | Decision |
|---|---|---|---|
| `0.3.5/0.3.6` | `beixwu41osp530um7bfabn8k` | Stopped at step 21. Reward rose to about `0.875` while truncation reached about `56%` and repetition about `50%`. | Repetition/length had to live inside deterministic p50, not only as external penalties. |
| `0.3.6` | `xi7nu5o2yu8761wuvxv0wjom` | One-step technical pass, but romance task `rl_v03_000392` still rewarded unsuitable recommendations around `0.68-0.72`. | Blocked scale-up. Add recommendation suitability cap. |
| `0.3.7` | `z3kf1g2nesrhc2xunybpi4mf`, W&B `uahdkmzb` | Train step reward `0.6058`, eval around `0.761`. No runaway. User audit found emoji, fake warmth, unsuitable films, and signoff still rewarded around `0.764`. | Blocked 50-step. Add emoji/hashtag/all-caps/signoff diagnostics. |
| `0.3.8` | `jfwfr98t1ncdeiab17w5rnss`, W&B `7b7dkxk2` | Train reward `0.5177`; no truncation. Highest-reward samples still used letter openings/signatures like `Dear William Brown... Best regards...`. | Blocked 50-step. Add salutation and stronger signature caps. |
| `0.3.9` | `kalrsxgjs04b48eoxs6k1cik`, W&B `3dk0pvf5` | One-step clean technically. Formal `Dear`/signature samples scored low; highest rewards were not the formal email failures. | Approved only for 50-step 0.8B smoke. |
| `0.3.9` | `d8gyr6h2340a4zkzfcrp5mkw`, W&B `wynbfzzl` | 50-step completed. p50 eval improved `0.6958 -> 0.9205`; strict v02 improved `-0.2203 -> 0.1984`; strict v03 improved `-0.2157 -> 0.2192`. Final p50 samples exposed high-reward hallucinations around `0.982-0.986`. | Do not run 2B/4B. Add semantic caps before further training. |
| `0.3.10` | local rescore of `wynbfzzl` failures | Old hallucination rewards `0.982-0.986` dropped to about `0.496-0.500`; deterministic contribution became `0.0`. | Correct direction, but local eval exposed noisy required facts from prompt scaffold. |
| `0.3.11` | published env, local eval | Scaffold filtering fixed `Dear User`, `Rewrite`, `Slack` false facts. But `We are writing to inform...` still scored `0.807` because `ai_tell_phrase` was diagnostic only. | Do not train. Put AI-tell phrase inside surface deterministic cap. |
| `0.3.12` | local eval `env0312` | Rewards `[0.507, 0.187, 0.861]`. `We are writing to inform...` dropped to `0.507`. Diagnostic rescore found false `missing_entity` from `Compliance Review`, `Firstly`, `Secondly`, `Understanding`. | Do not train. Remove title/discourse false entities. |
| `0.3.13` | published env SHA `5defcc...76f` and local eval `env0313` | Rewards `[0.507, 0.387, 0.888]`, no truncation, avg output tokens `49.667`. Bad phrase capped, formal long email low, direct hotfix clean/high. | Await verifier approval for hosted one-step `r8` only. |
| `0.3.13` | hosted one-step `wckm7qbo1r8oc8ipo3b77xum` / W&B `prime-qwen35-08b-p5050-docs-debug-r8` | Completed technically, but rollout audit failed scientifically. Prime-side reward mean/min/max was `0.645831` / `0.360613` / `0.881603`; `7/16` samples scored `>=0.75`. High-reward defects included instruction leakage, closing-line leakage, missing required phrases, weak exact-constraint handling, and too-short creative outputs. | Blocked 50-step. Patch reward before publishing a new env. |
| `0.3.14` candidate | local rescore of the same `r8` rollouts with patched source | Current reward mean/min/max is `0.408088` / `0.266424` / `0.499395`; `0/16` samples score `>=0.75`; all `7` formerly high samples fall below `0.75`. Problem `404`: old `[0.802, 0.773, 0.476, 0.490]`, new `[0.311, 0.473, 0.476, 0.290]`. Problem `520`: old `[0.716, 0.751, 0.721, 0.422]`, new `[0.266, 0.292, 0.341, 0.422]`. Problem `618`: old `[0.875, 0.863, 0.869, 0.882]`, new `[0.494, 0.482, 0.488, 0.499]`. | Direction is correct. Publish as `0.3.14`, rerun local eval, then ask verifier for hosted one-step `r9`. |
| `0.3.14` | published env SHA `66b332...cad9` and local eval `env0314` | Rewards `[0.507, 0.287, 0.861]`, avg `0.551`, no errors, no truncation, avg output tokens `49.667`. Row 16 dropped from `0.387` to `0.287` because `too_long` now triggers the length cap. | Ask verifier to approve hosted docs-debug `r9` only. |
| `0.3.14` | hosted one-step `l4zww8v68f4h7rigc48rk9e3`, W&B `d4le5ba5`, docs-debug `r9` | Train reward mean `0.440` on logged samples; `2/16` samples scored `>=0.75` and both had no failed diagnostics. Old bad letter/wrapper/format failures stayed low. But hosted train error was `29.3%` from `BadRequestError: Out of range float values are not JSON compliant: nan`, and hosted eval truncation was `100%` at step 0 and `33.3%` at step 1 because eval completions hit `4096`. | Reward gate improved, infra/config gate failed. Do not launch 50-step. Move hosted eval temp from `0.0` to `0.2` and rerun one-step `r10`. |

### What Changed in Reward

The reward started as a 50/50 ridge/deterministic blend, but p50 ignored additive penalties. That made diagnostics visible without stopping high-ridge bad samples. The fix was not to abandon p50; the fix was to move critical failure modes inside deterministic scoring.

Added or strengthened deterministic failures:

- repetition and target-length caps;
- unsuitable uplifting-romance recommendation cap;
- emoji, hashtag, and all-caps diagnostics;
- title-like subject heading detection;
- multiline and inline signoff detection;
- formal salutation detection;
- negative-instruction parsing, so `MUST_NOT: Use emoji...` does not authorize emoji;
- semantic zero-cap for `missing_required_fact`, `forbidden_fact`, `missing_number`, `missing_entity`, `invented_detail`;
- invented number and invented temporal detail diagnostics;
- unsupported hostile/contradictory phrases such as threats/bribes/refunds/deletions not present in source;
- low source-overlap diagnostic for long-source rewrites;
- AI-tell phrase cap for phrases such as `we are writing to`;
- scaffold filtering for subject lines, salutations, prompt terms, and title/discourse false entities;
- instruction-leak detection for outputs that talk about `source message`, `your draft`, `the draft`, or output requirements instead of producing the answer;
- expanded closing/signoff detection for end-of-message phrases such as `Thank you for your feedback and collaboration`;
- exact-constraint caps for missing must-include phrases, forbidden phrases, required placeholders, missing contractions, paragraph counts, sentence windows, and any length-window diagnostic.

Why this matters: p50 is intentionally simple, but simple rewards are dangerous if the deterministic half is soft. The deterministic half now acts as a gate for exact failures, while ridge still contributes the learned style/rubric signal.

### Key Observations

1. Repetition and truncation were an optimizer exploit.
   Early reward increased while output quality collapsed. This was not a model-choice problem. It was a reward-shaping problem.

2. Ridge style alone is not a faithfulness guard.
   The `0.3.9` 50-step run improved aggregate p50 and strict evals, but high-scoring final samples invented threats, stories, and missing facts. Ridge liked the surface; deterministic faithfulness was too soft.

3. Additive penalties are insufficient for `p50_50_no_penalty`.
   Because p50 does not subtract penalties, any must-not-learn behavior must reduce deterministic score directly.

4. Prompt/task scaffold can create false negatives.
   Required facts such as `Rewrite`, `Slack`, `Dear User`, `Focus\n\nDear`, and subject-title fragments polluted entity/fact checks. Without filtering, good direct answers were punished for not repeating scaffolding.

5. Surface human-likeness is not enough.
   Outputs with emoji, fake warmth, movie recommendations, or punchy threats can look less "AI" to a shallow style model but fail the actual task.

6. The current env is good enough for a one-step hosted smoke, not yet for full scale.
   `0.3.13` passes local gates, but the next scientific gate is hosted one-step rollout inspection. A 50-step run should not start until those samples are reviewed.

7. Hosted one-step inspection is not optional.
   `0.3.13` looked acceptable on a 3-row local eval, but the 16 hosted train rollouts exposed failures that local eval did not sample: instruction leakage, polite closing leakage, exact-constraint misses, and creative short-output overreward.

8. Creative tasks can hide behind ridge.
   The `rl_v03_000439` poem samples were too short but still scored `0.862-0.882` in hosted `0.3.13`. The post-`r8` length-diagnostic cap drops those samples to about `0.48-0.50`, which is more appropriate for training.

9. Hosted greedy eval is unsafe for Qwen here.
   Local `prime eval run` at `temperature=0.0` stopped normally, but Hosted Training eval at `temperature=0.0` generated to the full `4096` cap. Hosted train sampling at `temperature=0.7` did not truncate. The config fix is to use nonzero hosted eval temperature while keeping the 4096 cap.

### Current Published Env

- Prime env: `jayshah5696/humanize-rl-env@0.3.13`.
- Wheel: `humanize_rl_env-0.3.13-py3-none-any.whl`.
- SHA256: `5defcc96468ce76fdcfa252793b9875952d1f282b53cd46dd2890f6a06abe76f`.
- Local eval path:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env0313/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/2cf64932/results.jsonl`.
- Local eval command used `Qwen/Qwen3.5-0.8B`, `num_examples=3`, `rollouts_per_example=1`, `max_tokens=4096`, `temperature=0.0`.
- Local eval summary:
  - average reward: `0.594`;
  - row rewards: `0.507`, `0.387`, `0.888`;
  - truncation: `0%`;
  - average output tokens: `49.667`.

Row-level audit:

- `rl_v01_000002`, compression/email:
  - output: `We are writing to inform you that server maintenance is delayed. Your services remain unaffected.`
  - reward: `0.50658`
  - failed diagnostic: `ai_tell_phrase = we are writing to`
  - interpretation: still not ideal prose, but no longer high reward.

- `rl_v01_000016`, rewrite_repair/email:
  - output starts with `Dear Ms. Young...`
  - reward: `0.386842`
  - failed diagnostics: `salutation`, `too_long`
  - interpretation: false `missing_entity` is fixed; remaining low score is for real output defects.

- `rl_v01_000020`, compression/slack:
  - output: `Just confirming the hotfix deployed to production is now running smoothly.`
  - reward: `0.887642`
  - failed diagnostics: none
  - interpretation: this is the desired direct-answer behavior.

### Hosted `r8` One-Step Result

- Run: `wckm7qbo1r8oc8ipo3b77xum`.
- Dashboard:
  `https://app.primeintellect.ai/dashboard/training/wckm7qbo1r8oc8ipo3b77xum`.
- Config: `configs/prime/qwen35_08b_docs_debug.toml`.
- Model: `Qwen/Qwen3.5-0.8B`.
- Env: `jayshah5696/humanize-rl-env@0.3.13`.
- Max steps: `1`.
- Batch size: `16`.
- Rollouts per example: `4`.
- Max generation tokens: `4096`.
- W&B project: `jayshah5696/humanize-rl`.
- Raw rollout artifact:
  `runs/prime_training_smoke/wckm7qbo1r8oc8ipo3b77xum/rollouts_step0.json`.

Prime-side rollout metrics:

- rows: `16`;
- reward mean/min/max: `0.645831` / `0.360613` / `0.881603`;
- high reward samples `>=0.75`: `7/16`;
- problem ids inspected: `404`, `520`, `618`, `638`.

Failure analysis:

- `problem_id=404`, task `rl_v03_000009`:
  - high sample `0.802` added `Ensure your draft...` plus `Thank you for your feedback and collaboration`;
  - high sample `0.773` described the source/message instead of directly compressing it;
  - old reward missed instruction leakage and treated the polite closing as acceptable.
- `problem_id=520`, task `rl_v03_000268`:
  - samples around `0.716-0.751` missed exact constraints such as contraction, phrase, forbidden-phrase, length, and paragraph/sentence requirements;
  - conclusion: exact task constraints needed to affect the deterministic half more strongly.
- `problem_id=618`, task `rl_v03_000439`:
  - poem samples scored `0.862-0.882` while failing `too_short`;
  - conclusion: length-window diagnostics were too soft when ridge liked the style.
- `problem_id=638`, task `rl_v03_000480`:
  - bad letter shells were already low (`0.360-0.476`);
  - salutation, signoff, placeholder, wrapper, subject-line, markdown, and bullet diagnostics were doing their job here.

Decision: `r8` completed technically but failed the training gate. It is not a valid precursor for a 50-step run.

### Post-`r8` Patch Audit

Current source changes after the `r8` audit:

- `instruction_leak` diagnostic added and included in format/task-following caps;
- inline thank-you feedback/collaboration closings treated as signoffs;
- missing must-include phrases, forbidden phrases, required placeholders, missing contractions, paragraph counts, and sentence windows moved into hard deterministic caps;
- any explicit length diagnostic (`too_long`, `too_short`, `sentence_count`) now triggers the deterministic length cap, not only a zero length score.

Local rescore of the exact saved `r8` rollouts with current source:

- rows: `16`;
- old Prime reward mean/min/max: `0.645831` / `0.360613` / `0.881603`;
- current reward mean/min/max: `0.408088` / `0.266424` / `0.499395`;
- average reward delta: `-0.237744`;
- current high reward samples `>=0.75`: `0/16`;
- formerly high samples `>=0.75` that now fall below `0.75`: `7/7`.

Current diagnostic counts on the same rollouts:

- `sentence_window`: `8`;
- `signoff`: `7`;
- `too_short`: `7`;
- `missing_contraction`: `4`;
- `missing_must_include_phrase`: `3`;
- `placeholder_disallowed`: `3`;
- `salutation`: `3`;
- `ai_tell_phrase`: `2`;
- `instruction_leak`: `2`;
- singletons: `wrapper_phrase`, `subject_line`, `wrong_format_markdown`, `wrong_format_bullets`, `too_long`, `forbidden_phrase`, `paragraph_count`, `wrong_format_heading`.

Problem-level old/new rewards:

- `404`: `[0.802, 0.773, 0.476, 0.490] -> [0.311, 0.473, 0.476, 0.290]`;
- `520`: `[0.716, 0.751, 0.721, 0.422] -> [0.266, 0.292, 0.341, 0.422]`;
- `618`: `[0.875, 0.863, 0.869, 0.882] -> [0.494, 0.482, 0.488, 0.499]`;
- `638`: `[0.470, 0.476, 0.388, 0.361] -> [0.470, 0.476, 0.388, 0.361]`.

Interpretation: the patch fixes the observed reward hole without changing the already-low bad letter-shell rewards.

### Current Published Env `0.3.14`

- Prime env: `jayshah5696/humanize-rl-env@0.3.14`.
- Wheel: `humanize_rl_env-0.3.14-py3-none-any.whl`.
- SHA256: `66b332bbfa615d8c0fa78ce3041269da9f684d99ff0d7c6682ab6864f744cad9`.
- Metadata:
  `environments/humanize_rl_env/.prime/.env-metadata.json`.
- Local eval path:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env0314/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/c2e718a9/results.jsonl`.
- Local eval command used `Qwen/Qwen3.5-0.8B`, `num_examples=3`, `rollouts_per_example=1`, `max_tokens=4096`, `temperature=0.0`.
- Local eval summary:
  - average reward: `0.551`;
  - row rewards: `0.507`, `0.287`, `0.861`;
  - truncation: `0%`;
  - errors: `0%`;
  - average output tokens: `49.667`.

Row-level audit:

- `rl_v01_000002`, compression/email:
  - output: `We are writing to inform you that server maintenance is delayed. Your services remain unaffected.`
  - reward: `0.50658`
  - failed diagnostic: `ai_tell_phrase`
  - interpretation: still bad style, still capped.
- `rl_v01_000016`, rewrite_repair/email:
  - output starts with `Dear Ms. Young...`
  - reward: `0.286842`
  - failed diagnostics: `salutation`, `too_long`
  - interpretation: lower than `0.3.13` because the explicit length diagnostic now triggers the length cap.
- `rl_v01_000020`, compression/slack:
  - output: `Just confirming the hotfix is live and everything is running fine.`
  - reward: `0.860998`
  - failed diagnostics: none
  - interpretation: desired direct-answer behavior remains high.

### Verification Status

- Focused reward/env tests:
  `71 passed, 2 skipped`.
- Ruff on touched reward files:
  passed.
- Secret check:
  W&B token not found in repo text.
- Config check:
  active configs pin `0.3.14`, use 4096 generation tokens, disable Qwen thinking, and contain no local command-wrapper usage.
  Hosted eval sampling now uses `temperature=0.2` after `r9` showed greedy Qwen eval truncation at `4096`.
- Verifier handoff:
  approved hosted docs-debug `r9`; `r9` then failed the infra/config gate. The config-only `r10` change is waiting for verifier approval.

### Current Decision

Do not start a 50-step run yet.

Next serialized action:

1. Verify the config-only hosted eval-temperature change.
2. Launch hosted one-step `r10` with W&B enabled.
3. Inspect hosted rollouts, W&B curves, error rate, eval truncation, directness, semantic preservation, and option/wrapper rates.
4. If `r10` has train error below `10%`, eval truncation below `10%`, no high-reward bad samples, and no length collapse, launch the 50-step `Qwen/Qwen3.5-0.8B` smoke.
5. Only after that 50-step smoke passes qualitative and metric gates, run the `Qwen/Qwen3.5-2B` target.

Open research risk:

- The taskset is still only 972 rows. It is enough to test reward shape and model behavior, but a proper SFT/RL model will need a broader, cleaner supervised corpus plus frozen blind eval prompts.
- p50 is now less noisy, but ridge can still overvalue surface polish. Final model selection must include blind A/B samples, strict evals, family-level deltas, wrapper/options rate, and length-collapse checks.

### Hosted One-Step Docs-Debug `r10`

Run facts:

- Run: `h9r8y6sj9synobs19kmu9zc2`
- Prime URL:
  `https://app.primeintellect.ai/dashboard/training/h9r8y6sj9synobs19kmu9zc2`
- W&B run name: `prime-qwen35-08b-p5050-docs-debug-r10`
- W&B run id: `3vjqmco2`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Env: `jayshah5696/humanize-rl-env@0.3.14`
- Status: `COMPLETED`
- Started/completed: `2026-06-21 04:52:16 UTC` / `2026-06-21 04:54:10 UTC`
- Usage: `54,449` total tokens, `$0.0026`
- Checkpoint: `eizq3dbj2m99bjt6tkgfp8gt`, step `1`, still reported
  `UPLOADING` when checked.

Training metrics:

- Train sample count: `16`
- Prime train reward mean/min/max by logged sample audit:
  `0.4393588446` / `0.2504880428` / `0.7311608437`
- Prime aggregate reward by problem mean/min/max:
  `0.4393588446` / `0.3418707205` / `0.4895065989`
- Train truncation: `0%`
- Train error/cancel rate: `0%`
- Decode length mean/max: `172.75` / `254.25`
- Filters: `0` for `gibberish`, `repetition`, and `zero_advantage`

Hosted eval remained invalid:

- Step `0` eval completion length mean/min/max: `4096/4096/4096`
- Step `0` eval truncation: `100%`
- Step `1` eval completion length mean/min/max: `4096/4096/4096`
- Step `1` eval truncation: `100%`
- Changing hosted eval temperature from `0.0` to `0.2` did not fix Qwen hosted
  eval runaway. This is isolated to hosted eval: train generation did not
  truncate.

Saved artifacts:

- Rollouts:
  `runs/prime_training_smoke/h9r8y6sj9synobs19kmu9zc2/rollouts_step0.json`
- Reproducible local audit:
  `runs/prime_training_smoke/h9r8y6sj9synobs19kmu9zc2/audit_step0.json`
- Audit helper:
  `scripts/eval/audit_prime_rollouts.py`

Local rollout audit against current source:

- Rows matched: `16/16`
- Recomputed reward mean/min/max:
  `0.4393588446` / `0.2504880428` / `0.7311608437`
- High reward samples `>=0.75`: `0`
- High reward samples with failed diagnostics: `0`
- High reward samples with emoji: `0`
- High reward samples with all-caps: `0`
- High reward samples with option/wrapper behavior: `0`
- Failed diagnostic counts:
  - `sentence_window`: `10`
  - `signoff`: `6`
  - `too_short`: `6`
  - `missing_contraction`: `4`
  - `ai_tell_phrase`: `3`
  - `paragraph_count`: `3`
  - `salutation`: `3`
  - `forbidden_phrase`: `2`
  - `placeholder_disallowed`: `2`
  - `missing_must_include_phrase`: `1`

Interpretation:

- `0.3.14` fixed the earlier high-reward failure class. The r10 train batch
  has ugly mid-reward samples, but the caps keep them below the acceptance band.
- Emoji and all-caps are already deterministic diagnostics and p50 surface caps.
  The new audit report explicitly tracks high-reward emoji/all-caps regressions
  so the next gate catches them before scaling to 2B.
- The hosted eval subsystem is not reliable for Qwen here. A 50-step smoke can
  proceed only as a train-only Prime run, with post-run checkpoint evaluation
  handled separately.

Verification after adding the audit helper and train-only config:

- Focused reward/env/script tests:
  `72 passed, 2 skipped`.
- Ruff:
  passed for `scripts/eval/audit_prime_rollouts.py` and
  `tests/scripts/test_audit_prime_rollouts.py`.
- Config parse:
  `configs/prime/qwen35_08b_debug.toml` has `max_steps=50`, `batch_size=128`,
  `rollouts_per_example=8`, `max_tokens=4096`, train temperature `0.7`,
  `enable_thinking=false`, env `0.3.14`, and no `[eval]` or `[val]` section.
- Secret/config check:
  no committed W&B token in touched files, no local command-wrapper usage in the train-only config.
- One caveat:
  `uv run scripts/eval/audit_prime_rollouts.py` tried to build project
  dependency `fasttext-wheel` and failed against the local Xcode SDK. The audit
  was rerun successfully with `uv run --no-project` and explicit lightweight
  deps (`pydantic`, `numpy`, `scikit-learn`, `click`). This does not affect
  Prime hosted training.

### Current Decision After `r10`

Proceed to the 50-step `Qwen/Qwen3.5-0.8B` Prime RL smoke as a train-only run:

- Config: `configs/prime/qwen35_08b_debug.toml`
- Run name: `humanize-p5050-qwen35-08b-trainonly50-env0314-r1`
- W&B name: `prime-qwen35-08b-p5050-trainonly50-env0314-r1`
- Train env args:
  `split=train`, `task_set=mix_v2_p5050`, `reward_mode=p50_50_no_penalty`
- Stop/gate target:
  run 50 steps once, then audit train rollouts and evaluate the checkpoint
  separately. Do not launch 2B until the 0.8B checkpoint clears reward,
  truncation, option/wrapper, emoji/all-caps, length-collapse, and qualitative
  direct-answer gates.
- Verifier status:
  first verifier `019eed8f-6e2f-7140-bf55-0403ce9ee74e` was interrupted after
  timing out and returned no approval evidence or concrete findings. A narrower
  verifier was started for the train-only config and audit helper before launch.

### 50-Step 0.8B Train-Only Smoke Launched

- Run: `dlh5ecoj1niiznfu8qp6mbxn`
- Prime URL:
  `https://app.primeintellect.ai/dashboard/training/dlh5ecoj1niiznfu8qp6mbxn`
- Config: `configs/prime/qwen35_08b_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Env: `jayshah5696/humanize-rl-env@0.3.14`
- Env args:
  `split=train`, `task_set=mix_v2_p5050`, `reward_mode=p50_50_no_penalty`
- Run name: `humanize-p5050-qwen35-08b-trainonly50-env0314-r1`
- W&B name: `prime-qwen35-08b-p5050-trainonly50-env0314-r1`
- Created: `2026-06-22 04:30:27 UTC`
- Initial status: `PENDING`
- Started: `2026-06-22 04:31:10 UTC`
- Max steps: `50`
- Batch size: `128`
- Rollouts per example: `8`
- Max generation tokens: `4096`
- Train sampling temperature: `0.7`
- Qwen thinking: disabled
- Eval config: absent
- Validation config: absent
- W&B secret source: `/private/tmp/humanize_rl_prime_wandb.env`, key name only
  shown by CLI as `WANDB_API_KEY`
- Verifier:
  second verifier `019eed96-361c-7780-8625-ecb984e90fd5` returned
  `APPROVED`; config assertions, secret/wrapper/eval-val grep, focused pytest,
  and ruff all passed.
- First running poll:
  - status moved to `RUNNING`;
  - W&B run id: `fky3hckx`;
  - W&B URL: `https://wandb.ai/jayshah5696/humanize-rl/runs/fky3hckx`;
  - orchestrator loaded env `0.3.14`;
  - train environments ready;
  - student inference pool ready;
  - orchestrator loop started with `max_steps=50`;
  - Prime log showed `128 inflight rollouts (train=128, eval=0)`, confirming
    hosted eval is absent for this smoke.
- Early infra warning:
  - before step `0` metrics were available, the renderer/API emitted repeated
    `BadRequestError: Out of range float values are not JSON compliant: nan`;
  - progress still moved from `Train batch 0/128` to `26/128`, so the run was
    not stopped immediately;
  - decision: keep monitoring until step `0` either completes with measurable
    error/filter/truncation metrics or the run fails/stalls.

Next monitoring steps:

1. Poll `prime train progress/get/logs/metrics`.
2. Save step rollouts when available.
3. Run `scripts/eval/audit_prime_rollouts.py` on saved rollouts.
4. Check checkpoints and deploy/evaluate only if the training smoke completes
   cleanly.

### 50-Step Train-Only Smoke `r1` Stopped Before Step 0

- Run: `dlh5ecoj1niiznfu8qp6mbxn`
- Status: stopped manually after startup stalled before any optimizer step.
- Reason:
  - repeated renderer/API failures:
    `BadRequestError: Out of range float values are not JSON compliant: nan`;
  - progress reached `Train batch 26/128` but did not advance to step `0`;
  - buffered rollouts grew past the batch size, suggesting the high-concurrency
    batch was not forming a usable train step.
- Eval status: still absent; logs always showed `eval=0`.
- Usage before stop:
  - training tokens: `0`;
  - inference tokens: `23,231`;
  - cost: `$0.0006`.
- Final status check:
  `STOPPED`, completed at `2026-06-22 04:37:28 UTC`.
- Decision:
  - this is an infrastructure/concurrency failure, not a reward or model-choice
    result;
  - relaunch the same 50-step smoke with constrained inflight rollouts so Prime
    does not run `128` simultaneous generations against the Qwen renderer.

### 50-Step Train-Only Smoke `r2` Config

- Config: `configs/prime/qwen35_08b_debug.toml`
- Run name: `humanize-p5050-qwen35-08b-trainonly50-env0314-r2`
- W&B name: `prime-qwen35-08b-p5050-trainonly50-env0314-r2`
- Model/env/reward:
  `meta-llama/Llama-3.2-3B-Instruct`,
  `jayshah5696/humanize-rl-env@0.3.14`,
  `p50_50_no_penalty`
- Batch/step shape unchanged:
  `max_steps=50`, `batch_size=128`, `rollouts_per_example=8`
- New infrastructure control:
  `max_inflight_rollouts=16`
- Rationale:
  r10 completed cleanly at small inflight; r1 failed only after the run tried
  to keep `128` train rollouts in flight. Constraining inflight is the minimal
  change that addresses the observed failure while keeping the scientific run
  shape intact.

### 50-Step Train-Only Smoke `r2` Launched

- Run: `e7pie3x7j6uywy5rfwy407ct`
- Prime URL:
  `https://app.primeintellect.ai/dashboard/training/e7pie3x7j6uywy5rfwy407ct`
- Created: `2026-06-22 04:40:39 UTC`
- Status at launch: `PENDING`
- Verified config:
  `max_steps=50`, `batch_size=128`, `rollouts_per_example=8`,
  `max_inflight_rollouts=16`, `max_tokens=4096`, `enable_thinking=false`,
  env `0.3.14`, no eval/val config.
- W&B name: `prime-qwen35-08b-p5050-trainonly50-env0314-r2`
- W&B run id: `m0abpe1v`
- W&B URL: `https://wandb.ai/jayshah5696/humanize-rl/runs/m0abpe1v`
- Verifier:
  `019eed9f-7996-7730-8575-726a80c9337f` returned `APPROVED` for the r2
  concurrency-only change.
- First poll:
  status moved to `RUNNING`; env `0.3.14` installed; W&B initialized; training
  env loading started. Waiting for first train-loop lines to confirm inflight
  cap and then step `0` metrics.
- Train-loop confirmation:
  - `16 inflight rollouts (train=16, eval=0)` confirms the cap is active;
  - buffer reached `+26` rollouts by `04:42:17 UTC`;
  - no step metrics yet.

### 50-Step Train-Only Smoke `r2` Stopped Before Step 0

- Run: `e7pie3x7j6uywy5rfwy407ct`
- Final status: `STOPPED`
- Reason:
  - despite `max_inflight_rollouts=16`, the run stayed at `Train batch 0/128`
    with `+26 buffered` from about `04:42:17` through `04:45:18 UTC`;
  - no step metrics or samples were published;
  - training tokens stayed `0`;
  - total cost was `$0.0005`.
- Interpretation:
  - `r1` proved `128` simultaneous rollouts is too high;
  - `r2` proved a `128` train batch can still stall even with lower inflight;
  - this is a run-geometry/infrastructure issue. It does not invalidate env
    `0.3.14` or the reward audit.

### 50-Step Simple Smoke `r3` Config

- Config: `configs/prime/qwen35_08b_debug.toml`
- Run name: `humanize-p5050-qwen35-08b-simple50-env0314-r3`
- W&B name: `prime-qwen35-08b-p5050-simple50-env0314-r3`
- W&B run id: `t8m1ha11`
- W&B URL: `https://wandb.ai/jayshah5696/humanize-rl/runs/t8m1ha11`
- First poll:
  status moved to `RUNNING`; env `0.3.14` installed; train env ready; student
  inference pool ready; orchestrator loop started with `max_steps=50`. Waiting
  for first train batch/step metrics.
- Model/env/reward:
  `meta-llama/Llama-3.2-3B-Instruct`, `jayshah5696/humanize-rl-env@0.3.14`,
  `p50_50_no_penalty`
- Training shape:
  `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
  `max_inflight_rollouts=16`, `max_tokens=4096`
- Rationale:
  this copies the r10 train geometry that completed with `0%` train errors and
  `0%` train truncation, removes hosted eval, and extends it to 50 steps. This
  is now the correct simple full-smoke run before any 2B target.

### 50-Step Simple Smoke `r3` Launched

- Run: `ee1tk85aludc6606vqv5y2bs`
- Prime URL:
  `https://app.primeintellect.ai/dashboard/training/ee1tk85aludc6606vqv5y2bs`
- Created: `2026-06-22 04:46:46 UTC`
- Status at launch: `PENDING`
- Config:
  `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
  `max_inflight_rollouts=16`, `max_tokens=4096`, no eval/val.
- W&B name: `prime-qwen35-08b-p5050-simple50-env0314-r3`

### 50-Step Simple Smoke `r3` Stopped Before Step 0

- Run: `ee1tk85aludc6606vqv5y2bs`
- Final status: `STOPPED`
- Reason:
  - the run stayed at `Train batch 0/16`, `16 inflight rollouts`, and no
    buffered completions for the first sampled rollout set;
  - no step metrics or samples were published;
  - training tokens stayed `0`;
  - total cost was `$0.0001`.
- Taskset length check:
  - rows: `972`;
  - max `target_words`: `1350`;
  - target words over `512`: `27` tasks;
  - target words over `256`: `169` tasks;
  - `max_words` hard cap max: `100`.
- Interpretation:
  - Prime accepts `4096` as a config value, but the Qwen 0.8B renderer path is
    unstable for training with `4096` on this task mix;
  - keeping `4096` blocks even the simple smoke before step `0`;
  - the next simple run uses `1024` generation tokens. This is a practical cap,
    not a reward/model change. It still covers almost all expected outputs and
    is enough to train direct rewrites while avoiding indefinite generation.

### 50-Step Simple Smoke `r4` Config

- Config: `configs/prime/qwen35_08b_debug.toml`
- Run name: `humanize-p5050-qwen35-08b-simple50-env0314-r4`
- W&B name: `prime-qwen35-08b-p5050-simple50-env0314-r4`
- W&B run id: `y47njq5q`
- W&B URL: `https://wandb.ai/jayshah5696/humanize-rl/runs/y47njq5q`
- First poll:
  status moved to `RUNNING`; env `0.3.14` installed; train env ready; student
  inference pool ready; orchestrator loop started; first train-loop line was
  `Train batch 0/16`, `16 inflight rollouts (train=16, eval=0)`.
- First progress window:
  - batch advanced to `Train batch 5/16 (31.2%)`;
  - buffered samples reached `+23`;
  - hosted eval remained absent (`eval=0`);
  - repeated Qwen renderer/API `nan` failures still occurred, but unlike r1-r3
    the run is making train-batch progress. Continue until step `0` metrics are
    available, then gate on measured error/truncation/filter rates.

### 50-Step Simple Smoke `r4` Stopped After Step 0

- Run: `tbllkv5dayyodn2i8w00v93u`
- Final status: `STOPPED`
- Reason:
  step `0` completed but failed the infra gate due to renderer/API error rate.
- Step `0` metrics:
  - reward mean/min/max: `0.470368` / `0.267991` / `0.915446`
  - trainable samples: `11/16` (`68.8%`)
  - error rate: `81.3%`
  - truncation: `0%`
  - decode length mean/max: `76.91` / `246`
  - filters: `31.25%` zero-advantage, `0%` gibberish, `0%` repetition
  - step time: `252.7s`
- Usage:
  - training tokens: `5,558`
  - inference tokens: `7,549`
  - total cost: `$0.0003`
- Saved artifacts:
  - rollouts:
    `runs/prime_training_smoke/tbllkv5dayyodn2i8w00v93u/rollouts_step0.json`
  - audit:
    `runs/prime_training_smoke/tbllkv5dayyodn2i8w00v93u/audit_step0.json`
- Local audit:
  - matched rows: `16/16`
  - recomputed reward mean/min/max:
    `0.485123` / `0.263919` / `0.941314`
  - high reward with failed diagnostics: `0`
  - high reward with emoji/all-caps/options-or-wrapper: `0/0/0`
  - failed counts: `too_short=5`, `missing_entity=4`,
    `missing_must_include_phrase=4`, `placeholder_disallowed=3`,
    `em_dash=2`, `forbidden_phrase=2`, and one each of `ai_tell_phrase`,
    `missing_required_fact`, `sentence_window`, `subject_line`,
    `wrong_format_markdown`.
- Interpretation:
  reward behavior is acceptable on the saved samples; the Qwen renderer path is
  the blocker. Further Qwen geometry changes are not the right next move.

### Llama 3.2 3B Fallback Simple Smoke Config

- Config: `configs/prime/llama32_3b_debug.toml`
- Run name: `humanize-p5050-llama32-3b-simple50-env0314-r1`
- W&B name: `prime-llama32-3b-p5050-simple50-env0314-r1`
- W&B run id: `lvqx5v5g`
- W&B URL: `https://wandb.ai/jayshah5696/humanize-rl/runs/lvqx5v5g`
- First poll:
  - status moved to `RUNNING`;
  - env `0.3.14` installed;
  - renderer: `Llama3Renderer`;
  - hosted eval absent (`eval=0`);
  - step `0`: reward `0.3798`, trainable `16/16`, error `0%`,
    truncation `0%`;
  - step `1`: reward `0.3002`, trainable `16/16`, error `0%`,
    truncation `0%`;
  - step `2`: reward `0.4825`, trainable `16/16`, error `0%`,
    truncation `0%`.
- Interpretation:
  Llama is the first clean Prime train path. Continue to 50 steps, then audit
  rollouts and checkpoint. Qwen is deprioritized as a backend/renderer failure
  until Prime/Qwen NaN behavior is understood.
- Model: `meta-llama/Llama-3.2-3B-Instruct`
- Env/reward:
  `jayshah5696/humanize-rl-env@0.3.14`, `p50_50_no_penalty`
- Training shape:
  `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
  `max_inflight_rollouts=16`, `max_tokens=1024`
- Rationale:
  this is the planned fallback target after Qwen failed at the renderer/API
  layer. The env, reward, taskset, W&B tracking, and simple train-only geometry
  stay fixed so the variable being tested is the trainable model/backend.

### Llama 3.2 3B Fallback Simple Smoke Launched

- Run: `q6lsjfaxocod6xt4e2qebjgv`
- Prime URL:
  `https://app.primeintellect.ai/dashboard/training/q6lsjfaxocod6xt4e2qebjgv`
- Created: `2026-06-22 05:00:12 UTC`
- Status at launch: `PENDING`
- Cluster: `udqkwxkaf6b0iy0v8oh4flkw`
- Config:
  `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
  `max_inflight_rollouts=16`, `max_tokens=1024`, no eval/val.
- W&B name: `prime-llama32-3b-p5050-simple50-env0314-r1`
- Model/env/reward unchanged:
  `Qwen/Qwen3.5-0.8B`, `jayshah5696/humanize-rl-env@0.3.14`,
  `p50_50_no_penalty`
- Training shape:
  `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
  `max_inflight_rollouts=16`, `max_tokens=1024`
- Rationale:
  `4096` was tested and documented as unstable for the Prime/Qwen training
  renderer here. `1024` is the simplest next cap that should let the smoke
  produce actual optimizer steps before we decide whether to restore a higher
  cap for a narrower long-form taskset.

### Llama 3.2 3B Fallback Simple Smoke Progress

- Run: `q6lsjfaxocod6xt4e2qebjgv`
- Status poll:
  `RUNNING`, started `2026-06-22 05:00:58 UTC`, updated
  `2026-06-22 05:04:17 UTC`.
- Prime run metadata:
  - base model: `meta-llama/Llama-3.2-3B-Instruct`
  - renderer observed in logs: `Llama3Renderer`
  - env: `jayshah5696/humanize-rl-env@0.3.14`
  - task args:
    `split=train`, `task_set=mix_v2_p5050`,
    `reward_mode=p50_50_no_penalty`
  - `max_steps=50`, `batch_size=16`, `rollouts_per_example=4`,
    `max_inflight_rollouts=16`, `max_tokens=1024`
  - hosted eval/validation absent by design for this smoke.
- Step evidence through step `14`:
  - every logged step so far has `Trainable 16/16`, `Error 0.0%`,
    `Truncation 0.0%`;
  - rewards:
    step `0` `0.3798`, step `1` `0.3002`, step `2` `0.4825`,
    step `3` `0.4765`, step `4` `0.3029`, step `5` `0.4890`,
    step `6` `0.3633`, step `7` `0.3421`, step `8` `0.2756`,
    step `9` `0.3585`, step `10` `0.3283`, step `11` `0.2703`,
    step `12` `0.3217`, step `13` `0.3641`, step `14` `0.2975`;
  - Prime sample tables are available for steps `0` and `10`.
- Interpretation:
  Llama is currently the first Prime backend that actually trains cleanly on
  this env. Reward is not monotonically improving yet, but that is expected in a
  50-step smoke with tiny batches. The gate is still technical completion plus
  rollout audit, not reward-curve optimism.

### Llama 3.2 3B Fallback Rollout Audit Through Step 30

- Saved rollout artifacts:
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/rollouts_step0.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/rollouts_step10.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/rollouts_step20.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/rollouts_step30.json`
- Saved audit artifacts:
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/audit_step0.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/audit_step10.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/audit_step20.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/audit_step30.json`
- Local audit command shape:
  `PYTHONPATH=src uv run --no-project --with pydantic --with numpy --with scikit-learn --with click python scripts/eval/audit_prime_rollouts.py ...`
- Audit trend:
  - step `0`: mean/min/max `0.379800` / `0.190798` / `0.787307`
  - step `10`: mean/min/max `0.328307` / `0.124285` / `0.802346`
  - step `20`: mean/min/max `0.354857` / `0.148283` / `0.590346`
  - step `30`: mean/min/max `0.357433` / `0.147860` / `0.654141`
- Prime/local alignment:
  recomputed means exactly matched the Prime sample reward means for all four
  audited sample tables.
- High-reward defect check:
  across audited steps `0`, `10`, `20`, and `30`, high-reward threshold `0.75`
  had `0` high-reward samples with failed diagnostics, emoji, all-caps, or
  option/wrapper behavior.
- Current weak checks by step:
  - step `0`:
    `missing_must_include_phrase=4`, `sentence_window=4`, `signoff=4`,
    `subject_line=4`, `too_long=4`, `too_short=4`,
    `unsuitable_recommendation=4`, `wrapper_phrase=4`
  - step `10`:
    `placeholder_disallowed=8`, `wrong_format_markdown=7`,
    `missing_must_include_phrase=5`, plus length/signoff/subject issues
  - step `20`:
    `too_long=8`, `unsupported_negation=5`, `missing_entity=4`,
    `missing_required_fact=4`, `ai_tell_phrase=4`
  - step `30`:
    `missing_entity=8`, `placeholder_disallowed=8`,
    `missing_must_include_phrase=4`, `missing_required_fact=4`,
    `low_source_overlap=3`
- Qualitative read:
  the best step `20` sample scored only `0.590345` and still hit
  `ai_tell_phrase`; other top samples had invented details, repetition, or
  short/coverage failures. This is scientifically acceptable for the smoke
  because the reward is not treating flawed behavior as excellent, but it is not
  evidence of a useful final model.
- Decision:
  continue the 50-step run to completion. Do not start a larger run until final
  samples, checkpoint status, and post-run eval are checked.

### Llama 3.2 3B Fallback 50-Step Smoke Completed

- Run: `q6lsjfaxocod6xt4e2qebjgv`
- Final status:
  `COMPLETED`, started `2026-06-22 05:00:58 UTC`, completed
  `2026-06-22 05:11:24 UTC`, no run error message.
- W&B:
  `https://wandb.ai/jayshah5696/humanize-rl/runs/lvqx5v5g`
- Saved final artifacts:
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/run_get.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/usage.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/metrics_0_50.json`
  - `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/checkpoints.json`
- Usage:
  - training tokens: `366,097`, cost `$0.0548`
  - inference tokens: `376,975`, cost `$0.0328`
  - total tokens: `743,072`, total cost `$0.0876`
- Final checkpoint:
  `jtvwt4qjt9nf3aydt1b0x0u7`, step `50`, status `UPLOADING` at first
  post-run checkpoint poll.
- Metrics file summary:
  - metric rows: `50`, steps `0..49`
  - reward step `0`: `0.379800`
  - reward step `49`: `0.481414`
  - reward min/max over steps: `0.270295` / `0.535724`
  - reward mean over steps: `0.387821`
  - max truncation mean: `0.0`
  - max filter rate: `0.0`
  - max zero-advantage filter rate: `0.0`
  - decode mean step `0`: `147.625`
  - decode mean step `49`: `79.6875`
  - decode max over steps: `837`
  - final style/task/faithfulness/length:
    `0.870405` / `0.985119` / `0.866071` / `0.916791`
- Step `40` audit:
  - saved:
    `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/audit_step40.json`
  - mean/min/max:
    `0.386115` / `0.233678` / `0.713159`
  - high-reward failures at threshold `0.75`:
    failed diagnostics `0`, emoji `0`, all-caps `0`, option/wrapper `0`
  - main failed checks:
    `missing_entity=12`, `missing_required_fact=11`,
    `placeholder_disallowed=4`, `missing_must_include_phrase=3`,
    `signoff=2`, and one each of `invented_temporal_detail`,
    `missing_contraction`, `too_short`.
  - best sample:
    `0.713159`, no failed diagnostics, but still awkward around quoted
    `"sense of urgency"`. This confirms the high-reward gate is stricter than
    the visible-quality bar we want.
- Scientific decision:
  Llama 3.2 3B is the first clean Prime Hosted Training backend for this env.
  The smoke proves the train loop, W&B, env `0.3.14`, taskset, reward, rollout
  export, and local audit path work. It does not yet prove a useful model.
  Next gate is checkpoint deployment plus frozen validation eval against the
  base model.

### Baseline Eval Side Prepared While Checkpoint Uploads

- Reason:
  Prime checkpoint `jtvwt4qjt9nf3aydt1b0x0u7` was still `UPLOADING` after the
  50-step run completed, so post-train deployment/eval could not start yet.
  Baseline evals were run first using the exact env, model family, and `4096`
  eval cap intended for the checkpoint comparison.
- Base model:
  `meta-llama/Llama-3.2-3B-Instruct`
- Eval command shape:
  `prime --plain eval run jayshah5696/humanize-rl-env --provider prime --model meta-llama/Llama-3.2-3B-Instruct --max-tokens 4096 --temperature 0.2 --skip-upload --disable-env-server --disable-tui --max-retries 0 --save-results`
- Saved results:
  - p50 mix validation:
    `runs/prime_eval_smoke/llama32_3b_base_p5050_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct/ca5f6d5e/results.jsonl`
  - strict v02:
    `runs/prime_eval_smoke/llama32_3b_base_v02_strict_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct/014ddd4f/results.jsonl`
  - strict v03:
    `runs/prime_eval_smoke/llama32_3b_base_v03_strict_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct/9507d0b1/results.jsonl`
- Baseline p50 mix validation, `20` examples:
  - reward mean/min/max:
    `0.421325` / `0.192393` / `0.919898`
  - truncation count: `0`
  - average input/output tokens:
    `196.85` / `55.6`
  - wrapper/option averages:
    `0.0` / `0.0`
  - risk average:
    `-0.75`
- Baseline strict `v02_smoke`, `9` validation examples:
  - reward mean/min/max:
    `-0.271042` / `-1.0` / `0.846820`
  - truncation count: `0`
  - average input/output tokens:
    `183.33` / `50.89`
  - wrapper/option averages:
    `0.0` / `0.0`
  - risk average:
    `-0.716667`
- Baseline strict `v03`, `20` examples:
  - reward mean/min/max:
    `-0.225785` / `-1.0` / `0.860753`
  - truncation count: `0`
  - average input/output tokens:
    `271.9` / `158.4`
  - wrapper/option averages:
    `-0.01` / `0.0`
  - risk average:
    `-0.6525`
- Qualitative note:
  base strict `v03` still produced wrapper text such as
  `Here's a compressed version of the source message`, which strict reward
  penalized. This is a useful post-train comparison target.
- Next action:
  when checkpoint `jtvwt4qjt9nf3aydt1b0x0u7` becomes `READY`, deploy it and
  rerun the same three evals against the deployed adapter/model id.

### Llama Adapter Deployment Started

- Prime CLI note:
  installed CLI is `0.6.14`; CLI reports `0.6.15` available. Current plan keeps
  `0.6.14` because it satisfies the `0.6.14+` assumption and the active
  commands are working.
- Deployment list nuance:
  Prime exposes three records for run `q6lsjfaxocod6xt4e2qebjgv`:
  - `apqky15zanvxnyw160ooq5ki`, step `49`, status `UPLOADING`
  - `r5snl7aw0objspezj71ybs3t`, step `50`, status `UPLOADING`
  - `ykqmg5inv09jj3gm2htr7aoz`, step `null`, status `READY`
- Action taken:
  deployed the `READY` same-run model record:
  `ykqmg5inv09jj3gm2htr7aoz`.
- Inference/eval model string from Prime:
  `meta-llama/Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz`
- Deployment status after create:
  `DEPLOYING`.
- Saved deployment snapshot:
  `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/deployments_after_create.json`
- Caution:
  because Prime still reports the explicit step `50` checkpoint as
  `UPLOADING`, post-train evals should be labeled as using the deployed
  same-run adapter id above, not as the explicit step-50 checkpoint, unless
  Prime later maps them or the step-50 record becomes ready.

### Llama Adapter Eval Results

- Deployment result:
  `ykqmg5inv09jj3gm2htr7aoz` reached `DEPLOYED` at
  `2026-06-22 17:47:03 UTC`, with no deployment error.
- Evaluated model:
  `meta-llama/Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz`
- Saved comparison artifact:
  `runs/prime_eval_smoke/llama32_3b_rl_vs_base_summary_env0314.json`
- Adapter result files:
  - p50 mix validation:
    `runs/prime_eval_smoke/llama32_3b_rl_ykq_p5050_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz/7d0e3462/results.jsonl`
  - strict v02:
    `runs/prime_eval_smoke/llama32_3b_rl_ykq_v02_strict_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz/72e811d1/results.jsonl`
  - strict v03:
    `runs/prime_eval_smoke/llama32_3b_rl_ykq_v03_strict_val20_env0314/evals/humanize-rl-env--meta-llama--Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz/76186222/results.jsonl`
- Base versus adapter aggregate:
  - p50 mix validation, `20` examples:
    base `0.421325`, adapter `0.470892`, delta `+0.049567`;
    output tokens `55.6 -> 51.75`; truncation stayed `0`;
    risk improved `-0.75 -> -0.635`.
  - strict `v02_smoke`, `9` examples:
    base `-0.271042`, adapter `-0.128778`, delta `+0.142264`;
    output tokens `50.89 -> 55.89`; truncation stayed `0`;
    risk improved `-0.716667 -> -0.588889`.
  - strict `v03`, `20` examples:
    base `-0.225785`, adapter `-0.101596`, delta `+0.124189`;
    output tokens `158.4 -> 287.85`; truncation stayed `0`;
    risk improved `-0.6525 -> -0.5`;
    wrapper penalty improved from `-0.01` to `0.0`.
- Humanizer-check qualitative read:
  - positive:
    the adapter mostly removed explicit wrappers and option menus in these
    evals; no truncation; aggregate rewards improved across all three gates.
  - negative:
    the adapter still uses generic friendly filler such as `hope you're doing
    well`, `touch base`, `So, I was thinking`, and inflated prose.
  - negative:
    p50's largest win, example `16`, reached `0.885356` while using
    awkward business phrasing: `operational game stronger`. This means p50 can
    still reward a surface-human style that is not actually good prose.
  - negative:
    strict `v03` largest loss, example `99`, dropped from `0.860753` to
    `-0.184107` (`-1.044860`). The adapter expanded the parity-check program
    explanation and lost the reward. This is the main regression to inspect
    before scaling.
  - negative:
    strict `v03` output length increased by `+129.45` tokens on average. That
    is not truncation or collapse, but it is a drift toward longer tutorial
    answers.
- Decision:
  this is a real technical success and an aggregate eval improvement, but not a
  clean model-quality pass. Do not launch the larger Llama or Qwen target run
  until strict-v03 length drift and the example-99 regression are understood.
  The next useful work is family/mode-level eval analysis and either a smaller
  targeted second RL run or a simple SFT dataset focused on direct answers,
  short outputs, and humanizer-lint failures.

### Family and Mode Eval Join

- Saved artifact:
  `runs/prime_eval_smoke/llama32_3b_eval_family_mode_env0314.json`
- Join method:
  eval `example_id` matched the numeric suffix of the task id, e.g.
  `rl_v03_000099 -> 99`. All eval rows matched task metadata.
- p50 mix validation:
  - matched rows: `20`
  - compression: `+0.013468`, output tokens `32.14 -> 31.0`
  - rewrite_repair: `+0.162358`, output tokens `66.6 -> 65.0`
  - tone_shift: `+0.010658`, output tokens `69.25 -> 61.63`
  - read:
    aggregate p50 gain is mostly rewrite_repair, not broad improvement.
- strict `v02_smoke`:
  - matched rows: `9`
  - compression: `+0.212598`, output tokens `10.33 -> 21.67`
  - rewrite_repair: `+0.306387`, output tokens `61.67 -> 57.0`
  - tone_shift: `-0.092192`, output tokens `80.67 -> 89.0`
  - read:
    this violates the family gate because `tone_shift` drops worse than
    `-0.05`.
- strict `v03`:
  - matched rows: `20`
  - compression: `+0.363709`, output tokens `78.5 -> 70.75`
  - direct_email: `+0.019987`, output tokens `258.8 -> 362.4`
  - rewrite_repair: `+0.084456`, output tokens `141.82 -> 332.91`
  - by mode:
    - compression: `+0.363709`, output tokens `78.5 -> 70.75`
    - expansion: `+0.034430`, output tokens `168.67 -> 413.83`
    - long_form_generate: `+0.012432`, output tokens `279.25 -> 412.25`
    - multi_constraint_compose: `+0.050206`, output tokens `177.0 -> 163.0`
    - rewrite_humanize: `+0.144486`, output tokens `109.6 -> 235.8`
  - read:
    strict v03 has positive reward deltas, but several modes get much longer.
    This is the main length-drift concern.
- Gate verdict:
  failed scale-up gate due to `tone_shift` family drop on strict `v02_smoke`
  and strict-v03 length drift. Passed technical run, deployment, no-truncation,
  p50 aggregate, strict aggregate, and wrapper/option non-regression.
- Next experimental direction:
  do not train a bigger run yet. Build a narrow failure analysis set covering
  `tone_shift`, long-form `expansion`, long-form `rewrite_humanize`, and the
  strict-v03 example `99` regression, then choose between:
  - targeted SFT data that teaches direct shorter answers and removes
    humanizer-lint phrases;
  - a second small RL run with a stricter length/humanizer diagnostic gate.

### Failure Set and SFT Readiness

- Saved failure set:
  `runs/prime_eval_smoke/llama32_3b_failure_set_env0314.jsonl`
- Failure-set row count:
  `28`
- Failure-set reason counts:
  - `humanizer_phrase_hit=15`
  - `length_drift_ge_100_tokens=6`
  - `reward_loss_le_-0.05=5`
  - `high_reward_win_needs_manual_review=4`
  - `v02_tone_shift_family_gate=3`
  - `high_reward_humanizer_phrase=2`
- Failure-set comparison counts:
  - p50 mix validation: `11`
  - strict `v02_smoke`: `7`
  - strict `v03`: `10`
- Most important failures:
  - `rl_v03_000099`:
    strict v03 reward delta `-1.044860`, output token delta `+461`.
    This is the largest regression and should be inspected first.
  - `rl_v01_000369`:
    v02 `tone_shift`, reward delta `-0.278650`. This drives the family-gate
    failure.
  - `rl_v01_000016`:
    p50 reward `0.885356` despite humanizer phrase hits:
    `hope you're doing well`, `so,`, and `operational game`.
- SFT readiness check:
  current RL tasksets have no `reference_response` values.
  - `data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`:
    `972` rows, `0` references
  - `data/rl/humanize_tasks_v03_filtered.jsonl`:
    `492` rows, `0` references
  - `environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl`:
    `99` rows, `0` references
- Local SFT data check:
  `data/processed` is absent in this worktree, so there is no ready local SFT
  dataset to launch for Qwen/Llama.
- SFT implication:
  do not claim SFT is ready. The next SFT step is reference generation or
  restoration from Hugging Face/artifacts, then conversion through the existing
  SFT dataset builder.

### Current Pickup State

- Completed:
  - env `jayshah5696/humanize-rl-env@0.3.14` validated in Prime training;
  - Qwen path diagnosed as renderer/API failure, not just bad model choice;
  - Llama 3.2 3B 50-step RL smoke completed;
  - W&B run recorded:
    `https://wandb.ai/jayshah5696/humanize-rl/runs/lvqx5v5g`;
  - deployed same-run Llama adapter:
    `meta-llama/Llama-3.2-3B-Instruct:ykqmg5inv09jj3gm2htr7aoz`;
  - base and adapter evals saved for p50 mix, strict v02, and strict v03;
  - family/mode and failure-set analyses saved.
- Final Prime model status:
  - explicit step `50` checkpoint/model record:
    `r5snl7aw0objspezj71ybs3t`, still `UPLOADING`
  - train checkpoint record:
    `jtvwt4qjt9nf3aydt1b0x0u7`, still `UPLOADING`
  - deployed same-run adapter used for eval:
    `ykqmg5inv09jj3gm2htr7aoz`, `DEPLOYED`
  - saved final deployment snapshot:
    `runs/prime_training_smoke/q6lsjfaxocod6xt4e2qebjgv/deployments_final.json`
- Main result:
  the Llama adapter improves aggregate rewards:
  p50 `+0.049567`, strict v02 `+0.142264`, strict v03 `+0.124189`.
- Main blocker:
  the run fails the scale-up gate because strict v02 `tone_shift` drops
  `-0.092192`, and strict v03 gets much longer on expansion/rewrite tasks.
- Do next:
  1. Inspect `runs/prime_eval_smoke/llama32_3b_failure_set_env0314.jsonl`.
  2. Restore or generate SFT target responses. Current tasksets do not contain
     targets.
  3. Add stronger humanizer/length diagnostics before any bigger RL run.
  4. Rerun the same three evals after any SFT or reward change.
- Do not do next:
  - do not launch Qwen 2B or Llama larger training from this checkpoint yet;
  - do not call the adapter a final model;
  - do not treat aggregate reward as sufficient without family and qualitative
    gates.

## 2026-06-22: Targeted SFT Reference Data From Llama RL Failures

### Why This Step Exists

The Prime Llama 3.2 3B RL smoke was technically successful but failed the
scale-up gate:

- strict `v02_smoke` `tone_shift` family delta: `-0.092192`
- strict `v03` output length drift on expansion/rewrite/long-form modes
- several high-scoring outputs still had humanizer problems such as wrapper
  phrasing, polished filler, and bad reward tolerance around phrases

The local RL tasksets contain no `reference_response` values, and
`data/processed` was absent before this step. That means SFT cannot honestly be
called ready. The next scientific move is to create a small targeted
failure-correction SFT artifact, validate it, then merge it with the restored
full SFT corpus before any SFT/RL scale-up.

### Generator Fix

Script touched:
`scripts/data/build/generate_sft_references_from_failures.py`

Test touched:
`tests/scripts/test_generate_sft_references_from_failures.py`

The first live generation attempt used the original all-at-end writer:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/generate_sft_references_from_failures.py \
  --limit 28 --sleep-seconds 0.2
```

It ran for about four minutes with no partial output, so it was stopped. That
was not repeatable enough for a paid API generation job.

Fix:

- writes each generated row immediately with `append_jsonl_row()`;
- adds `--overwrite` for clean runs;
- adds `--resume` for partial retries;
- keys resume by `(task_id, failure_comparison)`, because the failure set can
  contain the same task under multiple eval comparisons;
- prints only progress counts and task ids, not generations or secrets;
- keeps the Google-only model guard: model must start with `google/`.
- tests cover helper behavior plus Click CLI behavior for `--resume`,
  `--overwrite`, and the Google-only model guard.

Verification:

```bash
PYTHONPATH=src uv run --no-project --with pytest --with pytest-cov --with click \
  pytest tests/scripts/test_generate_sft_references_from_failures.py -q
```

Result: `9 passed`. Coverage prints `No data was collected` because these
tests import a script outside `src`; this is not a behavioral failure.

```bash
PYTHONPATH=src uv run --no-project --with ruff ruff check \
  scripts/data/build/generate_sft_references_from_failures.py \
  tests/scripts/test_generate_sft_references_from_failures.py
```

Result: `All checks passed!`

Dry-run command:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/generate_sft_references_from_failures.py \
  --dry-run --limit 3 --overwrite \
  --output-path runs/prime_eval_smoke/reference_generation_dry_run.jsonl
```

Result:

- wrote `3` dry-run rows
- missing tasks: `0`
- output:
  `runs/prime_eval_smoke/reference_generation_dry_run.jsonl`

### Live Reference Generation

Command:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/generate_sft_references_from_failures.py \
  --limit 28 --sleep-seconds 0.2 --overwrite
```

Generator model:
`google/gemini-3.1-pro-preview` via OpenRouter.

Input:
`runs/prime_eval_smoke/llama32_3b_failure_set_env0314.jsonl`

Output:
`data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314.jsonl`

Result:

- wrote rows: `28`
- missing tasks: `0`
- duplicate task ids in raw references:
  - `rl_v01_000016`: `2`
  - `rl_v01_000133`: `2`
- duplicate `(task_id, failure_comparison)` keys: `0`

Raw audit artifact:
`data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_audit.json`

Raw audit:

- rows: `28`
- unique task ids: `26`
- empty responses: `0`
- emoji count: `0`
- all-caps count: `0`
- humanizer phrase/root hits: `1`
  - `rl_v01_000016` used `enhance/enhancements`
- wrapper-like openings/usages: `4`
  - `rl_v01_000053`: starts with rewrite-wrapper language
  - `rl_v01_000135`: starts with `Here are my notes`
  - `rl_v01_000462`: starts with `Here is a quick Slack update`
  - `rl_v03_000186`: includes `Here is a look`
- family counts:
  - compression: `6`
  - direct_email: `2`
  - rewrite_repair: `13`
  - tone_shift: `7`
- mode counts:
  - compression: `2`
  - expansion: `3`
  - long_form_generate: `2`
  - rewrite: `18`
  - rewrite_humanize: `3`
- response word length min/mean/max: `17 / 90.96 / 312`

### Existing SFT Builder Validation

First builder command with normal `uv run` failed before validating the data
because `fasttext-wheel==0.9.2` could not link against the local macOS SDK:

- missing SDK:
  `/Applications/Xcode_15.2.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX14.2.sdk`
- linker error:
  `ld: library 'c++' not found`

This is a local native dependency/build issue, not a data schema issue.

Second builder command used `--no-project` and a deliberately missing
Track-A scorer path so the existing builder could validate schema/splits
without importing `fasttext`:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/build_gemma4_sft_dataset.py \
  --input-path data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314.jsonl \
  --output-dir data/processed/sft/llama32_3b_failure_refs_env0314 \
  --mlx-dir data/processed/sft/llama32_3b_failure_refs_env0314_mlx_smoke \
  --track-a-scorer-path /tmp/humanize_missing_track_a_scorer.pkl \
  --smoke-train-size 10 \
  --smoke-valid-size 5 \
  --pilot-train-size 10 \
  --pilot-valid-size 5 \
  --max-ai-probability 1.0
```

Result:

- raw rows: `28`
- accepted rows: `26`
- rejected rows: `2`
- split counts: `{'train': 23, 'valid': 1, 'test': 2}`

Rejected rows:

- `rl_v03_000075`: `instruction_too_long`
- `rl_v03_000318`: `instruction_too_long`

The raw file is therefore not a clean training candidate. It is a reference
generation artifact plus audit input.

### Clean Failure-Correction SFT Candidate

Clean output:
`data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_clean.jsonl`

Clean report:
`data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_clean_report.json`

Clean audit:
`data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_clean_audit.json`

Filtering rule:

- drop rows with instruction length above the existing builder limit;
- drop empty responses;
- drop emoji;
- drop all-caps words outside an allowlist such as `API`, `JSON`, `PTO`;
- drop `Here is` / `Here are` wrapper-like rows;
- drop listed humanizer phrase/root hits such as `enhance`, `leverage`,
  `crucial`, `vital`, `certainly`, `in conclusion`;
- drop duplicate task ids after keeping the first clean row.

Clean candidate result:

- raw rows: `28`
- kept rows: `20`
- dropped rows: `8`
- unique task ids kept: `20`
- family counts:
  - compression: `4`
  - direct_email: `1`
  - rewrite_repair: `10`
  - tone_shift: `5`
- mode counts:
  - compression: `1`
  - expansion: `3`
  - long_form_generate: `1`
  - rewrite: `13`
  - rewrite_humanize: `2`
- response word length min/mean/max: `22 / 101.45 / 312`
- clean audit:
  - empty: `0`
  - emoji: `0`
  - wrapper `Here is/Here are`: `0`
  - phrase/root hits: `0`

Builder command on clean candidate:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/build_gemma4_sft_dataset.py \
  --input-path data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_clean.jsonl \
  --output-dir data/processed/sft/llama32_3b_failure_refs_env0314_clean \
  --mlx-dir data/processed/sft/llama32_3b_failure_refs_env0314_clean_mlx_smoke \
  --track-a-scorer-path /tmp/humanize_missing_track_a_scorer.pkl \
  --smoke-train-size 10 \
  --smoke-valid-size 5 \
  --pilot-train-size 10 \
  --pilot-valid-size 5 \
  --max-ai-probability 1.0
```

Result:

- raw rows: `20`
- accepted rows: `20`
- rejected rows: `0`
- split counts: `{'train': 18, 'valid': 1, 'test': 1}`
- manifest:
  `data/processed/sft/llama32_3b_failure_refs_env0314_clean/manifest.json`
- quality report:
  `data/processed/sft/llama32_3b_failure_refs_env0314_clean/quality_report.md`

Important caveat:
the clean candidate is a targeted correction set, not a full SFT corpus. It is
too small for a serious SFT run by itself. Use it as a repair slice to merge
or oversample with the restored full SFT data.

### Gate Decision After This Step

Completed:

- generated targeted reference responses for the 28 Prime Llama RL failure rows;
- converted those into a clean 20-row SFT correction candidate;
- validated the clean candidate through the existing SFT builder;
- fixed the generator so future paid generation jobs can resume safely;
- kept generation model policy compliant: Google OpenRouter model only.
- verifier rerun approved the artifact:
  - focused tests: `9 passed`
  - ruff: passed
  - clean rows: `20`
  - unique task ids: `20`
  - bad-count scan: `0`
  - manifest: `raw_rows=20`, `accepted_rows=20`, `rejected_rows=0`

Still blocked for full SFT/RL:

- the full historical SFT corpus is still not restored locally;
- the clean correction set has only 20 accepted rows;
- the Track-A scorer path triggers a local `fasttext` native build/import issue
  in this environment unless scorer loading is bypassed;
- no SFT model has been launched from this new data yet;
- no new RL run should start until the SFT data mix is approved and evaluated.

Next concrete pickup:

1. Restore the full SFT corpus from Hugging Face or prior artifacts.
2. Merge/oversample
   `data/processed/sft/reference_targets/llama32_3b_failure_refs_env0314_clean.jsonl`
   into that corpus as a targeted failure-correction slice.
3. Build one frozen SFT dataset manifest and quality report.
4. Run a tiny SFT smoke on the chosen target model.
5. Evaluate base vs SFT on the same frozen Prime eval prompts:
   p50 mix validation, strict `v02_smoke`, strict `v03`.
6. Only if SFT beats base and does not regress wrappers/length/family gates,
   run the next Prime RL smoke.

Stop condition for a full run:

- do not launch a bigger RL run until SFT data is restored, merged, audited,
  and a smoke SFT improves the frozen evals;
- the next “full” training run should be a single serious SFT or RL run only
  after those gates pass, not another chain of tiny reward tweaks.

## 2026-06-22: Restored Full SFT Corpus and Built Merged Candidate

### Hugging Face Restore

Repo checked:
`jayshah5696/humanize-rl-sft-dataset`

HF metadata:

- created: `2026-05-23T06:14:49+00:00`
- latest repo SHA: `5494ceb671b83723bea424b846c5e81a4ecb4b3c`
- description says `4,835` high-quality SFT pairs
- relevant file:
  `data/v2/v04_sft_final.jsonl`

Commands:

```bash
hf datasets info jayshah5696/humanize-rl-sft-dataset --format json

hf download jayshah5696/humanize-rl-sft-dataset \
  --type dataset \
  --include 'data/v2/v04_sft_final.jsonl' \
  --include 'README.md' \
  --local-dir data/hf/humanize-rl-sft-dataset

mkdir -p data/processed
cp data/hf/humanize-rl-sft-dataset/data/v2/v04_sft_final.jsonl \
  data/processed/v04_sft_final.jsonl
```

Restored file:
`data/processed/v04_sft_final.jsonl`

Restore audit:

- rows: `4,835`
- SHA256:
  `9fa71ca5f06ae18188002c91e34588c16984a2ba8935f80300273add746968cf`
- empty instruction rows: `0`
- empty response rows: `0`
- instruction words min/mean/max: `7 / 28.05 / 178`
- response words min/mean/max: `5 / 46.42 / 186`
- top sources:
  - `safe_expand_3000_raw`: `1,802`
  - `stream_b`: `1,617`
  - `safe_expand_raw`: `667`
  - `chat_expanded`: `423`
  - missing source: `326`

### Restored Baseline Builder Validation

Command:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/build_gemma4_sft_dataset.py \
  --input-path data/processed/v04_sft_final.jsonl \
  --output-dir data/processed/sft/gemma4_e2b_v04_restored \
  --mlx-dir data/processed/sft/gemma4_e2b_v04_restored_mlx_smoke \
  --track-a-scorer-path /tmp/humanize_missing_track_a_scorer.pkl \
  --smoke-train-size 100 \
  --smoke-valid-size 20 \
  --pilot-train-size 500 \
  --pilot-valid-size 50 \
  --max-ai-probability 1.0
```

Result:

- raw rows: `4,835`
- accepted rows: `4,773`
- rejected rows: `62`
- split counts: `{'train': 4295, 'valid': 238, 'test': 240}`

Top rejection reasons:

- `possible_phone_pii`: `20`
- `possible_fake_name:John`: `18`
- `possible_fake_name:Sarah`: `8`
- `possible_fake_name:Bob`: `5`
- `possible_fake_name:Charlie`: `3`
- `response_ai_tell:certainly`: `3`
- `response_ai_tell:of course`: `3`
- `exact_duplicate`: `2`

The scorer path is still bypassed because the local `fasttext` native import
is broken in this environment. This validation proves data schema and builder
compatibility, not Track-A scorer filtering.

### Merge Candidate With Failure-Correction Slice

Attempted oversampling first:
`data/processed/v04_sft_final_plus_llama_failure_refs_env0314_x10.jsonl`

Oversampling report:
`data/processed/v04_sft_final_plus_llama_failure_refs_env0314_x10_report.json`

Numbers:

- base rows: `4,835`
- repair unique rows: `20`
- oversample factor: `10`
- raw combined rows: `5,035`
- repair share: `3.97%`

Builder result for x10:

- raw rows: `5,035`
- accepted rows: `4,793`
- rejected rows: `242`
- `exact_duplicate`: `182`

Interpretation:
the existing SFT builder dedupes exact instruction/response pairs, so the x10
oversampling does not survive the canonical build. Do not use the x10 artifact
as a claim of oversampled training unless the training sampler is changed after
the builder stage.

Canonical merged file:
`data/processed/v04_sft_final_plus_llama_failure_refs_env0314.jsonl`

Canonical merge report:
`data/processed/v04_sft_final_plus_llama_failure_refs_env0314_report.json`

Numbers:

- base rows: `4,835`
- repair rows added: `20`
- raw combined rows: `4,855`
- repair share before builder: `0.41%`
- SHA256:
  `e1a250e109d58bfddba54a326d603897377e2a3f367fccba4985e079015c7633`
- repair family counts:
  - compression: `4`
  - direct_email: `1`
  - rewrite_repair: `10`
  - tone_shift: `5`
- repair mode counts:
  - compression: `1`
  - expansion: `3`
  - long_form_generate: `1`
  - rewrite: `13`
  - rewrite_humanize: `2`

Builder command:

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/build_gemma4_sft_dataset.py \
  --input-path data/processed/v04_sft_final_plus_llama_failure_refs_env0314.jsonl \
  --output-dir data/processed/sft/gemma4_e2b_v04_plus_llama_failure_refs_env0314 \
  --mlx-dir data/processed/sft/gemma4_e2b_v04_plus_llama_failure_refs_env0314_mlx_smoke \
  --track-a-scorer-path /tmp/humanize_missing_track_a_scorer.pkl \
  --smoke-train-size 100 \
  --smoke-valid-size 20 \
  --pilot-train-size 500 \
  --pilot-valid-size 50 \
  --max-ai-probability 1.0
```

Result:

- raw rows: `4,855`
- accepted rows: `4,793`
- rejected rows: `62`
- split counts: `{'train': 4313, 'valid': 239, 'test': 241}`
- manifest:
  `data/processed/sft/gemma4_e2b_v04_plus_llama_failure_refs_env0314/manifest.json`
- quality report:
  `data/processed/sft/gemma4_e2b_v04_plus_llama_failure_refs_env0314/quality_report.md`

Repair-row placement after builder:

- train: `18`
- valid: `1`
- test: `1`
- smoke_train: `1`
- smoke_valid: `0`
- pilot_train: `4`
- pilot_valid: `0`

Caveat:
the builder output keeps `source=prime_failure_reference_generation`, but it
drops nested repair metadata from the normalized split rows. Use the canonical
pre-builder JSONL and merge report for repair family/mode analysis.

### Gate Decision After Restore/Merge

Data gate:
passed for a canonical merged SFT dataset.

Training gate:
not passed yet. No SFT run has been launched from this merged dataset.

Next training-safe options:

1. Publish the canonical merged JSONL to HF under a new path/revision, then run
   a generic Qwen/Llama Modal SFT smoke from that HF file.
2. Or run the existing Gemma MLX/Modal SFT path as a fallback smoke only, with
   clear labeling that it is not the primary Qwen/Llama target.

Do not claim the model-training phase is complete until there is at least:

- a completed SFT smoke run id;
- W&B run link;
- base-vs-SFT eval on the frozen Prime eval prompts;
- documented pass/fail on p50, strict, wrapper, length, and family gates.

## 2026-06-22: Prime Training Capability Check and Platform Policy Correction

### Why This Was Checked

We briefly added/launched a generic Modal TRL SFT smoke for Qwen before checking
Prime's current SFT path. That ordering was wrong for this project. Prime is
the preferred training platform when it supports the target model and workflow;
Modal/TRL should be fallback/custom infrastructure.

User instruction after correction:

- do not stop the existing Modal run;
- finish it end to end;
- next time, check Prime Intellect first.

### Prime Docs Verified

Source:
`https://docs.primeintellect.ai/prime-rl/training`

Live Prime CLI checked:

```bash
prime --version
prime train --help
prime --help
```

Result:

- Prime CLI version: `0.6.14`
- `prime train` manages Hosted Training runs.
- `prime train models` lists Hosted Training models.
- `prime train init` generates Hosted Training TOML templates.
- `prime train logs`, `metrics`, `rollouts`, `progress`, `checkpoints`, and
  `usage` are available for run monitoring.

Prime docs say `prime-rl` supports these training entrypoints:

- `uv run rl @ config.toml`
  - RL trainer/orchestrator/inference wrapper.
- `uv run sft @ config.toml`
  - dataset-based supervised fine-tuning on a Hugging Face dataset.
  - launches torchrun internally.
  - do not call torchrun directly.
- `uv run inference`
  - vLLM server with Prime-specific endpoints such as `/update_weights`,
    `/load_lora_adapter`, and `/init_broadcaster`.
- `uv run trainer` / `uv run orchestrator`
  - standalone components for advanced separated launches.

Prime docs also say the RL entrypoint supports three modes through
`orchestrator.training_mode`:

- `rl`
  - standard RL.
- `opd`
  - on-policy distillation, student plus teacher, teacher must be vLLM because
    prompt logprobs are needed.
- `sft`
  - teacher-generated hard distillation through the orchestrator path.

Important distinction:

- Use `uv run sft` for traditional dataset SFT from HF data.
- Use `orchestrator.training_mode = "sft"` only when a teacher generates the
  supervision on the fly.

Prime SFT dataset formats:

- HF dataset with `prompt` + `completion` columns.
- HF dataset with a `messages` column.
- If both are present, `messages` takes precedence.
- Tool-use SFT can use `tools` or `tool_defs`.
- `chat_template_kwargs` is forwarded into `apply_chat_template`.

Qwen-specific note from Prime docs:

- Qwen3 upstream chat templates can corrupt multi-turn loss masks because they
  strip past `<think>` blocks.
- Prime recommends enabling a typed renderer, e.g. `[renderer] name = "qwen3"`.
- Prime renderers cover Qwen3/Qwen3.5 and several other families.
- For our Qwen3.5 target, future Prime SFT configs should use the corresponding
  Prime renderer rather than relying on generic tokenizer templates.

Prime observability and checkpoints:

- W&B is enabled with `--wandb` and can set project/name flags.
- Metrics include SFT `loss/mean`, `val/loss`, progress samples/tokens, LR,
  grad norm, throughput, MFU, peak memory, and step timing.
- Checkpoints can write HF-compatible weight snapshots under
  `<output_dir>/weights/step_N/`.
- LoRA runs can set `ckpt.weights.save_adapter_separately = true` to save the
  raw adapter separately.
- Resume uses `--ckpt.resume-step`.

### New Platform Rule

Documented in `AGENTS.md`:

1. Check Prime first for new Qwen/Llama/Nemotron/GPT-OSS SFT/RL work.
2. Use Prime `uv run sft` for dataset SFT when possible.
3. Use Prime RL / Hosted Training for env-based RL when possible.
4. Use Modal/TRL only when Prime cannot support the model/path, custom code is
   required, or an already-started Modal run must be finished.
5. Before adding new training infrastructure, record the Prime CLI/docs check
   in this log.

### Existing Modal SFT Smoke Status

This Modal run was already launched before the Prime-first correction and was
not stopped.

Script:
`src/humanize_rl/training/finetune_generic_sft_modal.py`

Tests:
`tests/training/test_finetune_generic_sft_modal.py`

Validation:

```bash
PYTHONPATH=src uv run --no-project --with pytest --with pytest-cov \
  pytest tests/training/test_finetune_generic_sft_modal.py -q

PYTHONPATH=src uv run --no-project --with ruff ruff check \
  src/humanize_rl/training/finetune_generic_sft_modal.py \
  tests/training/test_finetune_generic_sft_modal.py
```

Result:

- focused tests: `9 passed`
- ruff: passed

Modal r1:

- app: `ap-hFY4ILdujA2r6Pdw78fXYM`
- call: `fc-01KVR9MRSWY8HX7T2SQ42FJSE7`
- W&B:
  `https://wandb.ai/jayshah5696/humanize-rl/runs/qwen35-08b-humanize-sft-smoke-env0314-r1`
- failed before training:
  `TypeError: SFTConfig.__init__() got an unexpected keyword argument 'max_seq_length'`
- cause:
  TRL `1.6.0` uses `max_length`, not `max_seq_length`, and uses
  `processing_class` instead of older `tokenizer` in `SFTTrainer`.
- fix:
  script now introspects `SFTConfig` and `SFTTrainer` signatures and maps fields
  for current/older TRL versions.

Modal r2:

- app: `ap-IsBbT5cMGwDwyjJmnNgIYP`
- call: `fc-01KVR9VA2S9VJ8QCAR2YN3VRA5`
- command:

```bash
uvx modal run --detach \
  src/humanize_rl/training/finetune_generic_sft_modal.py \
  --model-name Qwen/Qwen3.5-0.8B \
  --data-files data/v3/v04_sft_final_plus_llama_failure_refs_env0314.jsonl \
  --train-limit 64 \
  --max-steps 5 \
  --max-seq-length 2048 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --learning-rate 2e-4 \
  --experiment-name qwen35-08b-humanize-sft-smoke-env0314-r2
```

W&B:
`https://wandb.ai/jayshah5696/humanize-rl/runs/qwen35-08b-humanize-sft-smoke-env0314-r2`

Observed r2 training metrics from Modal logs:

- train examples after split: `58`
- eval examples: `6`
- steps completed: `5/5`
- step losses:
  - step 1: `2.488`
  - step 2: `2.437`
  - step 3: `2.123`
  - step 4: `1.819`
  - step 5: `1.965`
- final eval loss: `1.88263`
- final eval mean token accuracy: `0.58969`
- train runtime: `44.3s`
- train samples/s: `0.903`
- train steps/s: `0.113`
- train loss summary: `2.166`

Current caveat:

- r2 completed trainer steps and W&B sync.
- Modal app state: `stopped`.
- Modal volume check confirmed persisted artifacts:
  - `/qwen35-08b-humanize-sft-smoke-env0314-r2/final_adapter`
  - `/qwen35-08b-humanize-sft-smoke-env0314-r2/checkpoint-5`
  - `/qwen35-08b-humanize-sft-smoke-env0314-r2/README.md`
- W&B reported `0` artifacts synced, so the adapter is in the Modal volume but
  was not uploaded/logged as a W&B artifact or pushed to HF.

Next action:

1. Do not launch another Modal SFT run unless explicitly needed.
2. Build the next SFT/RL attempt as a Prime-first config:
   - dataset SFT via Prime `uv run sft`;
   - Qwen3.5 renderer enabled;
   - W&B enabled;
   - checkpoint/adapters enabled;
   - then frozen Prime evals against base vs SFT.

## 2026-06-22: Prime-First SFT Dataset and Config Gate

### Modal Run Status Check

Checked live Modal state:

```bash
uvx modal app list
```

Result:

- no active Modal apps;
- the already-started Modal Qwen SFT smoke is complete/stopped;
- no dangling Modal run is still consuming GPU.

Checked Modal volume artifacts:

```bash
uvx modal volume ls humanize-rl-checkpoints \
  /qwen35-08b-humanize-sft-smoke-env0314-r2/final_adapter

uvx modal volume ls humanize-rl-checkpoints \
  /qwen35-08b-humanize-sft-smoke-env0314-r2/checkpoint-5
```

Persisted files:

- `final_adapter/adapter_model.safetensors`
- `final_adapter/adapter_config.json`
- tokenizer files and `chat_template.jinja`
- `checkpoint-5/optimizer.pt`
- `checkpoint-5/scheduler.pt`
- `checkpoint-5/trainer_state.json`
- `checkpoint-5/adapter_model.safetensors`

Interpretation:

- Modal r2 is a successful plumbing smoke, not a quality checkpoint.
- It trained only `5` steps over a tiny `64` row limit.
- It should not be treated as the project SFT model.
- Next training should go back to Prime-first ordering.

### Prime Docs and Source Check

Official docs checked:

- `https://docs.primeintellect.ai/prime-rl/training`
- `https://docs.primeintellect.ai/prime-rl/configuration`
- official GitHub example:
  `https://github.com/PrimeIntellect-ai/prime-rl/tree/main/examples/reverse_text`

Relevant confirmed facts:

- dataset SFT is `uv run sft @ config.toml`;
- env RL is `uv run rl @ config.toml`;
- `prime train` is Hosted Training, not the same as open `prime-rl` dataset SFT;
- SFT accepts `prompt` + `completion` or a `messages` column;
- `messages` takes precedence if both are present;
- Qwen3/Qwen3.5 should use a typed renderer because default chat-template
  masking can break on position-dependent templates;
- W&B is enabled by `[wandb]` or CLI `--wandb`;
- checkpoints write HF-compatible weights under `<output_dir>/weights/step_N`;
- LoRA adapter separation requires `[model.lora]` and
  `[ckpt.weights] save_adapter_separately = true`.

Prime source checkout used for schema/example verification:

```bash
git clone --depth 1 https://github.com/PrimeIntellect-ai/prime-rl.git \
  /tmp/prime-rl-src
```

Important local runtime finding:

```bash
uvx --from 'git+https://github.com/PrimeIntellect-ai/prime-rl.git' \
  sft --help
```

failed on macOS because full `prime-rl` depends on CUDA/Linux torch wheels
(`torch>=2.9.0` with CUDA wheel tags). That is a local platform limitation, not
a config failure. Config validation can still be done with
`prime-rl-configs`.

Hosted Training CLI check:

```bash
prime --plain train init /tmp/prime_hosted_template.toml -f
prime --plain train configs --output json
prime --plain train models --output json
```

Finding:

- Hosted Training template supports `loss = "rl"` and `loss = "sft"`;
- the hosted `sft` path is teacher distillation over env rollouts;
- it is not the same as dataset SFT from
  `jayshah5696/humanize-rl-prime-sft-messages-env0314`;
- hosted `loss = "sft"` would need an approved teacher/generator model before
  use, because project policy keeps data generation/judging/scoring Google-only
  unless an exception is explicit;
- installed Prime CLI `0.6.14` has no command that submits the open
  `prime-rl` dataset SFT config directly from this Mac;
- current hosted model list still includes the planned Qwen targets:
  `Qwen/Qwen3.5-0.8B`, `Qwen/Qwen3.5-2B`, `Qwen/Qwen3.5-4B`,
  `Qwen/Qwen3.5-9B`, and `Qwen/Qwen3.6-35B-A3B`, all not at capacity;
- therefore current SFT warmup remains Prime `prime-rl` dataset SFT via
  `uv run sft @ configs/prime_rl/...` on Linux/CUDA.

Renderer check:

```bash
uv run --no-project --with 'renderers>=0.1.8.dev28' python - <<'PY'
from renderers.base import MODEL_RENDERER_MAP
print(MODEL_RENDERER_MAP["Qwen/Qwen3.5-0.8B"])
print(MODEL_RENDERER_MAP["Qwen/Qwen3.5-2B"])
PY
```

Result:

- `Qwen/Qwen3.5-0.8B` maps to `qwen3.5`;
- `Qwen/Qwen3.5-2B` maps to `qwen3.5`.

### Prime SFT Dataset Publication

First, a raw Prime messages conversion was validated:

- local file:
  `data/processed/v04_sft_final_plus_llama_failure_refs_env0314_prime_messages.jsonl`
- rows: `4855`
- malformed rows: `0`
- empty rows: `0`
- repair-reference rows: `20`
- SHA256:
  `114b8cdcd0a6142cd93f83af6a8721da51bbfc7aca4a4323f19d61dc98d6ed09`

Uploaded raw conversion and report to the broad SFT dataset repo:

- dataset:
  `jayshah5696/humanize-rl-sft-dataset`
- raw JSONL commit:
  `560775f883574f190671bb284c35d6980ed475b1`
- report commit:
  `e348871f5a060ccfd8fa0a746ece21a6601e0993`

Then this was corrected. The broad dataset repo still contains old parquet and
v2 artifacts, so using it as `data.name` risks loading the wrong default split.
For Prime SFT, use a dedicated dataset repo instead.

Dedicated dataset:

```text
jayshah5696/humanize-rl-prime-sft-messages-env0314
```

Initial one-split upload:

- commit:
  `6dcddd03de32124e2edd09fd806f362d568365cc`
- replaced because it had only `train` and no held-out validation/test split.

Final split upload:

- commit:
  `e0895734e527ea3549d6a600fd31e3e598efd7ec`
- files:
  - `data/train.jsonl`
  - `data/validation.jsonl`
  - `data/test.jsonl`
  - `report.json`
  - `manifest.json`
  - `quality_report.md`
  - `README.md`

Published split report:

- train rows: `4313`
- validation rows: `239`
- test rows: `241`
- total accepted rows: `4793`
- upstream rejected rows: `62`
- duplicate ids: `0`
- malformed rows: `0`
- empty rows: `0`
- repair-reference rows: `20`
- source counts:
  - `safe_expand_3000_raw`: `1801`
  - `stream_b`: `1556`
  - `safe_expand_raw`: `667`
  - `chat_expanded`: `423`
  - `unknown`: `326`
  - `prime_failure_reference_generation`: `20`
- mode counts:
  - `rewrite_humanize`: `2990`
  - `direct_generation`: `1803`
- task type counts:
  - `slack_chat`: `1887`
  - `email`: `1377`
  - `direct_generation`: `832`
  - `rewrite_or_edit`: `697`

HF loader validation:

```bash
uv run --no-project --with datasets python - <<'PY'
from datasets import load_dataset
d = load_dataset("jayshah5696/humanize-rl-prime-sft-messages-env0314")
for split in ["train", "validation", "test"]:
    ds = d[split]
    print(split, len(ds), ds.column_names, [m["role"] for m in ds[0]["messages"]])
PY
```

Result:

- `train 4313`, roles `['user', 'assistant']`
- `validation 239`, roles `['user', 'assistant']`
- `test 241`, roles `['user', 'assistant']`

### Prime SFT Configs Added

New files:

- `configs/prime_rl/README.md`
- `configs/prime_rl/qwen35_08b_sft_smoke_env0314.toml`
- `configs/prime_rl/qwen35_2b_sft_target_env0314.toml`

Smoke config:

- model: `Qwen/Qwen3.5-0.8B`
- max steps: `20`
- sequence length: `4096`
- renderer: `qwen3.5`
- dataset: `jayshah5696/humanize-rl-prime-sft-messages-env0314`
- train split: `train`
- validation split: `validation`
- train batch size: `64`
- validation batch size: `32`
- LoRA: rank `32`, alpha `64`
- W&B run name: `qwen35-08b-prime-sft-smoke-env0314`
- checkpoint: weights-only, save adapter separately

Target config:

- model: `Qwen/Qwen3.5-2B`
- max steps: `200`
- sequence length: `4096`
- renderer: `qwen3.5`
- dataset: `jayshah5696/humanize-rl-prime-sft-messages-env0314`
- train split: `train`
- validation split: `validation`
- train batch size: `128`
- validation batch size: `64`
- LoRA: rank `32`, alpha `64`
- W&B run name: `qwen35-2b-prime-sft-target-env0314`
- checkpoint interval: `50`
- checkpoint: weights-only, save adapter separately

Schema validation command:

```bash
uv run --no-project \
  --with 'git+https://github.com/PrimeIntellect-ai/prime-rl.git#subdirectory=packages/prime-rl-configs' \
  --with 'renderers>=0.1.8.dev28' \
  --with pydantic-config \
  python - <<'PY'
from pathlib import Path
from pydantic_config import cli
from prime_rl.configs.sft import SFTConfig
for cfg in [
    Path("configs/prime_rl/qwen35_08b_sft_smoke_env0314.toml"),
    Path("configs/prime_rl/qwen35_2b_sft_target_env0314.toml"),
]:
    c = cli(SFTConfig, args=["@", str(cfg), "--dry-run"])
    print(cfg, c.model.name, c.renderer.name, c.data.splits, c.val.data.splits)
PY
```

Result:

- `qwen35_08b_sft_smoke_env0314.toml`:
  - model: `Qwen/Qwen3.5-0.8B`
  - renderer: `qwen3.5`
  - train split: `['train']`
  - validation split: `['validation']`
  - LoRA adapter saving: `true`
- `qwen35_2b_sft_target_env0314.toml`:
  - model: `Qwen/Qwen3.5-2B`
  - renderer: `qwen3.5`
  - train split: `['train']`
  - validation split: `['validation']`
  - LoRA adapter saving: `true`

Secret/forbidden wrapper check:

```bash
rg -n "wandb_v1_|WANDB_API_KEY|HF_TOKEN|PRIME_API_KEY|forbidden-wrapper-name" \
  AGENTS.md log.md configs/prime_rl configs/prime src scripts tests -S
```

Result:

- no W&B token committed;
- no new forbidden wrapper usage in `configs/prime_rl`;
- existing source references only read env vars such as `HF_TOKEN` and
  `WANDB_API_KEY`.

Final local verification for this slice:

```bash
PYTHONPATH=src uv run --no-project --with pytest --with pytest-cov --with click \
  pytest tests/scripts/test_generate_sft_references_from_failures.py \
  tests/training/test_finetune_generic_sft_modal.py -q
```

Result: `18 passed`.

```bash
PYTHONPATH=src uv run --no-project --with ruff ruff check \
  scripts/data/build/generate_sft_references_from_failures.py \
  tests/scripts/test_generate_sft_references_from_failures.py \
  src/humanize_rl/training/finetune_generic_sft_modal.py \
  tests/training/test_finetune_generic_sft_modal.py
```

Result: `All checks passed!`

```bash
git diff --check
```

Result: passed.

### Current Gate and Next Action

The next runnable training gate is Prime SFT smoke, not another Modal run:

```bash
uv run sft @ configs/prime_rl/qwen35_08b_sft_smoke_env0314.toml
```

Run this on Prime/Linux/CUDA or inside the Prime runtime, with W&B credentials
provided through environment/secrets, not committed.

Stop criteria for the smoke:

- config launches without renderer/schema errors;
- validation loss logs at step `0`, `10`, and final;
- adapter checkpoint appears under the output directory;
- W&B has the configured run name;
- no empty/reasoning-only outputs when the adapter is sampled on frozen eval
  prompts.

Only after that smoke passes:

1. run the `Qwen/Qwen3.5-2B` SFT target config;
2. evaluate base vs SFT on frozen p50/strict prompts;
3. start RL from the SFT checkpoint if Prime supports that checkpoint path;
4. otherwise document the limitation and use the base or a pushed HF checkpoint
   explicitly.

## 2026-06-22: Hosted Prime Full-Run Correction

### Correction

The previous local `prime-rl` dataset-SFT path is useful as a config artifact,
but it is not the execution path the user wanted. Prime Hosted Training is the
active training surface for this project.

Official Hosted Training docs checked:

- `https://docs.primeintellect.ai/hosted-training/advanced-configs`
- `https://docs.primeintellect.ai/hosted-training/models-and-pricing`

Hosted config capabilities confirmed:

- `.toml` run configs are launched with `prime train <config>`;
- required fields are `model`, `max_steps`, `batch_size`,
  `rollouts_per_example`, `[sampling]`, and at least one `[[env]]`;
- W&B is configured with `[wandb]`;
- validation and eval are configured with `[val]` and `[eval]`;
- checkpoints and adapters are configured with `[checkpoints]` and
  `[adapters]`;
- secrets can be passed at launch with `--env-var`.

### Hosted SFT Attempt

Config added:
`configs/prime/qwen35_2b_hosted_sft_gemini_env0314_full.toml`

Intent:

- Qwen3.5 2B student;
- Hosted `loss = "sft"`;
- Google teacher via OpenRouter:
  `google/gemini-3-flash-preview`;
- full `mix_v2_p5050` training taskset;
- 4096 generation cap;
- W&B enabled;
- checkpoints/adapters enabled.

Launch command shape:

```bash
prime --plain train \
  configs/prime/qwen35_2b_hosted_sft_gemini_env0314_full.toml \
  --env-var OPENROUTER_API_KEY \
  --env-var WANDB_API_KEY \
  --output json -y
```

Result:

```text
HTTP 403: loss='sft' is currently restricted to beta users
```

Conclusion:

- Hosted SFT is a Prime account/product-gate blocker, not a local-machine
  blocker.
- Do not spend more time trying to launch Hosted SFT until the account has SFT
  beta access or Prime support enables it.
- Keep the config for when access is enabled.

### Full Hosted RL Run Launched

Because Hosted SFT is beta-blocked and the user requested a full run instead
of more smoke iteration, launched full Hosted RL on the backend that already
completed a Prime run:

Config:
`configs/prime/llama32_3b_p5050.toml`

Run:
`zztqgqclh3y3hslpjsofzpcf`

Run name:
`humanize-p5050-llama32-3b`

W&B run name:
`prime-llama32-3b-p5050`

Model:
`meta-llama/Llama-3.2-3B-Instruct`

Training setup:

- loss: `rl`
- env: `jayshah5696/humanize-rl-env@0.3.14`
- train args:
  `{ split = "train", task_set = "mix_v2_p5050", reward_mode = "p50_50_no_penalty" }`
- max steps: `200`
- batch size: `256`
- rollouts per example: `16`
- max generation tokens: `4096`
- learning rate: `8e-5`
- LoRA alpha: `32`

Eval setup:

- interval: `50`
- examples: `64`
- rollouts per example: `2`
- eval base model: `true`
- eval max tokens: `4096`
- eval envs:
  - `mix_v2_p5050` validation with `p50_50_no_penalty`
  - `v02_smoke` validation with `strict`
  - `v03` validation with `strict`

Validation:

- interval: `25`
- examples: `128`
- rollouts per example: `1`

Launch result:

- status: `PENDING`
- created at: `2026-06-23 00:33:35 UTC`
- Prime accepted config and started the hosted run.

First log check:

```text
Hosted Training run is starting; waiting for orchestrator logs...
Resolving 1 environment...
Found jayshah5696/humanize-rl-env@0.3.14
Installing jayshah5696/humanize-rl-env@0.3.14 with uv...
Resolved 163 packages in 14.48s
```

First progress check:

```json
{
  "latest_step": null,
  "steps_with_samples": [],
  "steps_with_distributions": []
}
```

First metrics check:

```json
{ "metrics": [] }
```

Interpretation:

- run has launched and environment installation began;
- no training step had completed at first poll;
- continue polling logs/progress/metrics/checkpoints until completion or a
  hard failure.

Second poll:

- status: `RUNNING`
- started at: `2026-06-23 00:34:04 UTC`
- components:
  - orchestrator: `RUNNING`
  - train env-server: `RUNNING`
  - all three eval env-servers: `RUNNING`
- W&B:
  `https://wandb.ai/jayshah5696/humanize-rl/runs/akzopsz9`

Step 0 base eval:

- `eval_mix_v2_p5050`: `0.4510`
- `eval_v02_strict`: `-0.1526`
- `eval_v03_strict`: `-0.3422`
- error rate: `0.0%`
- truncation rate: `0.0%`

Training steps observed:

- step `0`: reward `0.4568`, trainable `256/256`, truncation `0.0%`
- step `1`: reward `0.3596`, trainable `256/256`, truncation `0.0%`
- step `2`: reward `0.2764`, trainable `256/256`, truncation `0.0%`

Usage at early poll:

- total tokens: `964.46K`
- total cost: `$0.11`

Initial rollout audit:

```bash
prime --plain train rollouts zztqgqclh3y3hslpjsofzpcf --step 0 --num 20
```

First 20 step-0 samples:

- emoji hits: `0`
- wrapper-like regex hits in raw sample JSON: `3`
- all-caps regex hits in raw sample JSON: present, partly from source/prompt
  subject lines and partly from completion scaffolding.

Interpretation:

- base policy still emits formal email scaffolding on some compression tasks;
- this is step 0/base behavior, not learned behavior yet;
- do not stop the run for this; compare step 50/100/150/200 rollouts and evals
  to see whether RL reduces those leaks or over-rewards them.

Third poll:

- latest step: `17`
- samples logged: steps `0`, `10`
- distributions logged: steps `0`, `10`
- components healthy:
  - orchestrator: `RUNNING`
  - all env-servers: `RUNNING`
- usage: `4.10M` tokens, `$0.48`

Observed step progression:

- step `3`: reward `0.3632`, trainable `256/256`, truncation `0.0%`
- step `4`: reward `0.4125`, trainable `256/256`, truncation `0.0%`
- step `5`: reward `0.3649`, trainable `256/256`, truncation `0.0%`
- step `6`: reward `0.4266`, trainable `256/256`, truncation `0.0%`
- step `7`: reward `0.3800`, trainable `256/256`, truncation `0.0%`
- step `8`: reward `0.3202`, trainable `256/256`, truncation `0.0%`
- step `9`: reward `0.4462`, trainable `256/256`, truncation `0.0%`
- step `10`: reward `0.3683`, trainable `256/256`, truncation `0.0%`
- step `11`: reward `0.3427`, trainable `256/256`, truncation `0.0%`
- step `12`: reward `0.3267`, trainable `256/256`, truncation `0.4%`
- step `13`: reward `0.4439`, trainable `256/256`, truncation `0.0%`
- step `14`: reward `0.3191`, trainable `256/256`, truncation `0.0%`
- step `15`: reward `0.4517`, trainable `256/256`, truncation `0.0%`
- step `16`: reward `0.3363`, trainable `256/256`, truncation `0.0%`
- step `17`: reward `0.4391`, trainable `256/256`, truncation `0.0%`

Step-10 rollout audit:

```bash
prime --plain train rollouts zztqgqclh3y3hslpjsofzpcf --step 10 --num 50
```

Completion-only regex counts over 50 samples:

- emoji: `0`
- all-caps token hits: `0`
- wrapper-like hits: `5`
- subject-line hits: `8`
- standalone signoff hits: `3`

The formal scaffold examples had low rewards, roughly `0.22-0.33`, so the
current reward appears to be penalizing rather than reinforcing them. Continue
run.

### Follow-up RL Template

Config added:
`configs/prime/qwen35_2b_p5050_after_sft_full_template.toml`

Purpose:

- ready template for Qwen3.5 2B RL after an SFT checkpoint exists;
- currently not launched because Hosted SFT is beta-blocked and no READY SFT
  checkpoint exists.

### Step 50 Gate

Poll result:

- latest step: `53` at first step-50 poll, then logs showed through step `56`;
- status: `RUNNING`;
- samples/distributions logged at steps `0`, `10`, `20`, `30`, `40`, `50`;
- usage: `12.73M` tokens, `$1.49`;
- checkpoint:
  - id: `auf3yfqgvdlfcfyc9l3hu51e`
  - step: `50`
  - status: `UPLOADING`

Step 50 eval:

- `eval_mix_v2_p5050`: `0.4489`
  - step 0 was `0.4510`, so roughly flat.
- `eval_v02_strict`: `-0.1075`
  - step 0 was `-0.1526`, improvement `+0.0451`.
- `eval_v03_strict`: `-0.1498`
  - step 0 was `-0.3422`, improvement `+0.1924`.
- error rate: `0.0%`
- truncation rate: `0.0%`

Step 50 training:

- reward: `0.4055`
- trainable: `256/256`
- error: `0.0%`
- truncation: `0.0%`

Step-50 rollout audit:

```bash
prime --plain train rollouts zztqgqclh3y3hslpjsofzpcf --step 50 --num 50
```

Completion-only regex counts:

- emoji: `0`
- all-caps: `0`
- subject-line: `0`
- standalone signoff: `0`
- wrapper-like: `12`

Reward summary over 50 sampled completions:

- min: `0.1961`
- mean: `0.4055`
- max: `0.7997`

Interpretation:

- operationally healthy at step 50;
- strict eval improved, especially v03;
- p50 validation is flat so far;
- formal subject/signoff/all-caps/emoji leakage is not present in the sampled
  step-50 completions;
- wrapper-like phrasing remains and must be checked again at step 100/150/200;
- continue full run.

### Full Hosted RL Step 100/120 Gate

Run:
`zztqgqclh3y3hslpjsofzpcf`

Status at `2026-06-23 00:56 UTC`:

- Prime status: `RUNNING`
- latest step: `123`
- samples logged through step `120`
- checkpoints:
  - step `50`: `auf3yfqgvdlfcfyc9l3hu51e`, `READY`
  - step `100`: `kwcsyl3ybqkintiqy4spm5b8`, `READY`
- usage at latest poll:
  - total tokens: `30.57M`
  - cost: `$3.63`

Step 100 eval:

- `eval_mix_v2_p5050`: `0.613097`
  - step 0 was `0.4510`
  - step 50 was `0.4489`
  - interpretation: real p50 validation jump by step 100.
- `eval_v02_strict`: `0.032065`
  - step 0 was `-0.1526`
  - step 50 was `-0.1075`
  - interpretation: strict v02 is now positive.
- `eval_v03_strict`: `-0.094165`
  - step 0 was `-0.3422`
  - step 50 was `-0.1498`
  - interpretation: still negative, but much improved from base.
- eval error: `0%`
- eval truncation:
  - p50: `0.8%`
  - v02 strict: `0%`
  - v03 strict: `9.4%`

Step 100 training:

- reward: `0.558190`
- trainable: `256/256`
- error: `0%`
- truncation: `1.2%`
- decode length mean: `208.63`
- repetition filter: `0.8%`

Step 100 rollout artifact:
`runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step100.json`

Step 100 rollout audit over 64 Prime API samples:

- reward min/mean/max:
  `0.355930` / `0.587121` / `0.992678`
- word length min/mean/max:
  `13` / `144.69` / `711`
- emoji: `0`
- all-caps token hits: `3`
- wrapper-like hits: `6`
- subject-line hits: `0`
- standalone signoff hits: `0`
- samples `>=0.75`: `15`
- high-reward samples with emoji/all-caps/wrapper: `0`

Important qualitative finding:

- The high-reward failures are no longer emoji, subject lines, signoffs, or
  explicit wrappers.
- The new problem is forced casualness and semantic thinness. Examples scoring
  around `0.95-0.99` include lines like:
  - `We got a meeting about Software Dynamics' partnership...`
  - `We got some bugs on the Project thing...`
  - `We did a supply chain update... and stuff.`
- This is not a hosted-infra blocker. It is a reward/data quality blocker for
  final model selection: p50 is overvaluing rough casual register when it
  should prefer natural, direct prose.

Step 100-120 training trend:

- rewards generally moved into the `0.52-0.62` range after step `100`.
- step `107` had training truncation `9.0%`, then later steps mostly returned
  to `0-1.2%`.
- step `112` and `117` trained only `240/256` samples; step `123` trained
  `224/256`.
- no training errors were reported in this window.

Decision:

- Continue the full hosted run to step `200`; do not stop it mid-run because
  aggregate evals are improving and no emoji/all-caps/wrapper exploit is
  dominating high reward.
- Treat step `150` and step `200` as real gates. If the forced-casual drift or
  v03 truncation/length drift worsens, the final decision should be "training
  run completed but model not accepted," followed by a targeted reward/SFT data
  fix for fake-casual phrases and long-form truncation.

### Full Hosted RL Step 150 Gate

Status:

- latest step at gate poll: `153`
- checkpoint:
  - step `150`: `znntfykvg2koon8nrs73izq7`, `READY`
- usage after the gate:
  - total tokens: `40.52M`
  - cost: `$4.87`

Step 150 eval:

- `eval_mix_v2_p5050`: `0.662144`
  - step 0: `0.4510`
  - step 50: `0.4489`
  - step 100: `0.6131`
  - read: aggregate p50 keeps improving.
- `eval_v02_strict`: `0.175574`
  - step 0: `-0.1526`
  - step 50: `-0.1075`
  - step 100: `0.0321`
  - read: strict v02 has a large positive delta.
- `eval_v03_strict`: `0.0044`
  - step 0: `-0.3422`
  - step 50: `-0.1498`
  - step 100: `-0.0942`
  - read: strict v03 is finally positive, but barely.
- eval error: `0%`
- eval truncation:
  - p50: `0%`
  - v02 strict: `0%`
  - v03 strict: `1%`

Step 150 training:

- reward: `0.672527`
- trainable: `256/256`
- error: `0%`
- truncation: `0%`
- decode length mean: `117.28`

Step 150 rollout artifacts:

- rollouts:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step150.json`
- audit:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step150.json`

Local audit over 64 samples:

- reward mean/max:
  `0.668170` / `0.998734`
- failed diagnostics:
  - `missing_entity=17`
  - `sentence_window=16`
  - `too_long=14`
  - `missing_must_include_phrase=10`
  - `missing_required_fact=8`
  - `forbidden_phrase=4`
  - `repetition=4`
  - `option_menu=1`
- high reward with failed diagnostics: `0`
- high reward with emoji: `0`
- high reward with all-caps: `0`
- high reward with option/wrapper: `0`

Scientific finding:

- The metric gates improved, but the top samples expose a new reward hole.
- Near-`1.0` samples include broken or fake-casual prose:
  - `We hit project timeline delay because technical stuff we didn't plan for.`
  - `We working on it now, we tell you when stuff good again.`
  - `Hey guys, we got good stuff going on here.`
  - `We're doing you a solid here...`
- Existing diagnostics catch emoji, signoffs, wrappers, length, and many
  faithfulness failures, but they do not penalize low-specificity/fake-casual
  filler such as `stuff`, `you guys`, `we got`, `we're good`, and broken
  grammar used as a shortcut to sound informal.

Decision:

- Continue to the requested full `200` steps, because this run is valuable as a
  complete hosted RL experiment.
- Do not accept the model solely on aggregate p50/strict improvements.
- If step `200` shows the same samples, the next fix is not another longer
  run. The next fix is a reward/data patch:
  - add fake-casual and low-specificity diagnostics;
  - add a grammar/naturalness cap separate from ridge;
  - add SFT repair rows where the answer is plain and direct, not slangy;
  - rerun a shorter RL smoke before another full target run.

### Full Hosted RL Final Result

Run:
`zztqgqclh3y3hslpjsofzpcf`

Final status:

- `COMPLETED`
- started: `2026-06-23 00:34:04 UTC`
- completed: `2026-06-23 01:11:13 UTC`
- orchestrator loop duration in logs: `35m 37s`
- W&B:
  `https://wandb.ai/jayshah5696/humanize-rl/runs/akzopsz9`
- W&B run name:
  `prime-llama32-3b-p5050`

Final usage:

- training tokens: `25,333,221`, cost `$3.8002`
- inference tokens: `26,124,677`, cost `$2.3697`
- total tokens: `51,457,898`
- total cost: `$6.1699`

Final checkpoint:

- Prime checkpoint id:
  `arsnu29hb9akbm2jc1b33pmc`
- step: `200`
- status: `READY`
- size: `390,246,137` bytes
- uploaded: `2026-06-23 01:11:21 UTC`

Prime deployment/model records:

- step `200` model record:
  `mc5y9fkiv3vkzpe053ssa8c9`
  - status at first poll: `UPLOADING`
  - deployment status: `NOT_DEPLOYED`
- step `199` model record:
  `fozdn0tw8of9ygus5zrltucs`
  - status at first poll: `UPLOADING`
  - deployment status: `NOT_DEPLOYED`
- same-run no-step record:
  `ws8vx0xxiozlagq2cca2b70t`
  - status at first poll: `PENDING`
  - deployment status: `NOT_DEPLOYED`

Saved final artifacts:

- run metadata:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/run_get_final.json`
- usage:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/usage_final.json`
- checkpoints:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/checkpoints_final.json`
- metrics:
  `runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/metrics_0_205.json`
- rollout/audit snapshots:
  `step100`, `step120`, `step130`, `step140`, `step150`, `step160`,
  `step170`, `step180`, `step190`

Final eval table:

| Step | Train reward | p50 validation | strict v02 | strict v03 | train truncation | v03 eval truncation |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | `0.456776` | `0.451041` | `-0.152610` | `-0.342210` | `0.0%` | `0.0%` |
| 50 | `0.405465` | `0.448943` | `-0.107508` | `-0.149771` | `0.0%` | `0.0%` |
| 100 | `0.558190` | `0.613097` | `0.032065` | `-0.094165` | `1.2%` | `9.4%` |
| 150 | `0.672527` | `0.662144` | `0.175574` | `0.004391` | `0.0%` | `1.0%` |
| 200 | n/a | `0.676841` | `0.118504` | `0.216975` | n/a | `1.0%` |

Metric verdict:

- passed hosted execution;
- passed cost/runtime expectations;
- passed final aggregate p50 improvement;
- passed final strict v02 and v03 aggregate improvement;
- no final eval truncation collapse;
- no emoji/all-caps/option-wrapper high-reward exploit in audited samples.

Quality verdict:

- failed model acceptance.
- The reward was exploited by low-specificity fake-casual prose.
- Late-run high-reward examples repeatedly used:
  - `stuff`
  - `we got`
  - `you guys`
  - `thanks`
  - broken casual grammar such as `we tell you when stuff good again`
- Step `180` examples scored `1.0` while saying:
  - `We got stuff we need for Project too.`
  - `Stuff's done now, look at stuff we did in shared folder.`
  - `We got stuff too - Project deliverables are done, too.`
- Step `190` examples still scored near `1.0` while saying:
  - `We got us an artist residency in Sri Lanka... personal and professional stuff`
  - `We at Digital Dynamics want you at Advanced Analytics Inc. We think we should do some stuff...`

Scientific conclusion:

This was the correct full hosted Prime run, but it is a rejected checkpoint.
It proves Prime Hosted RL works for this env/model/dataset and it proves the
current p50 reward is incomplete. The missing dimension is not emojis, all-caps,
or wrapper behavior anymore; it is fake-casual, low-information prose that the
ridge/deterministic blend mistakes for naturalness.

Next action before any more full training:

1. Add deterministic diagnostics/caps for fake-casual and low-specificity prose:
   `stuff`, repeated `we got`, repeated `you guys`, filler thanks, broken
   informal grammar, and vague placeholders replacing real facts.
2. Add targeted SFT repair rows from the failed high-reward samples: same
   prompts, plain/direct references, no slang.
3. Re-score saved rollouts from steps `140`, `150`, `170`, `180`, and `190`
   against the patched reward. The bad `1.0` examples should fall below the
   acceptance band before any new RL run.
4. Run a short hosted RL smoke after patching, not another 200-step run.
5. Only run the next full target after the short smoke passes qualitative
   rollout audit and family/mode gates.
