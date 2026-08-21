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

## 2026-06-24: Qwen Ablation Plan Update

User request:

- Update the ablation document for the next study.
- Define ablations across:
  - RL env updates;
  - SFT updates;
  - RL training after SFT.
- Focus on two Qwen targets:
  - `Qwen/Qwen3.5-2B`;
  - `Qwen/Qwen3.5-9B`.

Live Prime model check:

```bash
prime --plain train models --output json
```

Relevant result:

| Model | At capacity | Training $/M tok | Input $/M tok | Output $/M tok |
|---|---:|---:|---:|---:|
| `Qwen/Qwen3.5-2B` | no | `0.15` | `0.05` | `0.15` |
| `Qwen/Qwen3.5-9B` | yes | `0.60` | `0.20` | `0.60` |

Decision:

- Treat `Qwen/Qwen3.5-2B` as the next runnable primary target.
- Keep `Qwen/Qwen3.5-9B` as the second model for the same ablation, but do not
  launch until Prime capacity clears.
- Do not silently replace the requested 9B comparison with 4B or MoE. If 9B
  remains at capacity, log the blocker and finish the 2B branch.

Documentation update:

- Updated:
  `docs/plans/humanize_rl_ablation_and_next_actions.md`
- Added a dedicated "Next Two-Model Qwen Ablation Track" covering:
  - env-only reward ablations E0 to E6;
  - SFT variants S0 to S4;
  - RL-after-SFT variants R0 to R4;
  - the Qwen 2B/9B comparison matrix;
  - bad-output to target-repair examples;
  - stop conditions for reward hacking, family regression, truncation, and Qwen
    NaN/stall behavior.

Current scientific gate remains unchanged:

1. Patch reward diagnostics first.
2. Offline-rescore saved Llama rollout failures.
3. Build targeted repair SFT rows.
4. Prove Qwen 2B SFT improves base on frozen prompts.
5. Run short RL smoke from base and from SFT.
6. Run 200-step Qwen 2B RL-after-SFT only if the short smoke passes.
7. Repeat on Qwen 9B only after Prime capacity clears.

## 2026-06-24: Ignored Research Artifacts Archived

User concern:

- The branch is pushed, but generated run/data artifacts are ignored.
- If this worktree is deleted, the next researcher still needs the reward-hack
  evidence, offline rescore inputs, repair-reference rows, and SFT source files.

Archived minimal continuation set:

```text
runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/
runs/prime_eval_smoke/llama32_3b_failure_set_env0314.jsonl
runs/prime_eval_smoke/llama32_3b_rl_vs_base_summary_env0314.json
runs/reports/prime_mix_v2_p5050_taskset_report.json
data/processed/sft/reference_targets/
data/processed/v04_sft_final_plus_llama_failure_refs_env0314*
```

Not archived:

- caches;
- `.venv`;
- `__pycache__`;
- `.coverage`;
- env wheel build output, because `jayshah5696/humanize-rl-env@0.3.14` is
  published on Prime.

HF artifact dataset:

- repo:
  `jayshah5696/humanize-rl-research-artifacts-env0314`
- URL:
  `https://huggingface.co/datasets/jayshah5696/humanize-rl-research-artifacts-env0314`
- upload commit:
  `09457c42a56915ce649370fd4f42da92c0bf1080`
- files verified by dry run:
  `45`
- total size verified by dry run:
  `21.5M`

Verification commands:

```bash
hf datasets info jayshah5696/humanize-rl-research-artifacts-env0314 --format json
hf download jayshah5696/humanize-rl-research-artifacts-env0314 --type dataset --dry-run
```

Dry-run result:

- all `45` files are downloadable;
- archive contains `README.md`, `manifest.json`, saved rollout/audit files,
  failure set, eval summary, taskset report, repair-reference rows, and local
  SFT source JSONL/report files.

Restore command for a future worktree:

```bash
hf download jayshah5696/humanize-rl-research-artifacts-env0314 \
  --type dataset \
  --local-dir artifacts/humanize-rl-research-artifacts-env0314
```

## 2026-06-26: Fake-Casual Reward Patch and Env 0.3.15 Candidate

User request:

- Continue the Prime RL goal from
  `docs/plans/humanize_rl_ablation_and_next_actions.md`.
- Use the RL ablation study to improve env quality before more SFT/RL.
- Keep `log.md` and decisions current.
- Consider Pangram-style detector quality, but keep the Prime path first.

State reconstruction:

- Current accepted direction remains Prime-first.
- The completed Llama 3.2 3B Prime run
  `zztqgqclh3y3hslpjsofzpcf` is rejected as a model candidate because it
  learned low-specificity fake-casual prose.
- The next gate was not a full training run. It was offline reward patching and
  saved-rollout rescoring.

Implementation:

- Added deterministic diagnostics and p50 caps for:
  - `fake_casual_phrase`;
  - `low_specificity_substitution`;
  - `broken_informal_grammar`;
  - `register_mismatch`;
  - `thanks_padding`.
- Added focused tests for:
  - `stuff`, repeated `we got`, `you guys`, and fake-casual filler;
  - source facts replaced by vague placeholders;
  - broken informal grammar such as `we tell you when stuff good again`;
  - slang/register mismatch such as `doing you a solid` and `cut the crap`;
  - `thanks for looking` / `thanks for looking out` padding;
  - clean casual Slack controls.
- Mirrored the reward patch into `environments/humanize_rl_env`.
- Bumped the local env package version to `0.3.15`.

Artifact restore:

```bash
hf download jayshah5696/humanize-rl-research-artifacts-env0314 \
  --type dataset \
  --local-dir artifacts/humanize-rl-research-artifacts-env0314
```

Restored:

- saved rollouts/audits for `zztqgqclh3y3hslpjsofzpcf`;
- `llama32_3b_failure_set_env0314.jsonl`;
- repair reference rows;
- SFT source artifacts for the env0314 dataset.

Offline rescore command pattern:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-audit-min uv run --no-project \
  --python 3.12 \
  --with pydantic \
  --with click \
  --with numpy \
  --with scikit-learn==1.8.0 \
  python scripts/eval/audit_prime_rollouts.py \
  --rollouts artifacts/humanize-rl-research-artifacts-env0314/runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/rollouts_step180.json \
  --output runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step180_env0315_candidate.json \
  --top-n 10
```

Offline rescore result:

| Step | Rows | Mean | Max | High samples `>=0.75` |
|---:|---:|---:|---:|---:|
| 140 | `64` | `0.493312` | `0.500000` | `0` |
| 150 | `64` | `0.490982` | `0.500000` | `0` |
| 160 | `64` | `0.487403` | `0.500000` | `0` |
| 170 | `64` | `0.501132` | `0.696955` | `0` |
| 180 | `64` | `0.500554` | `0.657759` | `0` |
| 190 | `64` | `0.495079` | `0.500000` | `0` |

Important patch iteration:

- First rescore found two step-170 high survivors with no diagnostics:
  - `before day ends`;
  - `thanks for looking`.
- Patched both:
  - `before day ends` now triggers `broken_informal_grammar`;
  - semicolon-separated `thanks for looking` now triggers `thanks_padding`.
- After patching, no top saved sample from steps `140-190` remained at or
  above `0.75`.

Clean-control check:

- Scored the `20` archived clean repair references.
- New diagnostics false-positive rows: `0/20`.
- Only `7/20` controls scored `>=0.75`, but those drops came from older
  diagnostics such as `subject_line`, `missing_entity`, and
  `invented_temporal_detail`, not the new fake-casual patch.
- Decision: create or refresh a direct/plain clean-control set before a hosted
  training launch. Do not use the archived formal repair rows as the `90%`
  clean-direct acceptance set.

Verification:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-test-min uv run --no-project \
  --python 3.12 \
  --with pytest \
  --with pydantic \
  pytest -o addopts='' tests/reward/test_checks.py tests/reward/test_reward.py -q
```

Result:

- `58 passed in 0.24s`.

Environment caveat:

- Plain `uv run pytest ...` created a CPython `3.14` env and failed because
  `spacy==3.8.14` has no compatible wheel for `cp314`.
- `UV_PROJECT_ENVIRONMENT=.venv312 uv run --python 3.12 pytest ...` then hit a
  local `fasttext-wheel` build/link failure because the Mac points at a missing
  Xcode SDK.
- The focused reward tests do not need spaCy or fastText, so the validated path
  used `uv run --no-project` with only `pytest` and `pydantic`.

Pangram-style detector decision:

- Web check:
  - Pangram markets an AI detector for ChatGPT/Gemini/Claude and similar text.
  - Pangram exposes an API with realtime and bulk/batch analysis options.
  - Pangram also publishes SDK docs.
- Decision:
  - Do not call Pangram or any live external detector inside the Prime reward
    env.
  - Treat Pangram-style behavior as an external eval rail or frozen mimic set,
    not a training reward dependency.
  - Rationale: live detector calls would make hosted rewards slower, costlier,
    and non-reproducible. Prime reward should stay self-contained.

Current decision:

- The fake-casual reward hole is closed for the saved Llama late-run failures.
- Do not launch a 200-step Qwen run yet.
- Next Prime actions:
  1. Build and publish env `0.3.15`.
  2. Verify wheel contents include `ridge_state.pkl`.
  3. Run local Prime eval smoke on `0.3.15`.
  4. Refresh a direct/plain clean-control eval set.
  5. Run only a 20-50 step hosted smoke before any Qwen 2B full run.

### Env 0.3.15 Published and Local Smoke

Prime env publish:

- env: `jayshah5696/humanize-rl-env@0.3.15`
- created: `2026-06-26 23:13:37 UTC`
- Prime content hash: `51a8c3ea`
- wheel:
  `humanize_rl_env-0.3.15-py3-none-any.whl`
- wheel SHA256:
  `56bffe065a8296052d7aac8153037b3e80c7738ed6f6b20a097289853d99d6a2`

Pre-publish wheel check:

- built with `uv build environments/humanize_rl_env`;
- verified wheel contains:
  - `humanize_rl_env/ridge_state.pkl`;
  - `humanize_rl_env/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`;
  - `humanize_rl_env/reward/checks.py`;
  - `humanize_rl_env/reward/reward.py`.

Registry verification:

```bash
prime env version list jayshah5696/humanize-rl-env
```

Result:

- `0.3.15` is latest;
- artifacts: `2`;
- install command:
  `prime env install jayshah5696/humanize-rl-env@0.3.15`.

Local smoke command:

```bash
prime --plain eval run jayshah5696/humanize-rl-env \
  --provider prime \
  --model Qwen/Qwen3.5-0.8B \
  --env-args '{"split":"validation","task_set":"mix_v2_p5050","reward_mode":"p50_50_no_penalty"}' \
  --num-examples 3 \
  --rollouts-per-example 1 \
  --max-tokens 1024 \
  --temperature 0.2 \
  --skip-upload \
  --disable-env-server \
  --disable-tui \
  --max-retries 0 \
  --save-results \
  --output-dir runs/prime_eval_smoke/qwen35_08b_p5050_env0315
```

Output:

- results:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env0315/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/b003a93c/results.jsonl`
- examples / rollouts: `3 / 1`
- reward avg/std: `0.557 / 0.221`
- rewards: `[0.484, 0.329, 0.857]`
- truncation: `0%`
- scoring latency mean: `12ms`
- token usage avg:
  - input: `147.333`
  - output: `54.000`

Smoke interpretation:

- Published env loads and scores correctly.
- The low scores are appropriate for outputs that retain formal AI/template
  phrasing or ignore length.
- The direct hotfix output scored high:
  `Just finished deploying the hotfix to production and everything is running fine.`
- This is a technical env smoke, not approval for full training.

Updated next actions:

1. Refresh a direct/plain clean-control set for the `>=90% above 0.75` gate.
2. Run a 20-50 step hosted smoke on `0.3.15`.
3. Only then resume Qwen 2B SFT/RL comparison.

### Direct Clean-Control Gate for Env 0.3.15

Purpose:

- The archived clean repair rows were not a good control set because several
  were formal-letter repairs that intentionally still triggered older
  diagnostics such as `subject_line` or `missing_entity`.
- Built a smaller direct/plain control report from existing validation tasks
  to check whether the new fake-casual diagnostics over-penalize good plain
  answers.

Report:

```text
runs/prime_eval_smoke/direct_clean_controls_env0315.json
```

Result:

| Metric | Value |
|---|---:|
| rows | `10` |
| mean reward | `0.852037` |
| min reward | `0.469274` |
| max reward | `0.975566` |
| rows `>=0.75` | `9/10` |
| new diagnostic false-positive rows | `0/10` |

Interpretation:

- The direct-control gate passes the minimum `90%` threshold.
- The single low row failed an older `missing_entity` diagnostic, not the new
  fake-casual patch.
- The new diagnostics did not fire on clean direct controls.

Decision:

- Env `0.3.15` has passed:
  - saved bad rollout offline rescore;
  - local Prime eval smoke;
  - direct clean-control gate.
- Next allowed action is a short hosted smoke, not a full Qwen run.

### Hosted RL Smoke Launch for Env 0.3.15

Date: 2026-06-27 PDT / 2026-06-28 UTC

Config:

```bash
prime --plain train configs/prime/llama32_1b_fakecasualfix_env0315_smoke50.toml \
  --yes --output json
```

Decision:

- Skipped `sprints/Llama-3.2-1B-Instruct` despite the zero price because
  `configs/prime/README.md` records that it rejects this custom env.
- Used `meta-llama/Llama-3.2-1B-Instruct`, the cheapest same-family fallback
  currently available in `prime train models --output json`.
- Did not include a `[wandb]` section because
  `/private/tmp/humanize_rl_prime_wandb.env` is absent and `WANDB_API_KEY` is
  not set in the current shell.
- Kept hosted smoke small: `50` steps, batch `16`, `4` rollouts per example,
  max inflight `16`, max tokens `1024`.

Run:

- Prime run ID: `fj9oinokvx5zott096tgfwqw`
- Name:
  `humanize-p5050-fakecasualfix-llama32-1b-smoke50-env0315-r1`
- Env: `jayshah5696/humanize-rl-env@0.3.15`
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Launch status: `PENDING`, then `RUNNING` on first monitor poll.

Monitoring blocker:

- After launch, authenticated Prime endpoints started returning:
  `API key unauthorized`.
- Affected commands included `prime train get`, `prime train progress`,
  `prime train list`, `prime whoami`, and `prime wallet`.
- Current blocker is local Prime auth, not env scoring. Re-run `prime login` or
  refresh the saved Prime token, then resume with:

```bash
prime --plain train get fj9oinokvx5zott096tgfwqw --output json
prime --plain train progress fj9oinokvx5zott096tgfwqw
```

### Pangram-Style Detector-Mimic Gate v01

Date: 2026-06-28

Decision:

- Keep Pangram-style AI-detector behavior as an external eval rail, not as a
  hosted Prime reward dependency.
- Public Pangram docs describe async text detection with `prediction_short`,
  `fraction_ai`, `fraction_ai_assisted`, `fraction_human`, and window-level
  segment results. The local mimic mirrors that evidence shape but uses frozen
  deterministic heuristics so future SFT/RL comparisons are reproducible.
- Do not call a live detector from the reward env.
- Added an offline Pangram-export comparison tool so real detector exports can
  calibrate the mimic without making training nondeterministic.

Files:

- Frozen eval set: `data/eval/detector_mimic_v01.jsonl`
- Scorer module: `src/humanize_rl/scoring/detector_mimic.py`
- Click runner: `scripts/eval/evaluate_detector_mimic.py`
- Pangram-export comparator:
  `scripts/eval/compare_detector_mimic_to_pangram.py`
- Tests:
  - `tests/scoring/test_detector_mimic.py`
  - `tests/scripts/test_evaluate_detector_mimic.py`
  - `tests/scoring/test_pangram_alignment.py`
  - `tests/scripts/test_compare_detector_mimic_to_pangram.py`

Gate command:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-detector-min uv run --no-project \
  --python 3.12 \
  --with pydantic \
  --with click \
  scripts/eval/evaluate_detector_mimic.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --output runs/detector_mimic/detector_mimic_v01_report.json \
  --scored-output runs/detector_mimic/detector_mimic_v01_scored.jsonl
```

Optional real Pangram export comparison:

```bash
uv run scripts/eval/compare_detector_mimic_to_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --pangram-output runs/detector_mimic/pangram_export.json \
  --output runs/detector_mimic/pangram_alignment_report.json
```

Result:

| Metric | Value |
|---|---:|
| rows | `22` |
| human controls | `8` |
| nonhuman controls | `14` |
| false positives | `0` |
| false negatives | `0` |
| mean `fraction_ai` | `0.452857` |
| max `fraction_ai` | `0.980000` |
| gate | `pass` |

Read:

- Plain direct controls all stayed `Human`.
- Template AI tells, fake-casual reward hacks, and mixed artifacts were all
  flagged as `Mixed` or `AI`.
- The 2026-06-29 hardening added literal `seamless` / `unlock` human controls,
  mixed wrapper segments, stacked corporate boilerplate, and a fake-casual plus
  corporate-gloss reward hack. Single contextual corporate words now stay below
  the detector threshold unless they stack with other AI-style evidence.
- This gate is now available before promoting any full SFT/RL candidate.

Validation:

- Focused tests:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-test-min uv run --no-project \
  --python 3.12 \
  --with pytest \
  --with pydantic \
  --with click \
  pytest -o addopts='' \
  tests/reward/test_checks.py \
  tests/reward/test_reward.py \
  tests/scoring/test_detector_mimic.py \
  tests/scripts/test_evaluate_detector_mimic.py -q
```

- Result: `66 passed in 0.23s`.
- `ruff check` on touched reward, detector-mimic, script, and test files:
  passed.
- Smoke config TOML check: passed.
- `git diff --check`: passed.
- Final Prime hosted-training retry still returned `API key unauthorized`, so
  rollout audit for `fj9oinokvx5zott096tgfwqw` remains pending.

References:

- Pangram REST API quickstart:
  `https://docs.pangram.com/quickstart-rest`
- Pangram Python SDK response fields:
  `https://docs.pangram.com/sdk/python`

### Qwen 2B Env 0.3.15 SFT/RL Ablation Configs

Date: 2026-06-28

Purpose:

- Prepare the Prime-first full-model path while Prime auth is blocked.
- Do not launch another hosted run until `fj9oinokvx5zott096tgfwqw` is audited.

Added configs:

- Qwen 0.8B renderer smoke:
  `configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50.toml`
- Qwen 2B base RL full template:
  `configs/prime/qwen35_2b_p5050_env0315_full200_template.toml`
- Qwen 2B SFT-to-RL full template:
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`
- Qwen 2B dataset SFT target:
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`

Run order after Prime auth refresh:

1. Audit hosted Llama smoke `fj9oinokvx5zott096tgfwqw`.
2. If it passes, launch Qwen 0.8B env0315 smoke.
3. If Qwen 0.8B smoke passes, run Qwen 2B dataset SFT.
4. Compare Qwen 2B base RL vs Qwen 2B SFT-to-RL.
5. Run the detector-mimic gate before promoting any candidate.

Key config choices:

- Kept env version pinned to `0.3.15`.
- Kept Qwen hosted training/eval `max_tokens = 1024`.
- Added explicit `max_inflight_rollouts`.
- Reduced Qwen 2B full rollout geometry from the old env0314 `256 x 16`
  shape to `64 x 8` with max inflight `32`.
- Did not include W&B in hosted RL configs because the current shell still has
  no `WANDB_API_KEY`; dataset SFT config keeps W&B metadata for tracked GPU
  runs without storing secrets.

Validation:

- Config tests:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-test-min uv run --no-project \
  --python 3.12 \
  --with pytest \
  --with pydantic \
  --with click \
  pytest -o addopts='' tests/config/test_prime_ablation_configs.py -q
```

- Result: `3 passed in 0.01s`.
- TOML parse check passed for all four new configs.

### Prime Auth Restored, Hosted Smoke Audits, and Qwen 2B Base RL Gate

Date: 2026-06-28

Prime auth:

- `prime --plain whoami` succeeded after token refresh.
- Run `fj9oinokvx5zott096tgfwqw` was reachable and had completed.

Added audit bundle:

- Script: `scripts/eval/audit_prime_run_bundle.py`
- Test: `tests/scripts/test_audit_prime_run_bundle.py`
- Purpose: combine saved rollout audits plus detector-mimic results into one
  promotion gate report:
  `runs/prime_training_smoke/<run_id>/bundle_report.json`.

Llama 1B hosted smoke:

- Run ID: `fj9oinokvx5zott096tgfwqw`
- Status: `COMPLETED`
- Model: `meta-llama/Llama-3.2-1B-Instruct`
- Env: `jayshah5696/humanize-rl-env@0.3.15`
- Cost: `$0.0389`
- Rollout audits: steps `0`, `10`, `20`, `30`, `40`
- Bundle gate: `pass`
- Detector-mimic gate: `pass`
- Eval delta, step `50` vs step `0`:
  - `mix_v2_p5050`: `-0.021137`
  - `v02_strict`: `+0.054831`
  - `v03_strict`: `+0.067670`
- Read: stable and no high-reward fake-casual exploit, but not a p50 quality
  win. Good enough as an exploit/stability smoke, not as a candidate.

Qwen 0.8B r1:

- Config: `configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50.toml`
- Run ID: `r8xlz0csp79fu0z0elp9h1dv`
- Status: `COMPLETED`
- Cost: `$0.0457`
- Bundle gate: `pass`
- Eval delta, step `50` vs step `0`:
  - `mix_v2_p5050`: `+0.069854`
  - `v02_strict`: `-0.064143`
  - `v03_strict`: `-0.051501`
- Read: p50 improved and reward-hack audit passed, but strict-family drops
  crossed the guardrail.

Qwen 0.8B r2:

- Config: `configs/prime/qwen35_08b_fakecasualfix_env0315_smoke50_lr5e5.toml`
- Run ID: `pzcfi8aew2pnxnkhec2augts`
- Status: `COMPLETED`
- Change vs r1: `learning_rate = 5e-5`
- Cost: `$0.0347`
- Bundle gate: `pass`
- Detector-mimic gate: `pass`
- Eval delta, step `50` vs step `0`:
  - `mix_v2_p5050`: `+0.022284`
  - `v02_strict`: `+0.111589`
  - `v03_strict`: `+0.125909`
- Read: this is the first Qwen env0315 smoke that clears both reward-hack and
  strict eval guardrails.

Qwen 2B base RL:

- Config: `configs/prime/qwen35_2b_p5050_env0315_full200_template.toml`
- Run ID: `o48ryskshkn06b3o1b1kauql`
- Status: `STOPPED`
- Stop reason: step-50 eval failed the strict-family guardrail.
- Latest step at stop: `55`
- Saved/audited sample steps: `0`, `10`, `20`, `30`, `40`, `50`
- Cost: `$0.4213`
- Rollout bundle gate: `pass`
- Detector-mimic gate: `pass`
- Eval delta, step `50` vs step `0`:
  - `mix_v2_p5050`: `+0.056013`
  - `v02_strict`: `-0.168230`
  - `v03_strict`: `-0.128226`
- Read: base RL can increase p50 while damaging strict generalization. Do not
  continue this 2B base-RL direction at `8e-5`; the next full-model path should
  be Qwen 2B dataset SFT first, then SFT-to-RL from the resulting checkpoint.

### Prime Qwen 2B Dataset SFT Launch Preflight

Date: 2026-06-28

Current state:

- Prime auth is restored: `prime --plain whoami` succeeds for user
  `jayshah5696`.
- Wallet balance is `$92.15`.
- No running Prime pods or sandboxes.
- Hosted Training models are available, including `Qwen/Qwen3.5-2B`, but
  Hosted Training `prime train` still exposes the env/rollout schema. Do not
  submit `configs/prime_rl/*.toml` through `prime train`.
- Local `uv run --no-sync sft --help` fails with `No such file or directory`
  because this repo does not install the open `prime-rl` package as a local
  dependency.
- Fresh Prime `prime-rl` source checkout:
  - commit: `d700753`
  - root `pyproject.toml` exposes `sft = "prime_rl.entrypoints.sft:main"`
  - dependency closure is Linux/CUDA-only for the full runtime.
- Secrets:
  - local `HF_TOKEN`: set
  - local `WANDB_API_KEY`: missing
  - local `PRIME_API_KEY`: missing, but Prime CLI token auth is valid
  - Prime global secret store: empty

Decision:

- Do not launch Qwen 2B dataset SFT until W&B tracking is available.
- The correct launch path remains Linux/CUDA Prime `prime-rl`:

```bash
uv run sft @ configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml
```

- Prime sandbox is the concrete remote runner once secrets are present. Use a
  GPU VM sandbox with a CUDA/PyTorch image, upload the current repo state,
  install open `prime-rl`, then run the SFT command against the uploaded config.

Added executable preflight:

- Script: `scripts/train/prime_sft_preflight.py`
- Test: `tests/scripts/test_prime_sft_preflight.py`
- Dependency: added `click` to project dependencies because new scripts use
  Click. Local full sync on this Mac is still blocked by the existing
  `fasttext-wheel`/Xcode SDK build issue, so validation here used an isolated
  `uv run --no-project --with click` runner.
- Live result:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
```

2026-06-28 continuation checks:

- Prime `prime-rl` source schema validation passed against current source
  commit `d700753`:
  - config:
    `configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml`
  - model: `Qwen/Qwen3.5-2B`
  - renderer: `qwen3.5`
  - train split: `train`
  - validation split: `validation`
  - `ckpt.weights.save_adapter_separately = true`
- HF Dataset Viewer check passed:
  - viewer/search/filter/statistics available
  - splits: `train`, `validation`, `test`
  - row counts: train `4313`, validation `239`, test `241`, total `4793`
  - train schema includes `messages`
  - parquet shards are available for all three splits
- Updated `scripts/train/prime_sft_preflight.py` with optional
  `--check-hf-viewer` so this dataset drift check can be rerun before launch.
- Live preflight with dataset check:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4313 validation=239 test=241
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
```

Secret setup command, from a shell where `WANDB_API_KEY` is already set:

```bash
prime --plain secret create \
  --name WANDB_API_KEY \
  --value "$WANDB_API_KEY" \
  --description "Weights and Biases tracking token"
```

### Prime Qwen 2B SFT Sandbox Launch Kit

Date: 2026-06-28

Added:

- Script: `scripts/train/prepare_prime_sft_launch_kit.py`
- Test: `tests/scripts/test_prepare_prime_sft_launch_kit.py`
- Script index update: `scripts/README.md`

Generated artifact:

- Launch kit directory:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315/`
- Archive:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315.tar.gz`
- Archive contents:
  - `prime_sft_launch_kit/README.md`
  - `prime_sft_launch_kit/config.toml`
  - `prime_sft_launch_kit/manifest.json`
  - `prime_sft_launch_kit/run_sft.sh`
- Config SHA256:
  `f14efebb82630df65dbbab8441c87c60f687e9672ee0dff035cfa4a8c2bfc65c`
- Pinned Prime `prime-rl` ref in runner: `d700753`

The launch kit is secret-free. It still requires `HF_TOKEN` and
`WANDB_API_KEY` to be passed at sandbox run time:

```bash
prime --plain sandbox upload <sandbox_id> \
  runs/prime_sft_launch_kit/qwen35_2b_env0315.tar.gz \
  /tmp/qwen35_2b_env0315.tar.gz
prime --plain sandbox run <sandbox_id> -- \
  bash -lc 'mkdir -p /workspace && tar -xzf /tmp/qwen35_2b_env0315.tar.gz -C /workspace'
prime --plain sandbox run <sandbox_id> \
  -e HF_TOKEN="$HF_TOKEN" \
  -e WANDB_API_KEY="$WANDB_API_KEY" \
  --timeout 86400 \
  -- bash -lc 'bash /workspace/prime_sft_launch_kit/run_sft.sh'
```

### Prime Qwen 2B SFT-to-RL Config Handoff

Date: 2026-06-28

Added:

- Script: `scripts/train/prepare_prime_sft_to_rl_config.py`
- Test: `tests/scripts/test_prepare_prime_sft_to_rl_config.py`

Purpose:

- Once the Qwen 2B dataset SFT run has a READY checkpoint, produce a concrete
  hosted RL config from
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml`.
- Refuse the placeholder checkpoint ID.
- Refuse templates where:
  - `model` is not `Qwen/Qwen3.5-2B`;
  - `checkpoint_id` is not the placeholder;
  - `eval.eval_base_model` is not `false`;
  - any train/eval env version is not `0.3.15`.

Template validation command used:

```bash
PYTHONPATH=src UV_PROJECT_ENVIRONMENT=.venv-test-min uv run --no-project \
  --python 3.12 \
  --with click \
  scripts/train/prepare_prime_sft_to_rl_config.py \
  --checkpoint-id ckpt_dummy_ready_for_template_validation \
  --allow-unverified-checkpoint \
  --output /tmp/qwen35_2b_after_sft_validation.toml \
  --run-name humanize-p5050-qwen35-2b-after-sft-validation
```

Result:

```text
config=/tmp/qwen35_2b_after_sft_validation.toml
checkpoint_id=ckpt_dummy_ready_for_template_validation
next=prime --plain train /tmp/qwen35_2b_after_sft_validation.toml --yes --output json
```

### Prime Qwen 2B SFT Promotion Gate

Date: 2026-06-28

Added:

- Script: `scripts/eval/build_sft_promotion_gate.py`
- Test: `tests/scripts/test_build_sft_promotion_gate.py`
- Script index update: `scripts/README.md`

Purpose:

- Enforce the plan requirement that SFT must beat base before RL.
- Compare baseline vs SFT rollout audit reports on matching labels, such as:
  - `mix_v2_p5050`
  - `v02_strict`
  - `v03_strict`
- Require detector-mimic report pass.
- If a real Pangram export has been collected, pass its alignment report with
  `--pangram-alignment-report`; a supplied report must pass.
- Require human read JSON with `passed = true` and `20 <= sample_count <= 50`.
- Require SFT output verification report pass before candidate audit results
  are promotable.
- Fail if tracked diagnostics increase:
  - `wrapper_phrase`
  - `option_menu`
  - `emoji`
  - `all_caps`
  - `fake_casual_phrase`
  - `low_specificity_substitution`
- Fail if any candidate high-reward diagnostic counter is nonzero.

Expected command shape after SFT eval artifacts exist:

```bash
uv run scripts/eval/build_sft_human_read_packet.py \
  --checkpoint-id <READY_SFT_CHECKPOINT_ID> \
  --candidate-audit mix_v2_p5050=<sft_mix_audit.json> \
  --candidate-audit v02_strict=<sft_v02_audit.json> \
  --candidate-audit v03_strict=<sft_v03_audit.json> \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/human_read_packet.json
```

```bash
uv run scripts/eval/build_sft_promotion_gate.py \
  --checkpoint-id <READY_SFT_CHECKPOINT_ID> \
  --baseline-audit mix_v2_p5050=<base_mix_audit.json> \
  --baseline-audit v02_strict=<base_v02_audit.json> \
  --baseline-audit v03_strict=<base_v03_audit.json> \
  --candidate-audit mix_v2_p5050=<sft_mix_audit.json> \
  --candidate-audit v02_strict=<sft_v02_audit.json> \
  --candidate-audit v03_strict=<sft_v03_audit.json> \
  --detector-report <detector_mimic_report.json> \
  --pangram-alignment-report <optional_pangram_alignment_report.json> \
  --human-read runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/human_read_packet.json \
  --sft-output-verification runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/sft_output_verification.json \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/promotion_gate.json
```

2026-06-29 update:

- Added inline `uv` script metadata so the promotion gate can be checked with
  `uv run scripts/eval/build_sft_promotion_gate.py --help` without building
  the full project environment.
- Refreshed the live SFT preflight after Prime login:
  - dataset SFT config: pass
  - local `HF_TOKEN`: pass
  - Prime auth: pass
  - `WANDB_API_KEY` source: fail
- Decision: do not launch the tracked Qwen 2B SFT until `WANDB_API_KEY` is
  available locally or as a Prime secret.

Validation:

```text
build_sft_promotion_gate --help: pass
build_sft_promotion_gate focused tests: 7 passed
broad Prime/reward gate suite: 106 passed
ruff check: pass
git diff --check: pass
```

### Prime Qwen 2B SFT Human-Read Packet

Date: 2026-06-29

Added:

- Script: `scripts/eval/build_sft_human_read_packet.py`
- Test: `tests/scripts/test_build_sft_human_read_packet.py`

Purpose:

- Standardize the manual read artifact required by the SFT promotion gate.
- Collect a bounded `20..50` sample packet from SFT candidate audit
  `top_samples`.
- Keep `passed=false` and `review_status=needs_review` until the packet is
  manually reviewed.
- Preserve label, audit path, rank, task/problem/sample ids, reward,
  failed diagnostics, and completion preview for each item.

Command shape:

```bash
uv run scripts/eval/build_sft_human_read_packet.py \
  --checkpoint-id <READY_SFT_CHECKPOINT_ID> \
  --candidate-audit mix_v2_p5050=<sft_mix_audit.json> \
  --candidate-audit v02_strict=<sft_v02_audit.json> \
  --candidate-audit v03_strict=<sft_v03_audit.json> \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/human_read_packet.json
```

Decision:

- If this script reports fewer than `20` samples, rerun the candidate rollout
  audits with a larger `--top-n` before attempting SFT promotion.

### Prime Qwen 2B SFT Output Verification

Date: 2026-06-29

Added:

- Script: `scripts/train/verify_prime_sft_output.py`
- Test: `tests/scripts/test_verify_prime_sft_output.py`

Purpose:

- Verify that an open `prime-rl` dataset SFT run produced a usable filesystem
  artifact before eval or hosted-checkpoint handoff.
- Check the SFT config facts:
  - model: `Qwen/Qwen3.5-2B`
  - renderer: `qwen3.5`
  - dataset: approved S1/S2 Prime SFT corpus
- Check that `output_dir/weights/step_N` exists and contains safetensors files.
- Check adapter artifacts when `ckpt.weights.save_adapter_separately = true`.

Command shape after the SFT output directory is available:

```bash
uv run scripts/train/verify_prime_sft_output.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml \
  --step 200 \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/sft_output_verification.json
```

S2 command:

```bash
uv run scripts/train/verify_prime_sft_output.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml \
  --step 200 \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/sft_output_verification.json
```

Decision:

- Do not treat a completed SFT process as a usable model until this verifier
  passes.

Validation:

```text
verify_prime_sft_output focused tests: 5 passed
broad Prime/reward gate suite: 106 passed
ruff check: pass
git diff --check: pass
```

### Prime Qwen 2B Warm-Start Checkpoint Handoff

Date: 2026-06-29

Added:

- Script: `scripts/train/verify_prime_warm_start_checkpoint.py`
- Test: `tests/scripts/test_verify_prime_warm_start_checkpoint.py`
- Guarded `scripts/train/prepare_prime_sft_to_rl_config.py` so real after-SFT
  rendering requires `--checkpoint-handoff-report`.
- Also require a passing `--promotion-gate-report` from
  `scripts/eval/build_sft_promotion_gate.py` before rendering a real after-SFT
  hosted RL config.
- Added `--allow-unverified-checkpoint` only for template validation.

Purpose:

- Verify the checkpoint id before rendering an after-SFT hosted RL config.
- Require the checkpoint to be present in `prime train checkpoints`, `READY`,
  tied to the expected run id, and from `Qwen/Qwen3.5-2B`.
- Make the handoff explicit because open `prime-rl` dataset SFT writes local
  HF-compatible weights under `output_dir/weights/step_N`, while hosted
  `checkpoint_id` is a Prime Hosted Training warm-start id.

Command shape after a Prime run has a READY checkpoint:

```bash
uv run scripts/train/verify_prime_warm_start_checkpoint.py \
  --run-id <PRIME_RUN_ID_WITH_READY_CHECKPOINT> \
  --checkpoint-id <READY_SFT_CHECKPOINT_ID> \
  --output runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/checkpoint_handoff.json
```

Then render the hosted RL config only with the passing report:

```bash
uv run scripts/train/prepare_prime_sft_to_rl_config.py \
  --checkpoint-id <READY_SFT_CHECKPOINT_ID> \
  --checkpoint-handoff-report runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/checkpoint_handoff.json \
  --promotion-gate-report runs/prime_sft_promotion/<READY_SFT_CHECKPOINT_ID>/promotion_gate.json \
  --output configs/prime/qwen35_2b_p5050_after_sft_env0315_<checkpoint_slug>.toml \
  --run-name humanize-p5050-qwen35-2b-after-sft-env0315-<checkpoint_slug>
```

Decision:

- Do not render or launch the after-SFT hosted RL config from a pasted
  checkpoint id alone.
- Do not render or launch the after-SFT hosted RL config from a checkpoint that
  lacks a passing SFT promotion gate.
- If the SFT run only produces open `prime-rl` filesystem weights and no Prime
  Hosted Training checkpoint id, add a separate upload/import step before
  hosted RL.
- The config renderer now enforces this by default. Use
  `--allow-unverified-checkpoint` only for dummy template validation.

Validation:

```text
verify/prepare focused tests: 11 passed
broad Prime/reward gate suite: 99 passed
ruff check: pass
git diff --check: pass
```

Live preflight remains:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4313 validation=239 test=241
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
```

### Prime Qwen 2B SFT Preflight Recheck

Date: 2026-06-29

Change:

- Added inline `uv` script metadata to lightweight train helpers:
  - `scripts/train/prime_sft_preflight.py`
  - `scripts/train/prepare_prime_sft_launch_kit.py`
  - `scripts/train/prepare_prime_sft_to_rl_config.py`
- Added a regression test so those documented direct `uv run scripts/...`
  commands do not depend on the full project environment.

Reason:

- A direct local preflight first tried to build project dependency
  `fasttext-wheel` and failed against the local macOS/Xcode SDK before the
  preflight code ran.
- The scripts only need Click, so inline script metadata is the correct smaller
  runtime surface.

Live command:

```bash
uv run scripts/train/prime_sft_preflight.py --check-hf-viewer
```

Result:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4313 validation=239 test=241
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
Error: Prime SFT preflight failed
```

Prime secret store:

```json
{"secrets": []}
```

Decision:

- Prime login and HF data are ready.
- Do not launch the tracked Qwen 2B dataset SFT until `WANDB_API_KEY` is
  available locally or in Prime secrets.

### Prime Env0315 Cross-Run Ablation Matrix

Date: 2026-06-29

Added:

- Script: `scripts/eval/build_prime_ablation_matrix.py`
- Test: `tests/scripts/test_build_prime_ablation_matrix.py`
- Report: `runs/prime_training_smoke/ablation_matrix_env0315.json`

Purpose:

- Summarize saved Prime hosted RL ablations from one machine-readable artifact.
- Keep model-selection evidence separate from narrative reads in the log.
- Fail selection if a run is not `COMPLETED`, if the rollout bundle or
  detector-mimic gate fails, if p50 delta is negative, or if either strict eval
  delta is negative.

Refresh before matrix:

- Rebuilt the saved bundle reports for:
  - `fj9oinokvx5zott096tgfwqw`
  - `r8xlz0csp79fu0z0elp9h1dv`
  - `pzcfi8aew2pnxnkhec2augts`
  - `o48ryskshkn06b3o1b1kauql`
- The refreshed bundles now embed the expanded `22` row detector-mimic gate.

Command:

```bash
uv run scripts/eval/build_prime_ablation_matrix.py \
  --output runs/prime_training_smoke/ablation_matrix_env0315.json
```

Result:

```text
runs=4 selection_pass=1 best=pzcfi8aew2pnxnkhec2augts report=runs/prime_training_smoke/ablation_matrix_env0315.json
```

Matrix read:

- Best current passing smoke:
  `pzcfi8aew2pnxnkhec2augts`
  (`Qwen/Qwen3.5-0.8B`, lr `5e-5`, score `0.259782`).
- Rejected:
  - `fj9oinokvx5zott096tgfwqw`: p50 delta `-0.021137`.
  - `r8xlz0csp79fu0z0elp9h1dv`: strict deltas `-0.064143` and `-0.051501`.
  - `o48ryskshkn06b3o1b1kauql`: status `STOPPED`, strict deltas `-0.168230`
    and `-0.128226`.

Decision:

- The cross-run matrix agrees with the prior manual read: do not continue Qwen
  2B base RL at `8e-5`.
- The next full-model path remains Qwen 2B dataset SFT, then SFT-to-RL from a
  promoted SFT checkpoint.

Validation:

```text
build_prime_ablation_matrix --help: pass
build_prime_ablation_matrix focused tests: 3 passed
broad Prime/reward gate suite: 109 passed
ruff check: pass
git diff --check: pass
```

Live SFT preflight recheck:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4313 validation=239 test=241
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
Error: Prime SFT preflight failed
```

### Prime Qwen 2B SFT Preflight JSON Report

Date: 2026-06-29

Change:

- Added `--output` and `--no-fail-on-gate` to
  `scripts/train/prime_sft_preflight.py`.
- Added a regression test proving a failed launch gate can still be persisted
  as JSON.

Command:

```bash
uv run scripts/train/prime_sft_preflight.py \
  --config configs/prime_rl/qwen35_2b_sft_target_messages_env0314_gate_env0315.toml \
  --check-hf-viewer \
  --output runs/prime_sft_preflight/qwen35_2b_env0315.json \
  --no-fail-on-gate
```

Report:

```text
runs/prime_sft_preflight/qwen35_2b_env0315.json
```

Report summary:

```text
gate.passed=false
failed_checks=["WANDB_API_KEY source"]
```

Read:

- Dataset config, local `HF_TOKEN`, HF Dataset Viewer counts, and Prime auth all
  pass.
- W&B remains the only failed launch check.
- Do not launch tracked Qwen 2B dataset SFT until this report flips to
  `gate.passed=true`.

Validation:

```text
prime_sft_preflight --help: pass
prime_sft_preflight focused tests: 6 passed
broad Prime/reward gate suite: 110 passed
ruff check: pass
git diff --check: pass
```

### Prime Qwen 2B SFT Eval Manifest

Date: 2026-06-29

Change:

- Added `scripts/eval/build_sft_eval_manifest.py`.
- Added `tests/scripts/test_build_sft_eval_manifest.py`.
- Generated template manifest:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315/sft_eval_manifest.json`.

Purpose:

- Keep the post-SFT eval and promotion handoff as one deterministic artifact.
- Make SFT-to-RL launch depend on explicit checks instead of manual placeholder
  edits.

Command:

```bash
uv run scripts/eval/build_sft_eval_manifest.py \
  --output runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315/sft_eval_manifest.json
```

Manifest gate order:

1. `verify_sft_output`
2. `collect_base_and_sft_rollout_audits`
3. `build_human_read_packet`
4. `build_promotion_gate`
5. `verify_warm_start_checkpoint`
6. `render_after_sft_rl_config`

Live SFT preflight after Prime login:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0314
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4313 validation=239 test=241
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
```

Decision:

- Prime auth and HF data are ready.
- Do not launch tracked Qwen 2B dataset SFT until `WANDB_API_KEY` is available
  locally or in Prime secrets.
- Do not render or launch after-SFT RL until SFT output verification, human-read
  approval, promotion gate, and checkpoint handoff all pass.

Validation:

```text
build_sft_eval_manifest --help: pass
build_sft_eval_manifest focused tests: 2 passed
build_sft_eval_manifest ruff check: pass
broad Prime/reward gate suite: 112 passed
```

### Prime Env0315 Repair SFT Candidate

Date: 2026-06-29

Change:

- Added `scripts/data/build/extract_prime_audit_failures.py`.
- Added `tests/scripts/test_extract_prime_audit_failures.py`.
- Fixed `src/humanize_rl/training/gemma4_sft_dataset.py` so nested generated
  reference metadata contributes source/domain/mode during dataset builds.

Artifacts:

- Failure set:
  `data/processed/sft/prime_audit_failure_set_env0315.jsonl`
- Generated candidates:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_candidate.jsonl`
- Clean repair references:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50.jsonl`
- Clean report:
  `data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_clean50_report.json`
- Local merged SFT input:
  `data/processed/v04_sft_final_plus_prime_env0315_clean50.jsonl`
- Builder output:
  `data/processed/sft/gemma4_e2b_v04_prime_env0315_clean50/`

Commands:

```bash
uv run scripts/data/build/extract_prime_audit_failures.py \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step140_env0315_candidate.json \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step150_env0315_candidate.json \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step160_env0315_candidate.json \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step170_env0315_candidate.json \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step180_env0315_candidate.json \
  --audit runs/prime_training_smoke/zztqgqclh3y3hslpjsofzpcf/audit_step190_env0315_candidate.json \
  --audit runs/prime_training_smoke/o48ryskshkn06b3o1b1kauql/audit_step10_env0315.json \
  --audit runs/prime_training_smoke/o48ryskshkn06b3o1b1kauql/audit_step30_env0315.json \
  --audit runs/prime_training_smoke/o48ryskshkn06b3o1b1kauql/audit_step40_env0315.json \
  --audit runs/prime_training_smoke/o48ryskshkn06b3o1b1kauql/audit_step50_env0315.json \
  --output data/processed/sft/prime_audit_failure_set_env0315.jsonl
```

```bash
PYTHONPATH=src uv run --no-project --with click python \
  scripts/data/build/generate_sft_references_from_failures.py \
  --failure-path data/processed/sft/prime_audit_failure_set_env0315.jsonl \
  --limit 70 \
  --resume \
  --output-path data/processed/sft/reference_targets/prime_audit_failure_refs_env0315_candidate.jsonl
```

Result:

- Extracted failure rows: `100`
- Generated candidate rows: `70`
- Clean rows kept: `50`
- Unique task IDs in clean rows: `36`
- Clean family counts:
  - compression: `10`
  - direct_email: `9`
  - rewrite_repair: `19`
  - tone_shift: `12`
- Clean mode counts:
  - compression: `4`
  - expansion: `1`
  - long_form_generate: `7`
  - multi_constraint_compose: `2`
  - rewrite: `20`
  - rewrite_humanize: `9`
  - tone_shift: `7`

Builder result:

```text
raw_rows=4905
accepted_rows=4843
rejected_rows=62
split_counts={'train': 4358, 'valid': 242, 'test': 243}
prime_failure_reference_generation accepted rows=70
```

Decision:

- This is the S2 repair-data candidate, not yet the active Prime SFT dataset.
- Do not point Prime SFT at it until it is published to a dedicated HF dataset
  and the Prime SFT preflight is updated to that dataset.
- The existing S1 Qwen 2B config remains blocked only on W&B:
  `WANDB_API_KEY source`.

Validation:

```text
extract_prime_audit_failures focused tests: 3 passed
generate_sft_references_from_failures focused tests: included
gemma4_sft_dataset focused tests: included
expanded Prime/reward/SFT data suite: 131 passed
ruff check: pass
git diff --check: pass
```

### Prime Env0315 Clean50 HF Dataset And S2 Config

Date: 2026-06-29

Published dataset:

```text
jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50
```

HF commit:

```text
e1e6b1e839c0dc0450313e5f558c90a6ac925557
```

Added:

- Hub package helper:
  `scripts/data/build/prepare_prime_sft_messages_dataset.py`
- Test:
  `tests/scripts/test_prepare_prime_sft_messages_dataset.py`
- Hub-ready local folder:
  `runs/hf_datasets/humanize-rl-prime-sft-messages-env0315-clean50/`
- S2 Prime SFT config:
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml`
- S2 preflight report:
  `runs/prime_sft_preflight/qwen35_2b_env0315_clean50.json`
- S2 sandbox launch kit:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50/`
  and `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz`
- S2 post-SFT eval manifest:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json`

Hub package manifest:

```text
repo_id=jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50
total_rows=4843
train=4358
validation=242
test=243
repair_reference_rows=70
```

Live S2 preflight:

```text
dataset SFT config: pass - Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50
local HF_TOKEN: pass - set
HF Dataset Viewer: pass - train=4358 validation=242 test=243
Prime auth: pass - whoami succeeded
WANDB_API_KEY source: fail - set WANDB_API_KEY or create a Prime secret before tracked SFT
```

S2 eval manifest handoff:

```text
after_sft_config=configs/prime/qwen35_2b_p5050_after_sft_env0315_clean50_<checkpoint_slug>.toml
rl_run_name=humanize-p5050-qwen35-2b-after-sft-env0315-clean50-<checkpoint_slug>
```

Decision:

- S2 clean50 is now the quality-oriented Qwen 2B dataset-SFT candidate.
- Use S1 only if intentionally running the no-new-repairs baseline ablation.
- Do not launch either tracked SFT path until `WANDB_API_KEY` is available
  locally or in Prime secrets.

Validation:

```text
prepare_prime_sft_messages_dataset focused tests: 2 passed
prime_sft_preflight / launch kit / config focused tests: 14 passed
expanded Prime/reward/SFT data suite: 146 passed
ruff check: pass
git diff --check: pass
```

### Prime S2 Eval Manifest Variant Separation

Date: 2026-06-29

Change:

- Extended `scripts/eval/build_sft_eval_manifest.py` with:
  - `--after-sft-config`
  - `--rl-run-name`
- Regenerated the S2 clean50 manifest at:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json`.

Reason:

- S1 and S2 after-SFT RL configs should not collide once a READY checkpoint
  exists.
- The future hosted RL run name should preserve the S2 data provenance.

Result:

```text
after_sft_config=configs/prime/qwen35_2b_p5050_after_sft_env0315_clean50_<checkpoint_slug>.toml
rl_run_name=humanize-p5050-qwen35-2b-after-sft-env0315-clean50-<checkpoint_slug>
```

Decision:

- Keep S2 as the next quality-oriented SFT candidate.
- Do not render or launch the after-SFT RL config until SFT output verification,
  base-vs-SFT audits, detector mimic, human read, promotion gate, and Prime
  checkpoint handoff all pass.

Validation:

```text
build_sft_eval_manifest focused tests: 2 passed
expanded Prime/reward/SFT data suite: 146 passed
ruff check: pass
git diff --check: pass
```

### Pangram Export Alignment Adapter

Date: 2026-06-29

Added:

- Module: `src/humanize_rl/scoring/pangram_alignment.py`
- Script: `scripts/eval/compare_detector_mimic_to_pangram.py`
- Tests:
  - `tests/scoring/test_pangram_alignment.py`
  - `tests/scripts/test_compare_detector_mimic_to_pangram.py`

Purpose:

- Compare a saved Pangram JSON/JSONL export against the frozen
  `data/eval/detector_mimic_v01.jsonl` rows.
- Match rows by id first, then exact text.
- Report coverage, human/nonhuman label agreement, and AI-fraction deltas.
- Keep Pangram calibration offline so Prime reward scoring stays deterministic.

Command:

```bash
uv run scripts/eval/compare_detector_mimic_to_pangram.py \
  --input data/eval/detector_mimic_v01.jsonl \
  --pangram-output runs/detector_mimic/pangram_export.json \
  --output runs/detector_mimic/pangram_alignment_report.json
```

Decision:

- Do not require Pangram for training or hosted reward execution.
- Use this only as an optional promotion-read artifact if a real export is
  collected for the frozen mimic rows.

### Optional Pangram Alignment Promotion Gate

Date: 2026-06-29

Change:

- Extended `scripts/eval/build_sft_promotion_gate.py` with optional
  `--pangram-alignment-report`.
- Extended `scripts/eval/build_sft_eval_manifest.py` so manifests record the
  optional Pangram alignment report path.
- Regenerated the S2 clean50 manifest:
  `runs/prime_sft_promotion/TEMPLATE_QWEN35_2B_ENV0315_CLEAN50/sft_eval_manifest.json`.

Decision:

- Pangram remains optional and offline.
- If a real Pangram export is collected and supplied to the SFT promotion gate,
  the alignment report must pass before SFT-to-RL promotion.
- The stored S2 manifest marks
  `runs/detector_mimic/pangram_alignment_report.json` as `required_now=false`;
  missing Pangram output does not block the current S2 launch path.

### Prime S2 Output Verifier Dataset Guard

Date: 2026-06-29

Change:

- Updated `scripts/train/verify_prime_sft_output.py` so output verification
  accepts both approved SFT corpora:
  - `jayshah5696/humanize-rl-prime-sft-messages-env0314`
  - `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50`
- Added regression coverage proving clean50 passes the dataset guard and an
  untracked dataset still fails.

Reason:

- The S2 clean50 eval manifest uses
  `configs/prime_rl/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315.toml`.
  The old verifier hardcoded env0314, so a valid S2 output would fail before
  rollout audits.

Decision:

- Keep the guard explicit. Only S1 and S2 Prime SFT datasets are approved for
  this Qwen 2B verifier right now.

Validation:

```text
compare_detector_mimic_to_pangram --help: pass
build_sft_promotion_gate / build_sft_eval_manifest focused tests: 11 passed
verify_prime_sft_output focused tests: 5 passed
detector/Pangram focused tests: 13 passed
expanded Prime/reward/SFT data suite: 146 passed
ruff check: pass
git diff --check: pass
```

### SFT Eval Rollout Audit Command Plan

Date: 2026-06-29

Change:

- Added `--eval-label` to `scripts/eval/audit_prime_rollouts.py`.
- Labels:
  - `mix_v2_p5050` uses
    `data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl` with
    `p50_50_no_penalty`.
  - `v02_strict` uses
    `environments/humanize_rl_env/humanize_rl_env/humanize_tasks_v02_smoke.jsonl`
    with `strict`.
  - `v03_strict` uses `data/rl/humanize_tasks_v03_filtered.jsonl` with
    `strict`.
- Updated the S2 clean50 eval manifest so it records `baseline_rollouts`,
  `candidate_rollouts`, `audit_specs`, and six concrete audit commands for base
  and SFT across those three labels.
- Audit commands pin `scikit-learn>=1.8,<1.9` to avoid ridge pickle version
  drift during local rescoring.

Decision:

- This only plans auditing of saved rollout JSONs. It does not replace the need
  to collect actual base/SFT rollouts after SFT finishes.
- Prime login is healthy, but the tracked S2 SFT launch still waits on
  `WANDB_API_KEY` locally or in Prime secrets.

Validation:

```text
audit_prime_rollouts/build_sft_eval_manifest focused tests: 5 passed
audit_prime_rollouts --help with pinned deps: pass
expanded Prime/reward/SFT data suite: 149 passed
S2 clean50 preflight: Prime auth pass, WANDB_API_KEY source fail
ruff check: pass
git diff --check: pass
```

### Prime S2 Launch Kit Carries Eval Manifest

Date: 2026-06-29

Change:

- Added optional `--eval-manifest` to
  `scripts/train/prepare_prime_sft_launch_kit.py`.
- When supplied, the launch kit copies the SFT eval manifest to
  `prime_sft_launch_kit/sft_eval_manifest.json`, includes it in the tarball,
  and records its hash, checkpoint placeholder, promotion root, and gate order
  in `manifest.json`.
- Regenerated
  `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz` with the S2
  eval manifest embedded.

Decision:

- Keep the S2 run artifact tied to its post-SFT verification and SFT-to-RL
  handoff checklist. This reduces the chance of launching S2 but later using an
  S1 or stale promotion manifest.
- This still does not launch SFT. W&B remains the launch gate.

Validation:

```text
prepare_prime_sft_launch_kit focused tests: 5 passed
S2 kit tar includes prime_sft_launch_kit/sft_eval_manifest.json
S2 kit manifest records promotion root and gate order
expanded Prime/reward/SFT data suite: 150 passed
S2 clean50 preflight: Prime auth pass, WANDB_API_KEY source fail
ruff check: pass
git diff --check: pass
```

### SFT Eval Manifest Checkpoint Slug Materialization

Date: 2026-06-29

Change:

- Updated `scripts/eval/build_sft_eval_manifest.py` so a real
  `--checkpoint-id` replaces `<checkpoint_slug>` in the future after-SFT RL
  config path and W&B run name.
- The template checkpoint id `READY_SFT_CHECKPOINT_ID` keeps placeholders intact
  for reusable manifest templates.
- Added `checkpoint_slug` to the SFT eval manifest and to the launch-kit summary
  when an eval manifest is packaged.
- Regenerated the S2 clean50 eval manifest and the S2 launch kit tarball.

Decision:

- Use placeholder-preserving manifests for pre-launch templates.
- After SFT completes, rebuild the manifest with the real checkpoint id so the
  SFT-to-RL config path and run name are concrete and tied to that checkpoint.

Validation:

```text
build_sft_eval_manifest focused tests: 2 passed
materialization smoke: ckpt_ready_123 -> ckpt-ready-123
template manifest keeps <checkpoint_slug>: pass
focused manifest/launch-kit/SFT-to-RL tests: 18 passed
expanded Prime/reward/SFT data suite: 150 passed
S2 clean50 preflight: Prime auth pass, WANDB_API_KEY source fail
ruff check: pass
git diff --check: pass
```

### SFT Promotion Placeholder Gate Hardening

Date: 2026-06-29

Change:

- Updated `scripts/eval/build_sft_human_read_packet.py` and
  `scripts/eval/build_sft_promotion_gate.py` to reject both placeholder
  checkpoint strings:
  - `READY_SFT_CHECKPOINT_ID`
  - `FILL_WITH_READY_SFT_CHECKPOINT_ID`
- Added regressions proving the S2 template checkpoint cannot produce a
  human-read packet or pass the SFT promotion gate.

Decision:

- Template manifests are for handoff only. Any human-read packet or promotion
  gate must use a real READY checkpoint id from the completed SFT run.

Validation:

```text
human-read/promotion focused tests: 14 passed
expanded Prime/reward/SFT data suite: 152 passed
S2 clean50 preflight: Prime auth pass, WANDB_API_KEY source fail
ruff check: pass
git diff --check: pass
```

### Prime S2 Launch Readiness Verifier

Date: 2026-06-30

Change:

- Added `scripts/train/verify_prime_sft_launch_readiness.py`.
- Added regression tests covering:
  - matching config/preflight/launch-kit/eval-manifest artifacts pass;
  - W&B-only preflight failure blocks launch readiness;
  - launch-kit config hash drift fails.
- Wrote current S2 readiness report:
  `runs/prime_sft_preflight/qwen35_2b_env0315_clean50_launch_readiness.json`.

Decision:

- Before launching S2, run the readiness verifier after refreshing preflight and
  regenerating the launch kit. The verifier must pass, not only the standalone
  preflight.
- Current S2 artifact consistency is good: config hash and packaged eval
  manifest hash match. Launch readiness still fails only because
  `WANDB_API_KEY source` fails.

Validation:

```text
verify_prime_sft_launch_readiness focused tests: 3 passed
current S2 launch readiness: fail, failures=["preflight gate failed: WANDB_API_KEY source"]
focused preflight/launch-kit/readiness tests: 16 passed
expanded Prime/reward/SFT data suite: 155 passed
ruff check: pass
git diff --check: pass
```

### Prime S2 Launch Archive Readiness Check

Date: 2026-06-30

Change:

- Extended `scripts/train/verify_prime_sft_launch_readiness.py` with
  `--launch-archive`.
- The readiness report now verifies the uploaded tarball has:
  - `prime_sft_launch_kit/config.toml`
  - `prime_sft_launch_kit/manifest.json`
  - `prime_sft_launch_kit/sft_eval_manifest.json`
  - `prime_sft_launch_kit/run_sft.sh`
  - `prime_sft_launch_kit/README.md`
- It also compares the archived config, manifest, and eval manifest against the
  current source artifacts.
- Regenerated the S2 launch kit archive:
  `runs/prime_sft_launch_kit/qwen35_2b_env0315_clean50.tar.gz`.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
launch_archive_sha256=efef87de57617b11e01690aa2888b3f135a777c5fdf389d9b21a76c8ca2706da
```

Decision:

- Treat `qwen35_2b_env0315_clean50_launch_readiness.json` as the final
  pre-spend gate. It must pass after W&B is present, proving both the local kit
  directory and upload tarball match the S2 config and eval manifest.

Validation:

```text
verify_prime_sft_launch_readiness focused tests: 4 passed
current S2 launch readiness: fail only on WANDB_API_KEY source
focused preflight/launch-kit/readiness tests: 17 passed
expanded Prime/reward/SFT data suite: 156 passed
ruff check: pass
git diff --check: pass
```

### Prime S2 W&B Source Recheck

Date: 2026-06-30

Result:

- `/private/tmp/humanize_rl_prime_wandb.env`: missing.
- Prime secret list: empty.
- Current S2 launch readiness therefore remains blocked only on
  `WANDB_API_KEY source`.

Housekeeping:

- Added `scripts/train/verify_prime_sft_launch_readiness.py` to
  `scripts/README.md`.
- Added the readiness script to the lightweight train-script inline `uv`
  metadata regression.

Validation:

```text
focused preflight/launch-kit/readiness tests: 17 passed
expanded Prime/reward/SFT data suite: 156 passed
ruff check: pass
git diff --check: pass
```

### Prime S2 Launch Runner/README Archive Match

Date: 2026-06-30

Change:

- Extended `scripts/train/verify_prime_sft_launch_readiness.py` to compare the
  archived `run_sft.sh` and `README.md` against the local S2 launch kit.
- Added regressions for stale archived runner/readme files.
- Refreshed
  `runs/prime_sft_preflight/qwen35_2b_env0315_clean50_launch_readiness.json`.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
launch_archive_sha256=efef87de57617b11e01690aa2888b3f135a777c5fdf389d9b21a76c8ca2706da
runner_sha256=703f933925db9b3ca00019167c839d339e7b4fef623aa9e3405a2362f4ec661f
readme_sha256=15a96e2cf8ad5dbc44113a9a6879c0beb96669571d3b0e841270b0584fb7005d
```

Decision:

- Prime auth is now passing in preflight.
- Do not launch S2 until `WANDB_API_KEY` is present locally or as a Prime
  secret. Current `prime --plain secret list --output json` returns an empty
  secret list.

Validation:

```text
focused preflight/launch-kit/readiness tests: 19 passed
expanded Prime/reward/SFT data suite: 158 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 Eval Manifest Pins After-SFT Template

Date: 2026-07-01

Change:

- Updated `scripts/eval/build_sft_eval_manifest.py` so the post-SFT handoff
  manifest records the exact after-SFT hosted RL template and includes
  `--template` in the render command.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
sft_eval_manifest_sha256=92df8ad93946c55cfc631737f34dfb14efbc8b404a39ac7423046610f5e65b36
launch_archive_sha256=3c96fdc3bb1d3dd1cf2c33d990f8086351d3df4f52b52a3022da95731c4bdcc4
after_sft_template=configs/prime/qwen35_2b_p5050_after_sft_env0315_full200_template.toml
```

Decision:

- The S2 clean50 checkpoint will still render from the shared env0315 Qwen 2B
  after-SFT RL template. The clean50 branch is distinguished by checkpoint id,
  promotion root, rendered config path, and run name.
- Keep launch blocked until W&B is available. Prime auth, HF token, HF Dataset
  Viewer counts, and artifact consistency pass.

Validation:

```text
focused SFT handoff/launch tests: 24 passed
expanded Prime/reward/SFT data suite: 158 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 Renderer Requires Eval Manifest

Date: 2026-07-01

Change:

- Updated `scripts/eval/build_sft_eval_manifest.py` to record SHA256 for the
  pinned after-SFT hosted RL template.
- Updated `scripts/train/prepare_prime_sft_to_rl_config.py` so real after-SFT
  config rendering requires `--sft-eval-manifest` and verifies:
  - manifest checkpoint id matches;
  - manifest after-SFT config path matches `--output`;
  - manifest checkpoint handoff and promotion-gate paths match;
  - manifest template path and SHA256 match the current template file.
- Updated `scripts/train/verify_prime_sft_launch_readiness.py` so the pre-spend
  S2 readiness report also checks the pinned after-SFT template hash.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
sft_eval_manifest_sha256=12e163cd317a4dfc4151b950dc9ac9fd405cd4cc24f7a8361f4ba5a9e1875440
launch_archive_sha256=f9ad23eff8f76dd01d080673a0e0d4f7bd0be6097e47f0b2f244e1c3022fb3e5
after_sft_template_sha256=bb283712c27dff808e7c81bd65ebe042874d492bf4367f927512bad3e38815ca
```

Decision:

- Keep S2 blocked until W&B is available.
- Once SFT finishes, rebuild the eval manifest with the real checkpoint id and
  use that same manifest when rendering the after-SFT RL config.

Validation:

```text
focused SFT handoff/launch tests: 27 passed
expanded Prime/reward/SFT data suite: 161 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 SFT Output Verification Uses Eval Manifest

Date: 2026-07-01

Change:

- Updated the S2 eval manifest so the `verify_sft_output` command passes
  `--sft-eval-manifest`.
- Updated `scripts/train/verify_prime_sft_output.py` so, when supplied, the
  eval manifest must match:
  - the SFT config path;
  - the expected `sft_output_verification.json` output path.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
sft_eval_manifest_sha256=126a0cf9baac6973fd117621c23827ff4eac6ce097bc1cfae867073653420d39
launch_archive_sha256=d0f0fcb53fca44c5bb4c048795f736182c872f6951eeb5f3252976b64e7ca95e
after_sft_template_sha256=bb283712c27dff808e7c81bd65ebe042874d492bf4367f927512bad3e38815ca
```

Decision:

- Use the eval manifest as the post-SFT handoff contract for both output
  verification and after-SFT RL config rendering.
- S2 remains blocked only on W&B.

Validation:

```text
focused SFT handoff tests: 34 passed
expanded Prime/reward/SFT data suite: 163 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 Promotion Gate Uses Eval Manifest

Date: 2026-07-01

Change:

- Updated the S2 eval manifest so the `build_promotion_gate` command passes
  `--sft-eval-manifest`.
- Updated `scripts/eval/build_sft_promotion_gate.py` so, when supplied, the
  eval manifest must match:
  - checkpoint id;
  - base and SFT audit paths;
  - detector report path;
  - optional Pangram alignment path when supplied or required;
  - human-read packet path;
  - SFT output verification path;
  - promotion gate output path.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
sft_eval_manifest_sha256=18846849cfc391f86b5660fc36a4973f3ade0f6fa0a5b7aa5e8657b95bf4c11e
launch_archive_sha256=af45f94b4953f7a5a533b5df37a8101b39a25f80c42ae0dd8111bfdf56e4ac82
after_sft_template_sha256=bb283712c27dff808e7c81bd65ebe042874d492bf4367f927512bad3e38815ca
```

Decision:

- Use the eval manifest as the single post-SFT handoff contract for output
  verification, promotion, and after-SFT RL config rendering.
- S2 remains blocked only on W&B.

Validation:

```text
focused SFT handoff tests: 46 passed
expanded Prime/reward/SFT data suite: 165 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 Human-Read Packet Uses Eval Manifest

Date: 2026-07-01

Change:

- Updated the S2 eval manifest so the `build_human_read_packet` command passes
  `--sft-eval-manifest`.
- Updated `scripts/eval/build_sft_human_read_packet.py` so, when supplied, the
  eval manifest must match:
  - checkpoint id;
  - candidate audit paths;
  - human-read packet output path.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.
- Updated `configs/prime/README.md` so the manual human-read packet command
  uses the eval manifest too.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
sft_eval_manifest_sha256=1babc7f14d5ccf50d22884e3b47fc4a5684b7052e4f37e38a96d18ef23db84a0
launch_manifest_sha256=8e24daaaeb848703c87db98523b1b4a5ed4b67ea2c3f7f0018928189b5302fc7
launch_archive_sha256=4ef4b0d55ed4357d4e70de23b667dc903d28ca5d77de5e5949e875c102490161
preflight_report_sha256=91eb45dbc8422085ecfe2d1b21588048e51361cf0dd357cf08116d2b7d3870d9
launch_readiness_sha256=ec7c3649ac3679082b898bd4fd95f0359460ea8a105f5537571c16573c3fe8e7
```

Decision:

- Use the eval manifest as the single post-SFT handoff contract for output
  verification, human read, promotion, and after-SFT RL config rendering.
- S2 remains gated only on W&B.

Validation:

```text
focused SFT handoff tests: 52 passed
expanded Prime/reward/SFT data suite: 167 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Prime S2 Live Model Availability Preflight

Date: 2026-07-01

Change:

- Added a live Prime hosted-training model availability check to
  `scripts/train/prime_sft_preflight.py`.
- The check reads the configured model name and verifies it appears in
  `prime train models --output json` with `at_capacity=false`.
- Refreshed the S2 clean50 preflight and launch-readiness reports.

Current readiness:

```text
Prime train model availability: pass - Qwen/Qwen3.5-2B available; training_price_per_mtok=0.15
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=31c3d195807ec3bd7069abf18dbbcbde5896249599be2827440ba6f491337e29
launch_readiness_sha256=ec7c3649ac3679082b898bd4fd95f0359460ea8a105f5537571c16573c3fe8e7
sft_eval_manifest_sha256=1babc7f14d5ccf50d22884e3b47fc4a5684b7052e4f37e38a96d18ef23db84a0
launch_archive_sha256=4ef4b0d55ed4357d4e70de23b667dc903d28ca5d77de5e5949e875c102490161
```

Decision:

- Keep S2 as the next quality-oriented Qwen 2B dataset-SFT candidate.
- Do not launch until W&B is present, but no longer treat hosted model
  availability as assumed. It is now checked live before spend.

Validation:

```text
focused SFT launch/handoff tests: 63 passed
expanded Prime/reward/SFT data suite: 170 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Pangram Alignment Counts AI-Assisted Risk

Date: 2026-07-01

Change:

- Updated `src/humanize_rl/scoring/pangram_alignment.py` so Pangram alignment
  records `fraction_ai_assisted` and computes
  `fraction_nonhuman = fraction_ai + fraction_ai_assisted`, capped at `1.0`.
- The alignment delta now compares the local mimic risk against Pangram's
  nonhuman fraction, not only `fraction_ai`.
- Added a regression where Pangram returns `prediction_short="AI-Assisted"` and
  low `fraction_ai`; this now stays visible as detector risk.

Decision:

- Keep Pangram offline and optional for training, but count AI-assisted output
  as nonhuman risk when a real Pangram export is used for promotion review.
- Do not launch S2; W&B is still the only launch blocker.

Validation:

```text
pangram/detector focused tests: 14 passed
expanded Prime/reward/SFT data suite: 171 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Pangram Bulk Items Handoff

Date: 2026-07-01

Change:

- Added `scripts/eval/export_detector_mimic_for_pangram.py`.
- The script writes the frozen detector-mimic rows as SDK-ready
  `items=[{"id": ..., "text": ...}]` for `Pangram.submit_bulk(items=...)`.
- Added inline script metadata/path bootstrapping so these commands run with
  plain `uv run scripts/...`:
  - `scripts/eval/evaluate_detector_mimic.py`
  - `scripts/eval/export_detector_mimic_for_pangram.py`
  - `scripts/eval/compare_detector_mimic_to_pangram.py`
- Generated `runs/detector_mimic/pangram_bulk_items.json`.

Current detector/Pangram artifacts:

```text
detector_mimic_rows=22
detector_gate=pass
false_positive=0
false_negative=0
pangram_bulk_items_sha256=70ce023f0884848059ca167134b1e4188daa62620b26168371cda5411a878951
detector_mimic_report_sha256=129467b5d930205d1342a71f855640baddd2a1abe7c7d61a033231f2d58375c6
detector_mimic_scored_sha256=54538b124c0183ffc6aaf64e616262e2dba87b63bc3b76e3607ca49a985ffd5e
detector_mimic_input_sha256=b19909edb513e38080152bb96c9169e588297501e7d7b4ead720101c1991c7c8
```

Decision:

- Use `pangram_bulk_items.json` as the exact external-detector handoff. Pangram
  results must keep these row IDs so `compare_detector_mimic_to_pangram.py`
  can verify coverage and alignment before promotion.
- Keep Pangram offline and outside Prime reward scoring.
- Do not launch S2; W&B is still the only launch blocker.

Validation:

```text
detector/Pangram focused tests: 16 passed
expanded Prime/reward/SFT data suite: 173 passed
changed Python files ruff check: pass
git diff --check: pass
export/compare/detector plain uv --help: pass
export_detector_mimic_for_pangram generated 22 items
detector-mimic plain uv run: pass
```

### S2 Eval Manifest Pins Pangram Bulk Handoff

Date: 2026-07-01

Change:

- Added `--pangram-bulk-items` to `scripts/eval/build_sft_eval_manifest.py`.
- The S2 eval manifest now records:
  - `pangram_bulk_items.path`;
  - `pangram_bulk_items.exists`;
  - `pangram_bulk_items.required_now=false`;
  - `pangram_bulk_items.sha256` when the file exists.
- Added `external_detector_handoff` commands for:
  - exporting `runs/detector_mimic/pangram_bulk_items.json`;
  - comparing a saved `runs/detector_mimic/pangram_export.json`.
- Updated `scripts/train/verify_prime_sft_launch_readiness.py` to verify the
  recorded Pangram bulk-items hash when the optional file exists.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
pangram_bulk_items_sha256=70ce023f0884848059ca167134b1e4188daa62620b26168371cda5411a878951
sft_eval_manifest_sha256=4136e6f33904fa4e597d8eb03e1a4eaf25da58b0b6d50c735fe51eb736b5b318
launch_manifest_sha256=82e4ae95c7f12171ec83e8a38d5e8353a184f4427b1395f5ec2cec56c6cd1404
launch_archive_sha256=ec76f850c50f6b358802a10caf5a2de355f2b34cf4a32ad78983fc026f6b5c8b
preflight_report_sha256=31c3d195807ec3bd7069abf18dbbcbde5896249599be2827440ba6f491337e29
launch_readiness_sha256=aae92d2221d1d6939e283e102ed0ffb0bd82b43386d914b336c39a46ba348cfe
```

Decision:

- Treat `pangram_bulk_items.json` as part of the S2 post-SFT handoff contract,
  but not as a mandatory training input.
- Keep the Pangram API outside Prime reward scoring.
- Do not launch S2; W&B is still the only launch blocker.

Validation:

```text
focused manifest/readiness/Pangram tests: 26 passed
expanded Prime/reward/SFT data suite: 174 passed
changed Python files ruff check: pass
git diff --check: pass
```

### Pangram Bulk Detection Runner

Date: 2026-07-01

Change:

- Added `scripts/eval/run_pangram_bulk_detection.py`.
- The runner reads `runs/detector_mimic/pangram_bulk_items.json`, calls the
  Pangram SDK bulk flow, writes submission metadata to
  `runs/detector_mimic/pangram_bulk_submit.json`, and writes raw results to
  `runs/detector_mimic/pangram_export.json`.
- The script has `--dry-run` so the frozen 22-row payload can be validated
  without `PANGRAM_API_KEY` or an external API call.
- Updated the S2 eval manifest `external_detector_handoff` to include:
  - `pangram_export_command`;
  - `pangram_bulk_run_command`;
  - `pangram_alignment_command`.
- Regenerated the S2 clean50 eval manifest, launch kit, tarball, preflight
  report, and launch-readiness report.

Current readiness:

```text
PANGRAM_API_KEY=missing
WANDB_API_KEY=missing
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
pangram_bulk_items_sha256=70ce023f0884848059ca167134b1e4188daa62620b26168371cda5411a878951
pangram_bulk_dry_run_sha256=55d1840eeb7f6bb31752822092cda2831c64b1ba0de2036d6cd0ed6674ad70a3
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
launch_manifest_sha256=e1e85ebc38440d4d367760dec6562db9603b3ab2909e33b58c19218fddccc242
launch_archive_sha256=8368ad18a832b22280347de33bf9fdc5214d1ccea7f9199e90c4243cbaa81c9a
preflight_report_sha256=31c3d195807ec3bd7069abf18dbbcbde5896249599be2827440ba6f491337e29
launch_readiness_sha256=3517e3df1bd626928b8b6284bce793840e408d57cfeaafa9f0e48472e2c97ded
```

Decision:

- The Pangram rail now has a complete offline-to-online path:
  export frozen rows, submit/wait/fetch through Pangram bulk detection, then
  compare the result against the local mimic.
- Keep `PANGRAM_API_KEY` outside the repo and do not call Pangram unless the key
  is intentionally present.
- Keep Pangram outside Prime reward scoring.
- Do not launch S2; W&B is still the only launch blocker.

Validation:

```text
run_pangram_bulk_detection focused tests: 2 passed
focused Pangram/manifest/readiness tests: 28 passed
expanded Prime/reward/SFT data suite: 176 passed
active changed Python files ruff check: pass
git diff --check: pass
run_pangram_bulk_detection --help: pass
run_pangram_bulk_detection --dry-run: 22 items
Pangram SDK method surface check: submit_bulk/wait_for_bulk/get_bulk_results present
```

### Prime S2 Preflight Toolchain Evidence

Date: 2026-07-01

Change:

- Added a `Prime CLI version` check to `scripts/train/prime_sft_preflight.py`.
- Updated `scripts/train/verify_prime_sft_launch_readiness.py` so readiness
  embeds the preflight report SHA256 and normalized preflight check details.
- Refreshed the S2 clean50 preflight and launch-readiness reports.

Current readiness:

```text
Prime CLI version=0.6.14
HF Dataset Viewer=train=4358 validation=242 test=243
Prime auth=pass
Qwen/Qwen3.5-2B availability=pass training_price_per_mtok=0.15
WANDB_API_KEY=missing
PANGRAM_API_KEY=missing
Prime secret list=[]
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=7b24ff8c686a4311128c0ad9e1f5beb9749ec41015a729f21dd5162ad498b7b7
launch_readiness_sha256=fd96132c6b7dbe50efcf9080bee9050dba660a289898b119acc7750a327b753a
launch_manifest_sha256=e1e85ebc38440d4d367760dec6562db9603b3ab2909e33b58c19218fddccc242
launch_archive_sha256=8368ad18a832b22280347de33bf9fdc5214d1ccea7f9199e90c4243cbaa81c9a
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
pangram_bulk_items_sha256=70ce023f0884848059ca167134b1e4188daa62620b26168371cda5411a878951
```

Decision:

- Prime login/auth is good enough for the S2 preflight path.
- Do not launch S2 yet. W&B is still the only failed launch gate.
- Keep Pangram optional and outside Prime reward scoring.

Validation:

```text
focused preflight/readiness tests: 20 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 preflight refresh: pass except WANDB_API_KEY source
live S2 readiness refresh: fail only on WANDB_API_KEY source
```

### Prime S2 Sandbox W&B Gate Tightening

Date: 2026-07-02

Change:

- Checked the live Prime CLI sandbox surface:
  `prime sandbox run --help` supports `-e KEY=VALUE` and does not show automatic
  Prime global-secret injection.
- Tightened `scripts/train/prime_sft_preflight.py` so a Prime-only
  `WANDB_API_KEY` secret no longer passes the default sandbox SFT gate.
- Added `--allow-prime-wandb-secret` as an explicit escape hatch for a future
  non-sandbox or custom secret-injected path.
- Updated `scripts/train/prepare_prime_sft_launch_kit.py` so generated launch
  kit READMEs state that local `HF_TOKEN` and `WANDB_API_KEY` are required for
  the `prime sandbox run -e ...` command.
- Regenerated the S2 clean50 launch kit, preflight report, and launch-readiness
  report.

Current readiness:

```text
Prime CLI version=0.6.14
HF Dataset Viewer=train=4358 validation=242 test=243
Prime auth=pass
Qwen/Qwen3.5-2B availability=pass training_price_per_mtok=0.15
WANDB_API_KEY=missing
PANGRAM_API_KEY=missing
Prime secret list=[]
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
WANDB_API_KEY source detail="set local WANDB_API_KEY before sandbox SFT launch"
preflight_report_sha256=0ebd4bbba2575cf76e9c6f5f685242cca9782c62f4d54e87e815778b54c0759f
launch_readiness_sha256=dec30da54678aab792c39f03de5b0bb91db61ac4c6bea2c21bbcd90ae1be458b
launch_manifest_sha256=a358a265b89d8ecf251e3c556988698cc816a07698067b0b588f6cf9864c3731
launch_archive_sha256=c75957e342efce4e1f6fc93cdc33889dc25d57c111eb572e92ffd3e04e6713b4
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
pangram_bulk_items_sha256=70ce023f0884848059ca167134b1e4188daa62620b26168371cda5411a878951
```

Decision:

- For the S2 sandbox launch kit, require local `WANDB_API_KEY`. A Prime global
  secret alone is not launch-ready because the current sandbox CLI does not
  inject it.
- Keep the launch blocked until local W&B is present and readiness passes.
- Keep Pangram optional and outside Prime reward scoring.

Validation:

```text
focused preflight/launch-kit/readiness tests: 27 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 preflight refresh: pass except local WANDB_API_KEY source
live S2 readiness refresh: fail only on local WANDB_API_KEY source
```

### Prime S2 Preflight Policy In Readiness

Date: 2026-07-02

Change:

- Added `launch_policy` to `scripts/train/prime_sft_preflight.py` JSON output:
  - `runner=prime_sandbox`;
  - `wandb_source.local_env_required=true`;
  - `wandb_source.prime_secret_allowed=false`.
- Updated `scripts/train/verify_prime_sft_launch_readiness.py` so S2 readiness
  rejects a preflight report that allows Prime-only W&B secrets for the sandbox
  launch path.
- Refreshed the S2 clean50 preflight and launch-readiness reports.

Current readiness:

```text
Prime CLI version=0.6.14
HF Dataset Viewer=train=4358 validation=242 test=243
Prime auth=pass
Qwen/Qwen3.5-2B availability=pass training_price_per_mtok=0.15
launch_policy.runner=prime_sandbox
launch_policy.wandb_source.local_env_required=true
launch_policy.wandb_source.prime_secret_allowed=false
WANDB_API_KEY=missing
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=c03b287a3ee33331d64b9772f1701e0e93882d290a0c8153535aea7efb9c2bef
launch_readiness_sha256=629dcd93b1bed4b7af23f6362696dbb4d0b08e901b26c2f38cd1030c70d6f269
launch_manifest_sha256=a358a265b89d8ecf251e3c556988698cc816a07698067b0b588f6cf9864c3731
launch_archive_sha256=c75957e342efce4e1f6fc93cdc33889dc25d57c111eb572e92ffd3e04e6713b4
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
```

Decision:

- Treat the W&B source policy as part of the pre-spend launch contract, not just
  as README prose.
- Keep S2 blocked until local `WANDB_API_KEY` is present and readiness passes.

Validation:

```text
focused preflight/launch-kit/readiness tests: 28 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 readiness refresh: fail only on local WANDB_API_KEY source
```

### Prime S2 Target Config Guard

Date: 2026-07-02

Change:

- Extended `scripts/train/prime_sft_preflight.py` so `dataset SFT config`
  rejects smoke/config drift away from the intended S2 target shape:
  - `max_steps=200`;
  - `model.seq_len=4096`;
  - `data.seq_len=4096`;
  - train/validation batches `128/64`;
  - assistant-only train/eval loss masks;
  - LoRA `rank=32 alpha=64`;
  - `optim.lr=2e-5`;
  - checkpoint interval `50`;
  - sharded safetensors weight snapshots.
- The successful config check now records target facts in the preflight detail.
- Refreshed the S2 clean50 preflight and launch-readiness reports.

Current readiness:

```text
dataset SFT config=Qwen/Qwen3.5-2B on jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50; max_steps=200 train_batch=128 val_batch=64 lr=2e-05 lora_rank=32
launch_policy.runner=prime_sandbox
launch_policy.wandb_source.local_env_required=true
WANDB_API_KEY=missing
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=706e0bac3b7dcacacec737ae057590faa480d2fca2219728bfddc4b9b4bee087
launch_readiness_sha256=7fa6c5d12dfbe2b6d202eda395bca44c88f0a049a23c48cf9a0916b1e3935a7a
launch_manifest_sha256=a358a265b89d8ecf251e3c556988698cc816a07698067b0b588f6cf9864c3731
launch_archive_sha256=c75957e342efce4e1f6fc93cdc33889dc25d57c111eb572e92ffd3e04e6713b4
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
```

Decision:

- Treat the target SFT hyperparameters as launch-contract fields, not informal
  README expectations.
- Keep S2 blocked until local `WANDB_API_KEY` is present and readiness passes.

Validation:

```text
focused preflight/launch-kit/readiness tests: 29 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 readiness refresh: fail only on local WANDB_API_KEY source
```

### Prime S2 Runtime Ref Guard

Date: 2026-07-02

Change:

- Updated `scripts/train/verify_prime_sft_launch_readiness.py` so launch
  readiness requires `launch_manifest.prime_rl_ref=d700753`.
- Added `--expected-prime-rl-ref` with default `d700753` for future intentional
  runtime-pin changes.
- Readiness now records both `prime_rl_ref` and `expected_prime_rl_ref`.
- Refreshed the S2 clean50 launch-readiness report.

Current readiness:

```text
launch_kit.prime_rl_ref=d700753
launch_kit.expected_prime_rl_ref=d700753
WANDB_API_KEY=missing
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=706e0bac3b7dcacacec737ae057590faa480d2fca2219728bfddc4b9b4bee087
launch_readiness_sha256=27043fc771d295a8123b0448f1595f92e96824a3e2552b2fbb93f55c9046f7bf
launch_manifest_sha256=a358a265b89d8ecf251e3c556988698cc816a07698067b0b588f6cf9864c3731
launch_archive_sha256=c75957e342efce4e1f6fc93cdc33889dc25d57c111eb572e92ffd3e04e6713b4
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
```

Decision:

- Treat the Prime `prime-rl` runtime ref as a launch-contract field.
- Keep S2 blocked until local `WANDB_API_KEY` is present and readiness passes.

Validation:

```text
focused preflight/launch-kit/readiness tests: 30 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 readiness refresh: fail only on local WANDB_API_KEY source
```

### Prime S2 Runner Runtime Ref Guard

Date: 2026-07-02

Change:

- Extended `scripts/train/verify_prime_sft_launch_readiness.py` so readiness
  inspects `run_sft.sh`.
- The runner must contain both:
  - `git fetch --depth 1 origin d700753`;
  - `git checkout d700753`.
- Readiness now records `runner_uses_expected_prime_rl_ref=true`.
- Refreshed the S2 clean50 launch-readiness report.

Current readiness:

```text
launch_kit.prime_rl_ref=d700753
launch_kit.expected_prime_rl_ref=d700753
launch_kit.runner_uses_expected_prime_rl_ref=true
runner_sha256=703f933925db9b3ca00019167c839d339e7b4fef623aa9e3405a2362f4ec661f
WANDB_API_KEY=missing
passed=false
failures=["preflight gate failed: WANDB_API_KEY source"]
preflight_report_sha256=706e0bac3b7dcacacec737ae057590faa480d2fca2219728bfddc4b9b4bee087
launch_readiness_sha256=f608f6621f7eb32354aafcf75a9689e745f5112ced2f51538e0bc45a526a8211
launch_manifest_sha256=a358a265b89d8ecf251e3c556988698cc816a07698067b0b588f6cf9864c3731
launch_archive_sha256=c75957e342efce4e1f6fc93cdc33889dc25d57c111eb572e92ffd3e04e6713b4
sft_eval_manifest_sha256=60289822348ae19d3fe79c363149c1aa2beb126be943dd4e001cb6314e7cc102
```

Decision:

- Treat the actual sandbox runner checkout commands as launch-contract fields,
  not just the manifest's declared `prime_rl_ref`.
- Keep S2 blocked until local `WANDB_API_KEY` is present and readiness passes.

Validation:

```text
focused preflight/launch-kit/readiness tests: 31 passed
targeted changed train/preflight/readiness ruff check: pass
live S2 readiness refresh: fail only on local WANDB_API_KEY source
```

### Prime S2 Env Lock And Launch State

Date: 2026-07-02

Change:

- Locked env `0.3.15` for the next full-model SFT plus RL ablation.
- Stopped further reward/env hardening for this branch unless launched SFT/RL
  evals expose a concrete failure.
- Rechecked live Prime state for the launch path:
  - Prime auth passes;
  - no Prime sandboxes exist;
  - `Qwen/Qwen3.5-2B` Hosted Training is currently at capacity;
  - `Qwen/Qwen3.5-9B` Hosted Training is available, but remains behind the Qwen
    2B branch in the study order;
  - Hosted SFT with W&B configured stops before launch because local
    `WANDB_API_KEY` is absent.

Decision:

- Use S2 clean50 as the next quality-oriented Qwen 2B dataset-SFT branch.
- After a promoted S2 checkpoint exists, render the S2 after-SFT RL config and
  launch Qwen 2B RL-after-SFT.
- Do not spend on S1 unless explicitly running the no-new-repairs baseline.
- The only execution decision left is whether to provide online W&B and launch
  the tracked sandbox SFT, or intentionally waive tracking and run an offline
  sandbox SFT.

Validation:

```text
env/config/ablation gate subset: 23 passed, 2 skipped
focused preflight/launch-kit/readiness tests: 31 passed
live S2 preflight: pass except local WANDB_API_KEY source
live S2 readiness: fail only on local WANDB_API_KEY source
prime sandbox list: 0 sandboxes
prime train models: Qwen/Qwen3.5-2B at_capacity=true, Qwen/Qwen3.5-9B at_capacity=false
hosted SFT launch check: blocked before spend on missing WANDB_API_KEY
```

### Prime S2 Pod Launch Attempt

Date: 2026-07-02

Change:

- Rechecked live Prime docs before spending: GPU sandboxes are currently
  CPU-only/roadmap, so the prior sandbox launch path is stale for GPU SFT.
- Pivoted the S2 clean50 launch attempt to Prime GPU pods with the `prime_rl`
  image.
- Prime CLI was upgraded from `0.6.14`; the installer reported `0.6.16`, while
  `prime --plain --version` reports `0.6.15`.
- Live availability found `A100_80GB x1` options; the only row accepted by
  `pods create` in the attempted set was Crusoe `08e3e6`.
- Uploaded the local public SSH key to the Prime SSH key API as
  `codex-id-ed25519-20260702` after the API showed zero account SSH keys.
- Created and terminated two `A100_80GB x1` pods:
  - `71b5034cc93941cd8c9ceeee4edc11d5`
    (`humanize-s2-sft-a100-r1`), terminated after SSH public-key denial;
  - `183c7c922f9842cd9a2bac97317dd8bd`
    (`humanize-s2-sft-a100-r2`), terminated after the same denial even after
    uploading the account SSH key.
- `prime --plain pods list --output json` showed zero active pods after
  termination.

Decision:

- Keep env `0.3.15` and S2 clean50 locked; the SFT/RL blocker is Prime pod SSH
  access, not reward design or SFT config readiness.
- Do not pass secrets through `prime pods create --env` for this workflow:
  the CLI echoed env values during the first pod creation attempt.
- Next launch step should fix Prime pod key injection before provisioning a new
  GPU pod. The most direct candidates are a Prime-dashboard SSH key check or a
  fresh RSA key upload/recreate attempt.
- Treat sandbox wording in S2 launch docs as obsolete for GPU execution until
  Prime GPU sandboxes ship; use pods for open `prime-rl` SFT/RL runs.

Validation:

```text
prime docs check: GPU sandboxes currently CPU-only/roadmap
prime availability: Crusoe A100_80GB x1 row 08e3e6 accepted by pods create
prime ssh keys API before upload: total_count=0
prime ssh key upload: codex-id-ed25519-20260702 created and primary
pod r1: ACTIVE/FINISHED then SSH Permission denied (publickey), terminated
pod r2: ACTIVE/FINISHED then SSH Permission denied (publickey), terminated
prime pods list after cleanup: 0 active pods
wallet impact observed after r1: about $0.02
```

### Prime S2 MassedCompute Pod Runtime Bootstrap

Date: 2026-07-03

Change:

- Kept the S2 clean50 Qwen 2B run on Prime infrastructure, not local hardware.
- Uploaded a fresh RSA key to Prime and used it for a new MassedCompute pod:
  `62abc46cde1f4705b0ce65ab702005ae`
  (`humanize-s2-sft-a100-massed-r1`), `A100_80GB x1`,
  `ubuntu@154.54.100.38`.
- Confirmed this pod is active through `prime --plain pods list --output json`.
- MassedCompute rejected the `prime_rl` image, Datacrunch `prime_rl` returned
  no valid GPU configuration, and Crusoe `prime_rl` pods still failed SSH key
  auth. The live Prime path is therefore MassedCompute Ubuntu CUDA plus runtime
  bootstrap.
- Refreshed the S2 launch kit locally and copied it to the pod:
  - archive:
    `9d0419131be9d84d4bb6ea29914479e8db6395afffef7bfe3ed6d355ca082e5c`;
  - runner:
    `7cf0a338f99f617bb6448f4c570e7adff3cd0760a2d4520ee36295e776220807`;
  - config:
    `b3775839dda7abac69be33eb28b1c83c74c2e9c486d9293f3e6f087cd2d5d18a`;
  - manifest:
    `fe3d5ef43131021b2ffcb7ffdcc2ae8622e4120d981bb0249029f72f31651054`.
- Patched the launch runner to:
  - use sudo-aware apt installation;
  - rewrite GitHub SSH submodule URLs to HTTPS;
  - force recursive submodule init for `prime-rl@d700753`;
  - bootstrap missing `flash-attn==2.8.3.post1` with CUDA 12.8 nvcc,
    `g++-12`, `ninja`, and `FLASH_ATTN_CUDA_ARCHS=80`.
- The active remote bootstrap is compiling `flash-attn` inside Prime's `uv run`
  Python/Torch environment on the A100 pod. SFT should start only after
  `flash_attn_2_cuda` imports successfully.

Decision:

- Use Prime pods for open `prime-rl` dataset SFT/RL until Prime GPU sandboxes
  exist.
- Do not pass secrets through `prime pods create --env`; the CLI can echo env
  values. Use the pod-side `/workspace/sft.env` instead.
- Waive online W&B for this launch. The pod uses `WANDB_MODE=offline` and
  `WANDB_API_KEY=offline`; local launch readiness still records missing
  `WANDB_API_KEY`, but that is no longer blocking this Prime-pod run.
- Do not start another pod or another `flash-attn` build while the active build
  is still running.

Validation:

```text
prime docs check: prime-rl supports uv run sft/rl; GPU sandboxes are CPU-only; pods are the GPU path
prime active pod: 62abc46cde1f4705b0ce65ab702005ae ACTIVE A100_80GB x1
remote SSH: pass with RSA key
local/remote launch archive hashes: match
focused launch-kit/readiness tests: 16 passed
launch readiness: fail only on WANDB_API_KEY source, intentionally waived for offline pod run
remote runtime state: flash-attn source build in progress on Prime pod with only sm_80 gencode
```

### Prime S2 Prime-Compatible Dataset Relaunch

Date: 2026-07-03

Change:

- The first S2 pod SFT run reached `uv run sft`, built/imported
  `flash-attn==2.8.3.post1`, downloaded `Qwen/Qwen3.5-2B`, initialized the
  Qwen3.5 renderer and LoRA trainer, then failed at dataset load.
- Failure root cause:
  `ValueError: Feature type 'Json' not found`. The old S2 Hub dataset exposed
  the nested `quality` column as `_type: Json`, which the Prime `datasets`
  stack pinned in `prime-rl@d700753` does not accept.
- Built and published a Prime-compatible replacement dataset with identical
  rows and splits, but training rows now keep only Prime-safe fields:
  `id`, `messages`, `domain`, `task_type`, `mode`, `source`, `license`,
  `release_eligible`, and `split`.
- New dataset:
  `jayshah5696/humanize-rl-prime-sft-messages-env0315-clean50-primecompat`
  at HF commit `8f1d484cea21affed944479fdb3ef590de03a6ba`.
- Dataset Viewer verified:
  `train=4358`, `validation=242`, `test=243`; features are `Value` plus
  `messages` as `List`, with no `Json` feature type.
- Updated the active S2 config to the `-primecompat` dataset and rebuilt the
  launch kit:
  - config:
    `40071c8db47c0830a21d6dfb65c6a787971d0ab8aa20877663385ea68d12ade9`;
  - preflight:
    `feedb1cbbda0ab9010514895c9155e42b48d42b72eca9bdc4f3dca3dad88e595`;
  - readiness:
    `77824beb531ed817dff70d414ac00ae37f9ff92118dd9959f243725ee92fca14`;
  - launch manifest:
    `0bcc844688127d6abd2f84f93583cf4cb5e9860413642bbd33384de7318e91f6`;
  - runner:
    `7cf0a338f99f617bb6448f4c570e7adff3cd0760a2d4520ee36295e776220807`;
  - archive:
    `05e91c189140f9d2c4e6b6c0185b1225349fede07631fcc7314fa8e3fb753358`;
  - dataset manifest:
    `4c0026965bc23f873f5a17b5b8f3e1cb65c0dece17a833df281eda764a44ec5e`.
- Added a preflight guard that fails any HF Dataset Viewer feature exposing
  `_type: Json`, so this exact Prime load failure is caught before launch.
- Restaged the rebuilt kit on the Prime A100 pod
  `62abc46cde1f4705b0ce65ab702005ae` and moved the old failed output
  directory aside:
  `outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315_failed_json_feature_20260703T0526Z`.
- First `-primecompat` relaunch on the same Prime pod:
  - PID: `20530`;
  - PID file: `/workspace/s2_sft_logs/run_sft_primecompat.pid`;
  - outer log:
    `/workspace/s2_sft_logs/run_sft_primecompat_20260703T053505Z.log`;
  - trainer log:
    `/workspace/prime-rl/outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315/logs/trainer.log`;
  - offline W&B run:
    `outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315/wandb/offline-run-20260703_053529-v4wgvlpl`.
- That first `-primecompat` relaunch passed dataset load and emitted step-0
  metrics, then failed at `05:43:08Z` with:
  `RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
  ptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`.
- Added a guarded runner path that writes `sitecustomize.py` in
  `/workspace/prime-rl`, disables `torch.backends.cudnn.enabled`, and exports
  `TORCH_CUDNN_V8_API_DISABLED=1` for the trainer interpreter.
- Guarded launch-kit hashes:
  - launch manifest:
    `3ccd9dd11608ef52adf94f16f657730245046a09f0d445ae92f42ed97bc45887`;
  - runner:
    `02e117cc4934b77fc5ab5c65ff0bc2fcf9f04eb080ddab91b70e8ca8c0e9dbf7`;
  - archive:
    `191342cf4bb07e5741e18dbe4c509037285b311a9cc17d129d7fa08ad6ea1836`;
  - readiness:
    `34237d4ecc6eb90d0bff601532962d18e96f286b82dda39dab4a1928bd6d8335`.
- Guarded S2 SFT relaunch on the same Prime pod:
  - PID: `22250`;
  - PID file: `/workspace/s2_sft_logs/run_sft_cudnn_guard.pid`;
  - outer log:
    `/workspace/s2_sft_logs/run_sft_cudnn_guard_20260703T054716Z.log`;
  - trainer log:
    `/workspace/prime-rl/outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315/logs/trainer.log`;
  - offline W&B run:
    `outputs/prime_sft/qwen35_2b_sft_target_messages_env0315_clean50_gate_env0315/wandb/offline-run-20260703_054740-sxt2aig3`.

Live status:

```text
SFT location: Prime A100 pod, not local
pod: 62abc46cde1f4705b0ce65ab702005ae humanize-s2-sft-a100-massed-r1
dataset load: pass, train=4358 validation=242 test=243
trainer: Starting training loop (max_steps=200)
step 0 validation: Loss 2.0132
step 0 train: Loss 2.1609, Grad. Norm 2.3594, LR 2.00e-05
step 1 train: Loss 2.1475, Grad. Norm 2.1875, throughput 2545 tokens/s
step 2 train: Loss 2.0988, Grad. Norm 1.7734, throughput 2544 tokens/s
memory: Peak Mem. 20.2/79.2 GiB
latest GPU poll: 64% utilization, 21653/81920 MiB used
```

Decision:

- The S2 SFT ablation is now live on Prime GPU infrastructure.
- Keep online W&B waived for this run; the offline W&B directory is the run
  record until we choose to sync it.
- Do not launch RL-after-SFT until this SFT run produces a real checkpoint and
  passes the S2 promotion gates.
- Do not use the old `clean50` dataset for Prime SFT; use `clean50-primecompat`
  for active Prime runs.

Validation:

```text
prepare_prime_sft_messages_dataset / preflight / launch-kit / readiness / config / output verifier focused tests: 45 passed
local datasets load: train=4358 validation=242 test=243, no Json feature
HF Dataset Viewer: pass - train=4358 validation=242 test=243
launch readiness: fail only on local WANDB_API_KEY source, intentionally waived for offline pod run
remote SFT: passed prior Json feature failure point, passed prior cuDNN crash point, and emitted step 2 metrics
```

### Prime RL After-SFT Full400 Attempt

Date: 2026-07-03

Change:

- Treated the S2 Qwen3.5-2B SFT as already completed at step 200. Do not rerun
  SFT for this ablation.
- Created the after-SFT RL full400 config:
  `configs/prime_rl/qwen35_2b_rl_after_sft_env0315_clean50_step200_full400.toml`.
- Config keeps the promoted SFT model
  `jayshah5696/humanize-rl-qwen35-2b-sft-env0315-clean50-primecompat-step200`,
  env `jayshah5696/humanize-rl-env@0.3.15`, Qwen3.5 renderer, RL
  `batch_size=64`, `group_size=8`, eval every 50 steps, and offline W&B.
- Set `max_steps=400`, renamed output/tags to `full400`, and set checkpoint
  retention to `keep_last=8`.
- Confirmed `prime pods list` was empty before launch.
- Launched one Prime Datacrunch pod:
  `80161285ef8e4dafadfb5b3fc97da056`
  (`humanize-rl-after-sft-qwen35-2b-a100x2-full400-r1`), `A100_80GB x2`,
  created `2026-07-03 21:06:24 UTC`.
- Staged the launch kit on the pod and verified transferred hashes matched.

Launch correction:

- The first runner path manually installed `flash-attn==2.8.3.post1` from PyPI,
  which pulled the source tarball and began compiling 72 `sm_80` kernels on the
  paid A100 host.
- Stopped that source-build runner before training started.
- Prime `prime-rl@d700753` already pins a compatible prebuilt wheel through the
  `flash-attn` extra:
  `flash_attn-2.8.3+cu128torch2.11-cp312-cp312-linux_x86_64.whl`.
- Corrected the same pod to run:
  `uv run --extra flash-attn rl @ /workspace/prime_rl_launch_kit/config.toml`.
- The corrected path installed one package in 11 ms and verified
  `flash_attn_2_cuda OK`.
- Local launch archive was rebuilt with the corrected runner:
  - archive:
    `3365c03a1f80e4a250f617b88f6f8a6f1573934046ae97cdf826fb8e7125e75f`;
  - config:
    `c91c59fcf3d11d0d779e117c5890eba7630258eae56a140a413b48c063db59bb`;
  - runner:
    `d69234b8357bd3028cef49af02273fcefecdf4720f347ad607fb68d07d4dbb08`.

Result:

- Corrected RL launch reached Prime startup:
  model predownload passed, subconfigs were written, inference started on GPU 0,
  orchestrator started, and trainer started on GPU 1.
- Orchestrator initialized the tokenizer, Qwen35 renderer, offline W&B, rollout
  filters, and began loading training environments.
- The run then failed before any completed RL step:
  `Inference failed with exit code 1` at `2026-07-03 21:24:51 UTC`.
- Stop rule was applied immediately. The pod was terminated with
  `prime --plain pods terminate 80161285ef8e4dafadfb5b3fc97da056 --yes`.
- Follow-up `prime --plain pods list --output json` returned zero pods.
- No second paid attempt was started.
- The pod became unreachable after termination, so the deeper
  `inference.log` error was not recovered. The next approved paid run should
  copy logs to local storage or include a fail-trap before termination.

Decision:

- SFT remains done; do not rerun SFT.
- Do not start another RL full400 attempt without explicit approval.
- Do not use a runner that manually installs `flash-attn` from PyPI for
  `prime-rl@d700753`; use `uv run --extra flash-attn ...` so Prime's pinned
  prebuilt Torch 2.11/CUDA 12.8 wheel is used.
- Before the next paid RL retry, add a pod-side failure trap or live log copy
  for `inference.log`, `orchestrator.log`, and `trainer.log` so the exact
  failure is preserved while still terminating compute promptly.

### Prime RL After-SFT Full400 A100x1 Colocated Attempt

Date: 2026-07-04

Intent:

- Retry the after-SFT RL full400 launch on one `A100_80GB` instead of two.
- Keep the Prime TOML workflow and avoid upstream source patches.
- Use a colocated launcher trick: expose physical GPU `0` twice with
  `CUDA_VISIBLE_DEVICES=0,0`, so Prime's stock single-node launcher maps
  inference and trainer roles to the same physical A100.

Config:

- Added
  `configs/prime_rl/qwen35_2b_rl_after_sft_env0315_clean50_step200_full400_a100x1_colocated.toml`.
- It keeps `max_steps=400`, SFT model
  `jayshah5696/humanize-rl-qwen35-2b-sft-env0315-clean50-primecompat-step200`,
  env `jayshah5696/humanize-rl-env@0.3.15`, Qwen3.5 renderer, RL
  `batch_size=64`, `group_size=8`, eval every 50 steps, and offline W&B.
- To fit trainer and vLLM on one A100, the config lowers only runtime pressure:
  `max_inflight_rollouts=16`, `tasks_per_minute=180`, and
  `inference.gpu_memory_utilization=0.42`.
- Corrected launch-kit hashes after the Ubuntu `uv` config fix:
  - archive:
    `4cc64e27d96fb02cb09ad61d75db4c8803ae8768e0702448d5057872f9eeab39`;
  - config:
    `451ae29ad1cf3f07636f42f62ae72a5a462e607ec55035ffa0b2e60d4e4fc5bb`;
  - runner:
    `6c42586f221aa0f153ea4fbe0a88f0defed1096d9d80527c1550d3c2c4c8754a`.

Attempt:

- Confirmed Prime pod list was empty before launch.
- Launched one Prime MassedCompute pod:
  `c581141dcc104415b4978c12fd5b5f0c`
  (`humanize-rl-after-sft-qwen35-2b-a100x1-full400-colocated-r1`),
  `A100_80GB x1`, created `2026-07-04 00:43:15 UTC`, price shown by Prime
  availability as about `$1.20/h`.
- SSH verified one `NVIDIA A100 80GB PCIe, 81920 MiB`.
- Staged the launch kit and verified transfer hashes:
  archive `6054854bbf5c9ee7870274c905a49a01033cb4e52bf8af5401846e4e6322a779`,
  config `451ae29ad1cf3f07636f42f62ae72a5a462e607ec55035ffa0b2e60d4e4fc5bb`,
  runner `b8cbd5c47b60dbc2b763c23cf63a38aae1befef408493ef06a8c00660cfe3c74`.
- The first 1x runner failed before training during `uv` install:
  `ERROR: unable to create receipt directory at /home/ubuntu/.config/uv`.
- Failure bundle was copied locally to
  `runs/prime_rl_launch_kit/failure_bundle_qwen35_2b_rl_after_sft_env0315_clean50_step200_full400_a100x1_colocated_20260704T004610Z.tar.gz`.
- Stop rule applied. Pod terminated:
  `prime --plain pods terminate c581141dcc104415b4978c12fd5b5f0c --yes`.
- Follow-up `prime --plain pods list --output json` returned zero pods.

Fix:

- Updated the 1x runner to set writable pod-side paths:
  `XDG_CONFIG_HOME=/workspace/.config`,
  `XDG_CACHE_HOME=/workspace/.cache`, and
  `UV_CACHE_DIR=/workspace/.cache/uv`.
- No second paid 1x attempt has been launched after this fix.

Decision:

- The 1x A100 path is prepared but not yet proven.
- Do not launch another paid attempt until approved.
- If approved, use the corrected archive
  `4cc64e27d96fb02cb09ad61d75db4c8803ae8768e0702448d5057872f9eeab39`.

### Hosted Training Correction: No Pod For RL Env Full400

Date: 2026-07-03

Current status:

- All Prime pods are stopped. `prime --plain pods list --output json` returned
  zero pods.
- Installed Prime CLI supports Hosted Training with
  `prime train run <config.toml>`. It also accepts `prime train <config.toml>`;
  `prime rl` is a deprecated alias for `prime train`.
- The correct Hosted RL surface for this repo is the `configs/prime/` TOML
  family, not the open `configs/prime_rl/` pod/sandbox launch path.

Course check:

- `anakin87/llm-rl-environments-lil-course` validates the two-step pattern:
  SFT first, then RL against an environment.
- That course's SFT path uses open PRIME-RL on a GPU machine:
  `uv run sft @ primerl_sft.toml`.
- Its RL path uses Verifiers `vf.RLTrainer` on a GPU machine:
  `uv run vf-rl @ vfrltrainer_rl1.toml`.
- The course supports SFT -> RL, but does not remove the distinction between
  open GPU-machine training and Prime Hosted Training. For our RL env run,
  Hosted Training should be preferred when checkpoint handoff is valid.

Action:

- Added Hosted full400 after-SFT template:
  `configs/prime/qwen35_2b_p5050_after_sft_env0315_full400_template.toml`.
- Updated hosted README examples to use `prime train run`.

Blocker:

- Current Qwen SFT verification has no Hosted checkpoint id:
  `sft_eval_manifest.checkpoint_id = null`.
- Hosted warm-start requires a READY same-model Prime checkpoint id.
- Qwen Hosted base-RL checkpoint `jgeit425lcztmwslc50rbom1` is still
  `UPLOADING`, not READY.
- Llama Hosted checkpoint `arsnu29hb9akbm2jc1b33pmc` is READY but belongs to
  `meta-llama/Llama-3.2-3B-Instruct`, not `Qwen/Qwen3.5-2B`.
- Follow-up inventory with
  `prime --plain train list --num 100 --output json` found exactly one Hosted
  `Qwen/Qwen3.5-2B` run: `o48ryskshkn06b3o1b1kauql`, the stopped base-RL
  run above.
- Installed Prime CLI `0.6.15` exposes no Hosted checkpoint import/upload
  command. The only Hosted warm-start field exposed by
  `prime train configs --output json` is top-level `checkpoint_id`.

Decision:

- Do not launch more pods for this RL goal.
- Do not launch after-SFT Hosted full400 until a valid Qwen 3.5 2B READY Prime
  checkpoint id exists and passes the warm-start verifier.
- If we cannot produce/import that checkpoint id, the only honest Hosted RL
  launch is base-model RL full400, not after-SFT RL.
- Do not launch base Qwen 2B RL full400 as a substitute without a new approval,
  because the prior base Qwen 2B full200 run was stopped at step 55 after strict
  eval regression.

### Same-Artifact Hosted Follow-Up

Date: 2026-07-03

Goal:

- Keep the run honest: use the exact completed SFT artifact, not a fake
  checkpoint placeholder and not the prior base-RL adapter.

Findings:

- Existing open `prime-rl` after-SFT full400 config already contains a
  `[deployment]` block, so installed Prime CLI dispatches it to the managed
  dedicated full-FT endpoint with `prime train run`.
- Launch command attempted:
  `prime --plain train run -e HF_TOKEN --yes --output json configs/prime_rl/qwen35_2b_rl_after_sft_env0315_clean50_step200_full400.toml`.
- Prime rejected it before compute started:
  `HTTP 403: Dedicated training runs are admin-only`.
- No paid compute started.
- Hugging Face metadata confirms the exact SFT model is public and contains
  `model.safetensors`, `lora_adapters/adapter_config.json`, and
  `lora_adapters/adapter_model.safetensors`:
  `jayshah5696/humanize-rl-qwen35-2b-sft-env0315-clean50-primecompat-step200`.
- Submitted a Prime Hosted Training model request for that exact HF model.
- Prime deployable adapters list has READY adapters for the stopped Qwen 2B
  base-RL run, but those are not the SFT artifact and must not be treated as
  after-SFT.

Action:

- Added SFT-as-base shared Hosted template:
  `configs/prime/qwen35_2b_p5050_sft_model_env0315_full400_template.toml`.

Decision:

- Do not use adapter ids from the stopped base-RL run as `checkpoint_id`.
- Do not launch the SFT-as-base Hosted template until
  `prime train models --output json` lists the exact SFT HF model.
- Once Prime enables that model, the clean command is:
  `prime --plain train run configs/prime/qwen35_2b_p5050_sft_model_env0315_full400_template.toml --yes --output json`.

### Direct Prime-RL Full400 Attempt: MassedCompute 2x A100 DNS Failure

Date: 2026-07-08

Intent:

- User approved bypassing the Hosted Training wait and running the direct open
  `prime-rl` after-SFT RL job now.
- Target config:
  `configs/prime_rl/qwen35_2b_rl_after_sft_env0315_clean50_step200_full400.toml`.
- Launch archive:
  `runs/prime_rl_launch_kit/qwen35_2b_after_sft_env0315_clean50_step200_full400.tar.gz`.

Pre-flight:

- Confirmed `prime --plain pods list --output json` returned zero pods.
- Prime CLI `pods create --id 70eb73 ... --env HF_TOKEN=...` could not match
  the available 2x A100 resource because the CLI filters out Ubuntu-only
  configs when env vars are passed.
- Prime API confirmed the target resource was available:
  `massedcompute`, `gpu_2x_a100`, `A100_80GB x2`, `us-central-3`,
  `$2.40/h`, image list `["ubuntu_22_cuda_12"]`.

Attempt:

- Direct API create with creation-time env vars failed before provisioning:
  `HTTP 400: Environment variables are not allowed for this request`.
- Retried without creation-time env vars and created pod
  `a64c16b16a884a218ec79edc94332324`
  (`humanize-rl-after-sft-qwen35-2b-direct-full400-r4`),
  MassedCompute `A100_80GB x2`, `$2.40/h`.
- Pod became `ACTIVE`, SSH `ubuntu@216.81.248.94`.
- Verified two GPUs:
  `NVIDIA A100 80GB PCIe, 81920 MiB` x2.
- Plain Ubuntu image did not include `/workspace`; created it and staged the
  launch kit there.
- Remote hash verification matched:
  archive `3df2d89bdc69f2349573a9ca9ab0828adb59886a491a440a28171260524c3eac`,
  config `c91c59fcf3d11d0d779e117c5890eba7630258eae56a140a413b48c063db59bb`,
  runner `d062d5cdcd5491bf84a44b17a0f9a80ba0435e97b7b6d94b91f8f4e22adff51b`.

Failure:

- Started launcher as PID `2245`.
- Failure happened before `uv`, `prime-rl`, model load, or RL step 0.
- Launcher failed on:
  `curl: (6) Could not resolve host: astral.sh`.
- Failure bundle copied locally to:
  `runs/prime_rl_launch_kit/failure_bundles/a64c16b16a884a218ec79edc94332324/`.
- Bundle confirms GPUs were healthy and idle, and `/workspace` had about
  `1.5T` free.

Shutdown:

- Stop rule applied because launch failed before training started.
- Terminated pod:
  `prime --plain pods terminate a64c16b16a884a218ec79edc94332324 --yes`.
- Verified `prime --plain pods list --output json` returned zero pods.

Decision:

- No RL training ran in this attempt.
- Do not start another direct pod attempt until the bootstrap path avoids
  runtime DNS dependency on `astral.sh` or uses an image with `uv` already
  available.
- Post-failure hardening updated the direct 2x runner to DNS-preflight
  `astral.sh` and `github.com` and apply a `1.1.1.1` / `8.8.8.8` resolver
  fallback before installing `uv` or cloning `prime-rl`.
- Rebuilt clean direct 2x archive without macOS `._*` files:
  `runs/prime_rl_launch_kit/qwen35_2b_after_sft_env0315_clean50_step200_full400.tar.gz`.
- New hashes:
  archive `495c1dcd5281109466f9c5f960097f3ac8086d167546c3e1861af7da7ba20178`,
  config `c91c59fcf3d11d0d779e117c5890eba7630258eae56a140a413b48c063db59bb`,
  runner `7317cdfc3adbb17461f98f670696269bf9c346f13c81480d626874f580c6e210`.

### Corrected Direct RL Interpretation: Hosted Base RL, No SFT

Date: 2026-07-08

Correction:

- User clarified that "do RL directly" means run Hosted RL from the base model,
  without SFT warm-start and without the SFT artifact.
- This is not the open `prime-rl` pod path.
- Correct surface: `prime train run <configs/prime/...toml>`.

Prepared config:

- Added `configs/prime/qwen35_2b_p5050_env0315_full400.toml`.
- Shape:
  - `model = "Qwen/Qwen3.5-2B"`;
  - no `checkpoint_id`;
  - no SFT model or after-SFT reference;
  - `max_steps = 400`;
  - `batch_size = 64`;
  - `rollouts_per_example = 8`;
  - `max_inflight_rollouts = 32`;
  - env `jayshah5696/humanize-rl-env@0.3.15`;
  - train task `mix_v2_p5050`, reward `p50_50_no_penalty`;
  - evals: `mix_v2_p5050`, `v02_smoke` strict, `v03` strict;
  - checkpoint/adapters every `50`, keep `8`.

Checks:

- TOML parsed locally.
- Verified no `checkpoint_id`, `sft`, or after-SFT string in the config.
- `prime --plain pods list --output json` returned zero pods.
- `prime --plain train models --output json` shows `Qwen/Qwen3.5-2B` available
  and not at capacity, training price `$0.15/MTok`.

Launch command, if approved:

```bash
prime --plain train run configs/prime/qwen35_2b_p5050_env0315_full400.toml --yes --output json
```

Launch:

- User approved full Hosted base-RL run.
- Command:
  `prime --plain train run configs/prime/qwen35_2b_p5050_env0315_full400.toml --yes --output json`.
- Run id: `ln8ui3bmtx4skvxcu7pwvvbl`.
- Run name: `humanize-p5050-qwen35-2b-base-full400-env0315-r1`.
- Initial status: `PENDING`.
- Prime environment action check passed for `jayshah5696/humanize-rl-env`.
- No pods were launched; `prime --plain pods list --output json` returned zero
  pods immediately before launch.

Early status:

- Run moved to `RUNNING`; `started_at = 2026-07-08 20:54:55.663000`.
- Orchestrator log reached `Starting orchestrator loop (max_steps=400)`.
- Confirmed base run: log says `Training from scratch`.
- Step 0 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.4680149649051104`;
  - `eval_v02_strict_env0315/avg@1 = -0.03203864921298291`;
  - `eval_v03_strict_env0315/avg@1 = -0.31398944032614434`.
- At step 0 usage API still reported `total_tokens = 0`, `total_cost_usd = 0`.
- Step 50 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.4954063282882752`;
  - `eval_v02_strict_env0315/avg@1 = -0.24111945105509625`;
  - `eval_v03_strict_env0315/avg@1 = -0.46239168958096394`.
- Step 50 checkpoint id: `udaiv2e5svs9v3av3qevb8yu`.
- User requested full run despite earlier stop-rule style concerns; continue to
  full 400 and collect final score.
- Step 100 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.5991794093190483`;
  - `eval_v02_strict_env0315/avg@1 = -0.12140230706168545`;
  - `eval_v03_strict_env0315/avg@1 = -0.49897064914888084`.
- Step 100 checkpoint id: `lyuz0i0ku4rl3m552s0hl7jt`.
- Step 150 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.6211343963005735`;
  - `eval_v02_strict_env0315/avg@1 = -0.015911872674921113`;
  - `eval_v03_strict_env0315/avg@1 = -0.5616430495710423`.
- Step 150 checkpoint id: `ysd8wvky6ieqj6mpvpwzqsdk`.
- Step 200 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.657395762600936`;
  - `eval_v02_strict_env0315/avg@1 = 0.2943192811676549`;
  - `eval_v03_strict_env0315/avg@1 = -0.015758303823796196`.
- Step 200 checkpoint id: `ocn03o4gydpbwll8kpfchdtn`.
- Step 250 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.6700657164560968`;
  - `eval_v02_strict_env0315/avg@1 = 0.41526919993938827`;
  - `eval_v03_strict_env0315/avg@1 = -0.06080543639399875`.
- Step 250 checkpoint id: `gypnjylfvqz87aa9mvb73y2s`.
- Step 300 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.6715349756289073`;
  - `eval_v02_strict_env0315/avg@1 = 0.3328830744605463`;
  - `eval_v03_strict_env0315/avg@1 = -0.12436695657220899`.
- Step 300 checkpoint id: `xn1g88qiqrwtvsi65z3bv20b`.
- Step 350 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.717125491476916`;
  - `eval_v02_strict_env0315/avg@1 = 0.2968331216110124`;
  - `eval_v03_strict_env0315/avg@1 = 0.10482476999553221`.
- Step 350 checkpoint id: `ibgt0c5n1fqhv3502mg2f8kr`.
- Run completed at `2026-07-08 22:37:27.480000`.
- Final step 400 eval scores:
  - `eval_mix_v2_p5050_env0315/avg@1 = 0.7093008879222907`;
  - `eval_v02_strict_env0315/avg@1 = 0.43871350751982796`;
  - `eval_v03_strict_env0315/avg@1 = 0.124382966841523`.
- Final reported usage:
  - training tokens `12,367,936`, cost `$1.8552`;
  - inference tokens `12,986,407`, cost `$1.1554`;
  - total tokens `25,354,343`, total cost `$3.0106`.
- Checkpoints:
  - step 50: `udaiv2e5svs9v3av3qevb8yu`, `READY`;
  - step 100: `lyuz0i0ku4rl3m552s0hl7jt`, `READY`;
  - step 150: `ysd8wvky6ieqj6mpvpwzqsdk`, `READY`;
  - step 200: `ocn03o4gydpbwll8kpfchdtn`, `READY`;
  - step 250: `gypnjylfvqz87aa9mvb73y2s`, `READY`;
  - step 300: `xn1g88qiqrwtvsi65z3bv20b`, `READY`;
  - step 350: `ibgt0c5n1fqhv3502mg2f8kr`, `READY`;
  - step 400: `un0pjopifbkt5s37v4vc514h`, still `UPLOADING` after 20
    post-completion polls ending `2026-07-08 22:48:42 UTC`.
- `prime --plain pods list --output json` returned zero pods after completion.

HF publication and report:

- Published Hugging Face repo:
  `https://huggingface.co/jayshah5696/humanize-p5050-qwen35-2b-base-full400-env0315-r1`.
- Initial upload commit:
  `79289e37b69f64634313d6f25f4b38dcc2a780ae`.
- README clarification commit:
  `4fa16bddbac0388f9b5ce0888d009e7865596995`.
- Local report artifact:
  `runs/reports/prime_qwen35_2b_base_full400_env0315_hf/FINAL_REPORT.md`.
- HF artifact contents include `README.md`, `FINAL_REPORT.md`,
  `training_config.toml`, `eval_summary.json`, `usage.json`,
  `checkpoints.json`, `adapters.json`, `before_after_examples.json`,
  `metrics.json`, `run.json`, and rollout samples every 10 steps from `0`
  through `390`.
- Publication caveat: this is a Prime adapter reference and report, not a
  standalone downloadable Transformers checkpoint. Prime exposes the adapter as
  hosted ids, including final step 400 adapter `wxhjhfx6hc7xzuqbneinr7zr`.
- Qualitative report caveat: matched before/after rollout examples show metric
  improvement but also reward-hacking artifacts (`thanks out`, `sign up`,
  repeated greetings/signatures). Treat as a candidate/report, not production.
