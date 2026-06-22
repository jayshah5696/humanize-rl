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
- No `rtk` config is present.
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
  active configs pin `0.3.14`, use 4096 generation tokens, disable Qwen thinking, and contain no `rtk`.
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
  no committed W&B token in touched files, no `rtk` in the train-only config.
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
  `APPROVED`; config assertions, secret/`rtk`/eval-val grep, focused pytest,
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
- Model/env/reward unchanged:
  `Qwen/Qwen3.5-0.8B`, `jayshah5696/humanize-rl-env@0.3.14`,
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
