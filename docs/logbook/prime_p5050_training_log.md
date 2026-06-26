# Prime p50 Training Log

## 2026-06-18/19

### Data Gate

- Task set: `mix_v2_p5050`
- File: `data/rl/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`
- Published-env bundle:
  `environments/humanize_rl_env/humanize_rl_env/humanize_tasks_rl_mix_v2_p5050_filtered.jsonl`
- Rows: 972 total; 777 train / 92 validation / 103 test.
- Row shape: `prompt`, `answer`, `info`, `example_id`; full RL task preserved
  under `info.task`.
- Ridge state bundled at `humanize_rl_env/ridge_state.pkl`.

### Env `0.3.6`

- Prime env: `jayshah5696/humanize-rl-env@0.3.6`
- Wheel sha: `c831f2ada610c0b4ed4f335915178d5f163fe1d41f38155307ed603eedb909aa`
- Change: target-word length and repetition diagnostics/caps inside the
  deterministic half of p50.

Stopped run:

- Run: `beixwu41osp530um7bfabn8k`
- Model: `Qwen/Qwen3.5-0.8B`
- Config: `configs/prime/qwen35_08b_debug.toml`
- Stopped at step 21.
- Reason: reward rose while truncation/repetition exploded. Step 20 reward was
  about `0.875` with about `56%` truncation and about `50%` repetition.

One-step smoke:

- Run: `xi7nu5o2yu8761wuvxv0wjom`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/c91u4trl`
- Result: technical pass, but verifier blocked scale-up.
- Reason: `rl_v03_000392` samples recommended unsuitable films for an uplifting
  romance prompt while still scoring around `0.68-0.72`.

### Env `0.3.7`

- Prime env: `jayshah5696/humanize-rl-env@0.3.7`
- Wheel sha: `890f3eef43ac87354ef5c12072dd4d0837a215b87b6f6dc72cd56181a4eedf0f`
- Change: deterministic `recommendation_suitability` for the observed
  uplifting-romance recommendation leak. `p50_50_no_penalty` still optimizes
  only `0.5 * ridge_rubric + 0.5 * deterministic`; penalties remain diagnostics.

Local verification:

- Focused tests: `44 passed, 2 skipped`.
- Lint: `ruff check` passed.
- Bundled-env smoke: `v02_smoke`, `v03`, and `mix_v2_p5050` loaded; bundled
  ridge loaded; bad `rl_v03_000392` output scored with deterministic
  contribution `0.0`.
- Published-env local Prime eval:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env037`
  - Model: `Qwen/Qwen3.5-0.8B`
  - Examples: 3
  - Reward avg: `0.752`
  - Truncation: `0%`
  - Output tokens avg: `49.7`

Hosted one-step smoke:

- Run: `z3kf1g2nesrhc2xunybpi4mf`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/uahdkmzb`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Result: completed.
- Eval step 0 reward: `0.7619`
- Eval step 1 reward: `0.7617`
- Train step 0 reward: `0.6058`
- Trainable: `16/16`
- Error: `0%`
- Truncation: `0%`
- Filters: `0`
- Decode length mean/max: `111.6` / `165`
- Rollout audit: 16 samples; reward mean/min/max `0.6058` / `0.2639` /
  `0.8518`; romance-task bad/unknown recommendation samples now score low
  (`0.26-0.39`). Base quality is still mixed and formal, but no runaway or
  option-menu failure appeared.
- User audit blocker: W&B sample view showed a `problem_id=595` romance rollout
  with emojis, fake warmth, unsuitable movie recommendations, and signoff still
  displayed around `0.764`. Decision: do not launch the 50-step run from
  `0.3.7`.

### Env `0.3.8`

- Change: task-aware `emoji`, `hashtag`, and `all_caps` diagnostics; expanded
  signoff detection; broader observed bad-title list for the romance trace.
- Humanizer-skill alignment: decorative emoji/unicode, fake social sparkle,
  pedagogical framing (`let's dive`, `think about it`), and shouty caps are
  treated as deterministic failures unless the task explicitly asks for them.
- Local direct check on the screenshot-style `rl_v03_000392` sample:
  reward `0.273`, deterministic contribution `0.0`, penalties include
  `emoji`, `signoff`, `ai_tell_phrase`, `unsuitable_recommendation`, `too_long`.
- Local direct all-caps check on `rl_v03_000480`: reward `0.482`, penalties
  include `all_caps` and `too_short`.
- Verifier blocked the first `0.3.8` draft because title-like subject headings,
  inline signoffs, and detected markdown/bullets could still keep high
  deterministic reward. Follow-up fix adds first-line title detection, inline
  signoff detection, and a hard-format cap for `subject_line`, `signoff`,
  `placeholder_disallowed`, `wrong_format_markdown`, `wrong_format_bullets`,
  and `wrong_format_heading`.
- Bundled-env follow-up checks:
  - screenshot-style romance sample: reward `0.315`, deterministic `0.0`;
  - all-caps sample: reward `0.482`;
  - title + inline signoff sample: reward `0.550`;
  - markdown/bullet sample: reward `0.418`.
- Verifier caught one final signoff miss: `Best regards,` followed by
  multi-line name/title/company. Added `best regards` to standalone signoff
  detection. Bundled-env check now scores the block at `0.565`, deterministic
  `0.2`, with `signoff` present.
- Published `0.3.8` and ran hosted smoke:
  - Run: `jfwfr98t1ncdeiab17w5rnss`
  - W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/7b7dkxk2`
  - Eval step 0/1 reward: `0.7685` / `0.7683`
  - Train reward: `0.5177`
  - Trainable: `16/16`; error `0%`; truncation `0%`; decode mean/max `98.6` /
    `156`.
- Decision: do not launch 50-step from `0.3.8`. Highest-reward samples still
  used formal letter openings and inline signatures, e.g. `Dear William Brown,
  ... Best regards, Patricia Adams, Engineer, Cloud Nine Systems`.

### Env `0.3.9`

- Change: formal `Dear ...` salutation diagnostic, stronger inline signoff
  detection for title/company suffixes, and `salutation` included in the
  hard-format cap.
- Follow-up fix after verifier block: permission checks now treat negative
  clauses as dominant for salutations, emoji, hashtags, and all caps. A prompt
  such as `MUST_NOT: Use any greeting...` no longer authorizes `Dear ...`, and
  `MUST_NOT: Use emoji, hashtags, or all caps` no longer authorizes those
  surface artifacts.
- Focused local tests: `56 passed, 2 skipped`.
- Lint: `uvx ruff check` passed on source/env reward checks and reward tests.
- Bundled-env direct probes:
  - formal salutation/signature sample: reward `0.584`, deterministic `0.2`,
    penalties include `salutation`, `signoff`, and `ai_tell_phrase`;
  - formal/corporate email sample: reward `0.476`, deterministic `0.2`,
    penalties include `salutation` and `ai_tell_phrase`;
  - emoji/bad-romance/signoff sample from the user W&B trace: reward `0.315`,
    deterministic `0.0`, penalties include `emoji`, `signoff`, and
    `unsuitable_recommendation`;
  - explicit negated-decoration prompt with all caps, emoji, and hashtag:
    reward `0.460`, penalties include `all_caps`, `emoji`, and `hashtag`.
- Bundled env loads passed for `v02_smoke`, `v03`, and `mix_v2_p5050`.
- Verifier: approved for publish and debug smoke. Watchpoint: inspect hosted
  rollouts before any 50-step run because a perfect ridge can still let
  hard-format-capped samples reach about `0.70`.
- Prime env: `jayshah5696/humanize-rl-env@0.3.9`
- Wheel sha: `4566f583d77db40755b168cb7a5737addb3db5c8ae37fe55ccb2dafd046e449b`
- Published-env local Prime eval:
  `runs/prime_eval_smoke/qwen35_08b_p5050_env039`
  - Model: `Qwen/Qwen3.5-0.8B`
  - Examples: 3
  - Reward avg: `0.675`
  - Truncation: `0%`
  - Output tokens avg: `49.7`
  - Observed base-model miss: copied formal `Dear Ms. Young` and corporate
    phrasing; reward dropped to `0.387` with length `0.373` and format `0.923`.
- Status: local Prime eval smoke passed. Next serialized step is hosted
  one-step docs-debug smoke.

Hosted one-step smoke:

- Run: `kalrsxgjs04b48eoxs6k1cik`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/3dk0pvf5`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Result: completed.
- Eval step 0 reward: `0.6844`
- Eval step 1 reward: `0.6841`
- Train step 0 reward: `0.5932`
- Trainable: `16/16`
- Error: `0%`
- Truncation: `0%`
- Filters: `0`
- Decode length mean/max: `124.1` / `226.8`
- Usage: `26,965` total tokens; cost about `$0.0011`.
- Rollout audit:
  - reward mean/min/max: `0.5932` / `0.3325` / `0.8774`;
  - `Dear` openings in `7/16`, signoff/signature patterns in `9/16`, but those
    samples scored low (`0.33-0.56`) rather than appearing as top rewards;
  - no emoji, no shouty all-caps, no option-menu/wrapper samples observed;
  - highest rewards were all on `problem_id=618`, a creative/poem task, not the
    formal email failures.
- Verifier approved the 50-step `0.8B` smoke only.

Hosted 50-step smoke:

- Run: `d8gyr6h2340a4zkzfcrp5mkw`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/wynbfzzl`
- Config: `configs/prime/qwen35_08b_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Result: completed.
- Cost: about `$0.31`; total tokens `6,486,917`.
- Final checkpoint: `g960vwnq8mye413o46j1e6sq` at step `50` (upload pending
  when last checked).
- Eval deltas:
  - `mix_v2_p5050` p50: `0.6958 -> 0.9205`;
  - `v02_smoke` strict: `-0.2203 -> 0.1984`;
  - `v03` strict: `-0.2157 -> 0.2192`;
  - final eval errors `0%`;
  - final eval truncation: `0%` on v02 strict, `3.1%` on p50 and v03 strict.
- Train health:
  - trainable `128/128` throughout;
  - filters `0`;
  - train truncation mostly `0%`, with small blips up to `3.9%`;
  - final logged train decode mean/max at step 40: `180.6` / `446.5`.
- Step-30 rollout audit passed the previous stop gate:
  - no `Dear`, no signoff, no emoji, no wrapper/options, no repetition;
  - two all-caps samples, max reward `0.635`.
- Final eval qualitative blocker:
  - high p50 rewards exposed hallucinated, punchy outputs that missed required
    facts, e.g. strategic-planning/vendor-policy emails turned into invented
    threats or unrelated stories while scoring `0.982-0.986`;
  - local diagnostics showed `missing_entity` and `missing_required_fact`, but
    p50 ignored additive penalties and averaged faithfulness too softly, leaving
    deterministic at about `0.486` and ridge at `0.5`;
  - repeated 4096-cap p50 validation rows were low (`0.500-0.563`), so the
    remaining scale-up blocker is semantic grounding, not repetition reward.
- Decision: do not run 2B/4B from `0.3.9`.

### Env `0.3.10`

- Change: core semantic failures now zero the deterministic half of p50:
  `missing_required_fact`, `forbidden_fact`, `missing_number`,
  `missing_entity`, and `invented_detail`.
- Regression test added:
  `test_missing_required_fact_caps_p50_even_with_perfect_ridge`.
- Focused tests: `57 passed, 2 skipped`.
- Lint: `uvx ruff check` passed on source/env reward code and tests.
- Direct rescoring of the `0.3.9` high-reward hallucinations:
  - old rewards `0.982-0.986`;
  - new rewards `0.496-0.500`;
  - deterministic contribution `0.0`;
  - `semantic_faithfulness = 0.0`;
  - penalties include `missing_entity` and `missing_required_fact`.
- Status: publish `0.3.10`, rerun local Prime eval and one-step docs-debug
  smoke before any new 50-step or 2B run.

### Env `0.3.11`

- Reason: `0.3.10` local eval exposed noisy required facts copied from prompt
  scaffold (`Rewrite`, `Slack`, `Dear User`, `Focus\n\nDear`) and the
  `0.3.9` W&B traces still had some high-reward unsupported punchy outputs.
- Reward changes:
  - filter prompt/email scaffold out of active `required_facts` and entity
    preservation;
  - add semantic diagnostics for invented numbers, invented temporal details,
    unsupported hostile/contradictory phrases, and low source overlap on long
    source rewrites;
  - make emoji, hashtags, all-caps, em dash, canned inline closings, formal
    salutations, and signoffs visible diagnostics;
  - include the new semantic diagnostics in the p50 deterministic-half cap.
- Tests: reward/env slice `63 passed, 2 skipped`; lint passed on touched
  source/env reward files and tests.
- Trace rescore against W&B run `wynbfzzl`, eval table
  `runs/wandb_artifacts/wynbfzzl_evalsamples_v8/7-1781852974317.eval/samples.table.json`:
  - unique `eval_mix_v2_p5050` samples: `69`;
  - old high hallucinations at `0.982-0.986` now rescore near `0.496-0.500`;
  - samples with new reward `>=0.9`: `4`, all inspected as plausible direct
    Slack/workplace rewrites;
  - semantic failures: `63/69`;
  - hard-format failures: `4/69`.
- Local env0310 eval rows rescored with current code:
  - `rl_v01_000002`: `0.307 -> 0.807` after dropping `Dear User` scaffold;
  - `rl_v01_000020`: `0.361 -> 0.861` after dropping `Rewrite`/`Slack`
    scaffold;
  - formal `Dear Ms. Young` row remains low at `0.187`.
- Status: published `0.3.11`; local Prime eval smoke passed technically, but
  row `rl_v01_000002` still scored `0.807` for the formal phrase
  `We are writing to inform you...`. Do not launch hosted training from
  `0.3.11`.

### Env `0.3.12`

- Reason: `0.3.11` kept `ai_tell_phrase` as a diagnostic/risk penalty only;
  p50 ignores additive penalties, so a high ridge score could still keep
  formal AI-tell phrasing too high.
- Change: include `ai_tell_phrase` in `surface_naturalness`, so AI-tell
  failures cap the deterministic half of p50 like emoji, all-caps, hashtags,
  and em dashes.
- Regression test added:
  `test_ai_tell_phrase_caps_p50_surface_naturalness`.
- Tests: reward/env slice `65 passed, 2 skipped`; lint passed on touched
  source/env reward files and tests.
- Published `0.3.12` and ran local Prime eval smoke:
  - results:
    `runs/prime_eval_smoke/qwen35_08b_p5050_env0312/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/91e4baee/results.jsonl`;
  - average reward `0.518`, rows `0.507`, `0.187`, `0.861`;
  - `We are writing to inform...` now scores `0.507` and fails
    `ai_tell_phrase`;
  - formal `Dear Ms. Young...` remains low, but diagnostic rescore exposed
    false missing-entity matches from subject/discourse fragments.
- Status: do not launch training from `0.3.12`; publish `0.3.13` with the
  false-entity cleanup first.

### Env `0.3.13`

- Reason: local `0.3.12` diagnostics still treated subject-title/discourse
  fragments (`Compliance Review`, `Firstly`, `Secondly`, `Understanding`) as
  required entities, adding reward noise.
- Change: expand entity stop-list for title/discourse fragments in both source
  and bundled Prime env.
- Regression: `test_email_subject_and_salutation_scaffold_are_ignored` now
  covers `Compliance Review`, `Review`, `Key Areas`, `Firstly`, `Secondly`,
  and `Understanding`.
- Tests: reward/env slice `65 passed, 2 skipped`; ruff passed on touched reward
  files.
- Diagnostic rescore of the saved local eval rows with current code:
  - `rl_v01_000002`: `0.507`, fails `ai_tell_phrase`;
  - `rl_v01_000016`: `0.387`, fails only `salutation` and `too_long`;
  - `rl_v01_000020`: `0.861`, no failed diagnostics.
- Published `0.3.13`:
  - wheel: `humanize_rl_env-0.3.13-py3-none-any.whl`;
  - SHA256:
    `5defcc96468ce76fdcfa252793b9875952d1f282b53cd46dd2890f6a06abe76f`.
- Local Prime eval smoke against published `0.3.13`:
  - command used `Qwen/Qwen3.5-0.8B`, `num_examples=3`,
    `rollouts_per_example=1`, `max_tokens=4096`, `temperature=0.0`;
  - result path:
    `runs/prime_eval_smoke/qwen35_08b_p5050_env0313/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/2cf64932/results.jsonl`;
  - average reward `0.594`, rows `0.507`, `0.387`, `0.888`;
  - no truncation; average output tokens `49.667`;
  - audited diagnostics:
    - `rl_v01_000002`: fails `ai_tell_phrase` for `we are writing to`;
    - `rl_v01_000016`: fails `salutation` and `too_long`;
    - `rl_v01_000020`: no failed diagnostics.
- Verifier approved hosted one-step docs-debug `r8` only. No 50-step approval.

Hosted one-step docs-debug:

- Run: `wckm7qbo1r8oc8ipo3b77xum`
- Dashboard:
  `https://app.primeintellect.ai/dashboard/training/wckm7qbo1r8oc8ipo3b77xum`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Env: `jayshah5696/humanize-rl-env@0.3.13`
- Env name: `train_mix_v2_p5050_docs_debug_r8`
- Result: completed one step.
- W&B project: `jayshah5696/humanize-rl`
- Sampling cap: `4096`
- Batch size: `16`
- Rollouts per example: `4`
- Raw rollout artifact:
  `runs/prime_training_smoke/wckm7qbo1r8oc8ipo3b77xum/rollouts_step0.json`

Hosted rollout audit:

- Rows: `16`
- Prime reward mean/min/max: `0.645831` / `0.360613` / `0.881603`
- High-reward samples `>=0.75`: `7/16`
- Problem ids: `404`, `520`, `618`, `638`
- Blockers:
  - `problem_id=404`, task `rl_v03_000009`: sample `0.802` leaked
    instruction language (`your draft`) and ended with a polite collaboration
    closing; sample `0.773` described the source instead of directly
    compressing it.
  - `problem_id=520`, task `rl_v03_000268`: samples around `0.716-0.751`
    missed exact constraints such as contraction count, required phrase,
    forbidden phrase, paragraph/sentence requirements, and length.
  - `problem_id=618`, task `rl_v03_000439`: poem samples scored
    `0.862-0.882` while failing `too_short`.
  - `problem_id=638`, task `rl_v03_000480`: bad letter shells stayed low
    (`0.360-0.476`), so the existing salutation/signoff/placeholder/wrapper
    caps were working for that class.
- Decision: do not launch a 50-step run from `0.3.13`.

### Env `0.3.14`

- Reason: `0.3.13` hosted `r8` showed that local eval was not enough. The
  reward still overvalued instruction leakage, polite closing leakage, exact
  constraint misses, and too-short creative outputs.
- Reward changes:
  - add `instruction_leak` diagnostic for outputs that talk about `source
    message`, `your draft`, `the draft`, or output requirements instead of
    producing the answer;
  - treat inline feedback/collaboration thank-you closings as signoffs;
  - include `instruction_leak`, `placeholder_required`, `missing_contraction`,
    `paragraph_count`, and `sentence_window` in hard-format failures;
  - include `missing_must_include_phrase`, `forbidden_phrase`, and
    `placeholder_required` in semantic failures;
  - make any explicit length diagnostic (`too_long`, `too_short`,
    `sentence_count`) trigger the deterministic length cap.
- Regression tests added for:
  - inline thank-you closing as signoff;
  - instruction leak on direct rewrite/compression tasks;
  - instruction-like language still allowed for `multi_constraint_compose`;
  - instruction leak capping the p50 deterministic half;
  - phrase-constraint failure zeroing the p50 deterministic half;
  - mild length-window failure capping the p50 deterministic half.
- Verification:
  - reward/env tests: `71 passed, 2 skipped`;
  - ruff: passed on touched source/env reward files and reward tests.

Local rescore of the exact saved `r8` rollouts using current source:

- Rows: `16`
- Old Prime reward mean/min/max:
  `0.645831` / `0.360613` / `0.881603`
- Current reward mean/min/max:
  `0.408088` / `0.266424` / `0.499395`
- Average delta: `-0.237744`
- Current high-reward samples `>=0.75`: `0/16`
- Previously high samples `>=0.75` now below `0.75`: `7/7`
- Diagnostic counts:
  - `sentence_window`: `8`
  - `signoff`: `7`
  - `too_short`: `7`
  - `missing_contraction`: `4`
  - `missing_must_include_phrase`: `3`
  - `placeholder_disallowed`: `3`
  - `salutation`: `3`
  - `ai_tell_phrase`: `2`
  - `instruction_leak`: `2`
  - singletons: `wrapper_phrase`, `subject_line`, `wrong_format_markdown`,
    `wrong_format_bullets`, `too_long`, `forbidden_phrase`,
    `paragraph_count`, `wrong_format_heading`
- Problem-level old/new rewards:
  - `404`: `[0.802, 0.773, 0.476, 0.490] -> [0.311, 0.473, 0.476, 0.290]`
  - `520`: `[0.716, 0.751, 0.721, 0.422] -> [0.266, 0.292, 0.341, 0.422]`
  - `618`: `[0.875, 0.863, 0.869, 0.882] -> [0.494, 0.482, 0.488, 0.499]`
  - `638`: `[0.470, 0.476, 0.388, 0.361] -> [0.470, 0.476, 0.388, 0.361]`

Interpretation:

- The patch fixes the observed high-reward `r8` failure class.
- The already-low bad letter-shell samples did not change, so the patch is not
  broadly suppressing everything.
- Published `0.3.14`:
  - wheel: `humanize_rl_env-0.3.14-py3-none-any.whl`;
  - SHA256:
    `66b332bbfa615d8c0fa78ce3041269da9f684d99ff0d7c6682ab6864f744cad9`;
  - metadata:
    `environments/humanize_rl_env/.prime/.env-metadata.json`.
- Local Prime eval smoke against published `0.3.14`:
  - command used `Qwen/Qwen3.5-0.8B`, `num_examples=3`,
    `rollouts_per_example=1`, `max_tokens=4096`, `temperature=0.0`;
  - result path:
    `runs/prime_eval_smoke/qwen35_08b_p5050_env0314/evals/humanize-rl-env--Qwen--Qwen3.5-0.8B/c2e718a9/results.jsonl`;
  - average reward `0.551`, rows `0.507`, `0.287`, `0.861`;
  - no errors; no truncation; average output tokens `49.667`;
  - audited diagnostics:
    - `rl_v01_000002`: fails `ai_tell_phrase` for `we are writing to`;
    - `rl_v01_000016`: fails `salutation` and `too_long`, now lower because
      length diagnostics trigger the deterministic length cap;
    - `rl_v01_000020`: no failed diagnostics.
- Live Prime model list refreshed with CLI:
  - `Qwen/Qwen3.5-0.8B`, `2B`, `4B`, `9B`, `35B-A3B`,
    `Qwen/Qwen3.6-35B-A3B`: available;
  - `meta-llama/Llama-3.2-1B-Instruct` and `3B-Instruct`: available;
  - `poolside/Laguna-XS.2` and `sprints/Llama-3.2-1B-Instruct`: listed as
    free, but previous Sprints custom-env smoke rejected this env class.
- Next serialized gate: ask verifier for hosted one-step docs-debug `r9`, then
  launch only that one-step run if approved.

Hosted one-step docs-debug `r9`:

- Run: `l4zww8v68f4h7rigc48rk9e3`
- W&B: `https://wandb.ai/jayshah5696/humanize-rl/runs/d4le5ba5`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model: `Qwen/Qwen3.5-0.8B`
- Env: `jayshah5696/humanize-rl-env@0.3.14`
- Result: completed one optimizer step and checkpoint `liktxup4mab6c9hjqb71dmet`
  is ready, but the gate failed.
- Usage: `36.50K` tokens, about `$0.0017`.
- Train step:
  - logged rollout reward mean/min/max: `0.440` / `0.183` / `0.863`;
  - `2/16` samples scored `>=0.75`, both with no failed diagnostics;
  - direct good samples stayed high, e.g. sprint timeline rewrites at `0.863`
    and `0.789`;
  - old bad classes stayed low, e.g. letter/wrapper/markdown outputs around
    `0.183-0.487`.
- Rollout audit:
  - high-current `>=0.75`: `2/16`;
  - high-current `>=0.70` with failed diagnostics: `0`;
  - diagnostic counts: `missing_entity=7`, `signoff=5`,
    `sentence_window=5`, `missing_required_fact=4`,
    `placeholder_disallowed=4`, `too_long=3`, `ai_tell_phrase=2`,
    `salutation=2`, and one each of `option_menu`, `wrong_format_markdown`,
    `wrong_format_bullets`, `forbidden_phrase`, `invented_temporal_detail`.
- Infra/config blockers:
  - train error `29.3%` with repeated model/API failures:
    `BadRequestError: Out of range float values are not JSON compliant: nan`;
  - hosted eval step 0 truncation `100%`, completion length `4096/4096/4096`;
  - hosted eval step 1 truncation `33.3%`, one completion at `4096`;
  - train sampling did not truncate, so the truncation problem is isolated to
    hosted eval sampling.
- Decision: do not launch the 50-step smoke from `r9`.
- Config-only fix for `r10`:
  - keep train sampling temperature `0.7`;
  - keep max generation cap `4096`;
  - change hosted `[eval.sampling] temperature` from `0.0` to `0.2` across
    Prime configs because greedy hosted Qwen eval runs to the token cap.

Hosted one-step docs-debug `r10`:

- Run: `h9r8y6sj9synobs19kmu9zc2`
- W&B run name/id: `prime-qwen35-08b-p5050-docs-debug-r10` / `3vjqmco2`
- Config: `configs/prime/qwen35_08b_docs_debug.toml`
- Model/env: `Qwen/Qwen3.5-0.8B` /
  `jayshah5696/humanize-rl-env@0.3.14`
- Status: `COMPLETED`
- Usage: `54,449` tokens, `$0.0026`
- Checkpoint: `eizq3dbj2m99bjt6tkgfp8gt`, step `1`, reported `UPLOADING`
  at check time.
- Train metrics:
  - sample audit reward mean/min/max:
    `0.4393588446` / `0.2504880428` / `0.7311608437`;
  - aggregate per-problem reward mean/min/max:
    `0.4393588446` / `0.3418707205` / `0.4895065989`;
  - train truncation `0%`;
  - train error/cancel rate `0%`;
  - decode length mean/max `172.75` / `254.25`;
  - repetition/gibberish/zero-advantage filters `0`.
- Hosted eval still failed:
  - step `0` truncation `100%`, completion length mean/min/max
    `4096/4096/4096`;
  - step `1` truncation `100%`, completion length mean/min/max
    `4096/4096/4096`;
  - temperature `0.2` did not fix hosted Qwen eval runaway.
- Saved artifacts:
  - rollouts:
    `runs/prime_training_smoke/h9r8y6sj9synobs19kmu9zc2/rollouts_step0.json`;
  - local audit:
    `runs/prime_training_smoke/h9r8y6sj9synobs19kmu9zc2/audit_step0.json`;
  - audit helper: `scripts/eval/audit_prime_rollouts.py`.
- Local audit:
  - matched rows `16/16`;
  - recomputed reward mean/min/max:
    `0.4393588446` / `0.2504880428` / `0.7311608437`;
  - high reward `>=0.75`: `0`;
  - high reward with failed diagnostics: `0`;
  - high reward with emoji/all-caps/option-or-wrapper: `0/0/0`;
  - failed counts: `sentence_window=10`, `signoff=6`, `too_short=6`,
    `missing_contraction=4`, `ai_tell_phrase=3`, `paragraph_count=3`,
    `salutation=3`, `forbidden_phrase=2`, `placeholder_disallowed=2`,
    `missing_must_include_phrase=1`.
- Verification:
  - focused reward/env/script tests: `72 passed, 2 skipped`;
  - ruff passed on the new audit helper and test;
  - train-only 50-step config parses with `max_tokens=4096`,
    `enable_thinking=false`, env `0.3.14`, no `[eval]`, no `[val]`, no `rtk`,
    and no committed W&B token.
- Tooling caveat:
  - project `uv run` tried to build `fasttext-wheel` and failed against local
    Xcode SDK; reran the audit with `uv run --no-project` and explicit
    lightweight deps. This is local tooling only, not a Prime training blocker.
- Decision:
  - proceed to `configs/prime/qwen35_08b_debug.toml` as a 50-step train-only
    0.8B smoke;
  - do not launch 2B until this 0.8B run is audited and checkpoint evaluation
    clears reward, truncation, wrapper/options, emoji/all-caps, length-collapse,
    and qualitative direct-answer gates.
