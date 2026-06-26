# Prime p50 Sweep Runbook

## Chronological State

1. Data generation and judging produced v01/v02/v03 RL tasks.
2. Modal difficulty rollouts produced `humanize_tasks_rl_mix_v2.jsonl` with 998 tasks and 5,988 saved rollouts.
3. Strict/softened filtering was too restrictive/noisy for the next RL run.
4. The current task set is rebuilt as `humanize_tasks_rl_mix_v2_p5050_filtered.jsonl` using:
   `p50_50_no_penalty = 0.50 * ridge_rubric + 0.50 * deterministic`.
5. Result: 972 useful tasks, split 777 train / 92 validation / 103 test.

## Implemented Defaults

- Env package version: `0.3.14`
- Default task set: `mix_v2_p5050_filtered`
- Training reward: `p50_50_no_penalty`
- Strict penalties remain diagnostics and eval gates.
- Ridge scorer is bundled at `humanize_rl_env/ridge_state.pkl`; p50 mode raises if it cannot load.
- Prime CLI `0.6.14` rejects `pre_batch_filters` / `post_batch_filters` even
  though current Prime docs list them. Keep TOMLs runnable under the installed
  CLI; `0.3.14` handles repetition/runaway, the observed uplifting-romance
  recommendation leak, fake-human surface failures such as emojis/hashtags/
  shouty all-caps, and formal salutation/signature leaks. Negative clauses
  such as `MUST_NOT: Use emoji...` or `MUST_NOT: Use any greeting...` override
  keyword mentions and remain penalized. It also filters prompt/email scaffold
  out of required facts and zeros the deterministic half of p50 on semantic
  failures, so high-ridge hallucinations, invented numbers/times, hostile
  unsupported phrasing, and low-overlap rewrites cannot become top-reward
  samples. It also caps AI-tell phrases and ignores subject/discourse false
  entities such as `Compliance Review`, `Firstly`, `Secondly`, and
  `Understanding`. The current `0.3.14` candidate also caps instruction
  leakage, polite closing leakage, exact phrase/structure misses, and
  too-short creative outputs found in hosted `r8`.

## Stopped Smokes

- Run: `beixwu41osp530um7bfabn8k` (`0.3.5`)
- Status: stopped at step 21; cost: `$0.25`.
- Failure: reward rose while truncation/repetition exploded. Step 20 p50 reward
  was about `0.875` with `56%` truncation and `50%` repetition, so repeated junk
  was still being rewarded.
- Decision: do not start the 2B run until a tiny 0.8B smoke passes repetition,
  truncation, p50 validation, and strict eval gates.
- Fix: p50 remains `0.50 * ridge_rubric + 0.50 * deterministic`, but repetition
  and target-word length now live inside deterministic scoring.
- Run: `xi7nu5o2yu8761wuvxv0wjom` (`0.3.6`)
- Status: completed one step; technical metrics were clean, but verifier blocked
  scale-up because sampled rollouts for `rl_v03_000392` recommended unsuitable
  films for an uplifting romance task while still receiving about `0.68-0.72`.
- Fix: `0.3.7` keeps the same p50 blend, but adds deterministic
  `recommendation_suitability`; known unsuitable recommendations zero the
  deterministic half even with a perfect ridge score. Re-enable Prime TOML
  rollout filters only after the local CLI schema accepts them.
- Run: `z3kf1g2nesrhc2xunybpi4mf` (`0.3.7`)
- Status: completed one step; technical metrics were clean, but user audit found
  a W&B rollout with emojis, fake warmth, unsuitable movie recommendations, and
  signoff still showing a high reward.
- Fix: `0.3.8` adds task-aware `emoji`, `hashtag`, and `all_caps` diagnostics,
  expands signoff detection, and broadens the bad-title guard for the observed
  romance traces.
- Run: `jfwfr98t1ncdeiab17w5rnss` (`0.3.8`)
- Status: completed one step; technical metrics were clean, but highest-reward
  samples still used formal letter openings and inline signatures such as
  `Dear William Brown, ... Best regards, Patricia Adams, Engineer...`.
- Fix: `0.3.9` adds formal salutation detection, stronger inline/multiline
  signature detection, includes `salutation` in the hard-format cap, and fixes
  negated style-permission parsing for greeting/emoji/hashtag/all-caps prompts.
- Run: `d8gyr6h2340a4zkzfcrp5mkw` (`0.3.9`)
- Status: completed 50-step smoke; technical metrics and strict eval deltas
  improved, but final p50 validation exposed high-reward hallucinations that
  missed required facts because p50 ignored penalties and deterministic
  faithfulness was averaged too softly.
- Fix: `0.3.13` filters scaffold required facts and zeros the deterministic
  half when semantic checks fail (`missing_required_fact`, `forbidden_fact`,
  `missing_number`, `missing_entity`, `invented_detail`, invented number/time,
  unsupported negation, low source overlap). Re-run the one-step and 50-step
  smokes before any 2B/4B run.
- Run: `wckm7qbo1r8oc8ipo3b77xum` (`0.3.13`)
- Status: completed one hosted docs-debug step, but `7/16` rollouts scored
  `>=0.75` despite instruction leakage, polite closing leakage, exact
  constraint misses, and too-short creative outputs.
- Fix: `0.3.14` adds `instruction_leak`, stronger inline closing detection,
  exact-constraint hard caps, and length-diagnostic caps. Re-run local eval and
  hosted docs-debug `r9` before any 50-step smoke.

## Commands

```bash
prime --plain login
prime --plain train models --output json
prime --plain env push --path environments/humanize_rl_env
```

Smoke preview before training:

```bash
PYTHONPATH=environments/humanize_rl_env uv run --no-project --with pydantic --with scikit-learn --with numpy python - <<'PY'
import json
from humanize_rl_env import preview_dataset_row
row = preview_dataset_row(split="validation", task_set="v03")
info = json.loads(row["info"])
print(sorted(row))
print(info["task"]["id"])
PY
```

Local Prime eval smoke:

```bash
prime --plain eval run jayshah5696/humanize-rl-env \
  --env-args '{"split":"validation","task_set":"mix_v2_p5050","reward_mode":"p50_50_no_penalty"}' \
  --model Qwen/Qwen3.5-0.8B \
  --num-examples 3 \
  --rollouts-per-example 1 \
  --max-tokens 4096 \
  --temperature 0.0 \
  --skip-upload \
  --disable-env-server \
  --disable-tui \
  --max-retries 0 \
  --save-results \
  --output-dir runs/prime_eval_smoke/qwen35_08b_p5050_env0314
```

Do not pass `enable_thinking` through local `prime eval run --sampling-args`;
the local OpenAI-compatible client rejects that keyword. Keep
`enable_thinking = false` in Prime TOML configs, where Hosted Training handles it
as a first-class sampling field. Hosted Training eval uses `temperature = 0.2`
after docs-debug `r9` showed Qwen greedy eval at `temperature = 0.0` can run to
the full `4096` cap even though local eval stops normally.

Run order:

```bash
prime --plain train configs/prime/qwen35_08b_docs_debug.toml
prime --plain train configs/prime/qwen35_08b_debug.toml
prime --plain train configs/prime/qwen35_2b_p5050.toml
prime --plain train configs/prime/qwen35_4b_p5050.toml
prime --plain train configs/prime/qwen36_35b_a3b_p5050.toml
```

The docs-debug training smoke must use multiple rollouts per example. Prime
enforces post-batch `zero_advantage` filtering by default, so
`rollouts_per_example = 1` cannot produce a within-prompt advantage estimate and
can drop every train rollout. Keep this smoke tiny with `max_steps = 1`,
`batch_size = 16`, and `rollouts_per_example = 4`; do not use
`[buffer] skip_verification = true` for this RL smoke.

Current Prime docs show `[[pre_batch_filters]]` and `[[post_batch_filters]]`,
but CLI `0.6.14` rejects those sections as extra inputs. Do not add them back
until `prime train` accepts the config locally.

## Selection Gates

- p50 validation on `mix_v2_p5050` improves over base.
- strict validation improves on `v02_smoke` and `v03`.
- No family mean drops below `-0.05`.
- Option-menu/wrapper penalties do not regress.
- No length collapse.
- Blind A/B beats SFT-only and `gemma4-e2b-humanize-rl-candidate-v1`.

## Follow-On Tracks

- Continue Gemma on Modal only if the Prime sweep underperforms candidate-v1.
- Liquid comes after Prime with a generic Modal TRL loader:
  `LiquidAI/LFM2.5-1.2B-Instruct`, then `LiquidAI/LFM2-2.6B`.
