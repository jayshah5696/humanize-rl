# V03 Seed + Benchmark Spec

**Date:** 2026-04-18  
**Status:** Proposed plan for the next benchmark/data rebuild  
**Scope:** Human seed sourcing, benchmark construction, SFT pair acceptance, and prompt revisions

## Executive Summary

The next data rebuild should optimize for **matched, domain-balanced, corpus-backed human writing** instead of adding more hand-curated tech prose.

Current issues to correct:

- current human seeds are too short (median ~49 words vs target 100-400+)
- current seed pool is over-indexed on tech/blog voice
- current main benchmark mixes matched triples with out-of-distribution long AI explainers
- some accepted SFT pairs improve the score by **changing the discourse role** rather than preserving the original writing act
- current AIify prompt adds **all 8 patterns at once**, making the task too easy and synthetic

## Decisions

1. **Freeze `test_set_v02` as exploratory**, not the final headline benchmark.
2. Build a new **v03 matched benchmark** from a **200-seed human pool**.
3. Keep the extra long-form ChatGPT/Claude AI set as an **OOD stress split only**; do not merge it into the main matched benchmark.
4. Source the core human pool primarily from **corpus-backed datasets**, not hand writing.
5. Tighten pair quality gates to punish:
   - discourse-role drift
   - length drift
   - entity/number drift
   - over-correction where humanized text scores far above the original human
6. Make AIify **domain-aware** and **subset-based** instead of injecting all 8 patterns every time.
7. Make Humanize preserve **domain + discourse role + stance + facts**, not just “sound natural”.

---

## 1. Dataset Layout

### 1.1 Core matched benchmark: `benchmark_v03_core`

Use for headline reporting and model iteration.

- **200 human seeds**
- **200 AIified texts**
- **200 humanized texts**
- **600 total benchmark rows**

Every row in the AI and humanized class must trace back to one human seed.

### 1.2 OOD AI benchmark: `benchmark_v03_ood_ai`

Use for generalization checks only.

- existing external long-form AI explainers
- kept separate from the matched core benchmark
- reported independently as “OOD AI prose generalization”

### 1.3 Diagnostic hard-case set: `benchmark_v03_diagnostics`

Small curated set for failure analysis.

Use this to stress:

- em-dash-heavy real human prose
- formal human writing with naturally low contraction rate
- terse emails
- forum questions
- academic paragraphs with appropriate hedging
- high-specificity technical prose
- humanized outputs that still feel off despite high scores

Recommended size: **30-50 rows**.

---

## 2. Human Seed Targets

### 2.1 Accepted seed target: 200

This directly matches the intended domain mix.

| Domain bucket | % | Accepted seeds |
|---|---:|---:|
| email / professional | 25% | 50 |
| instruction / technical | 25% | 50 |
| blog / opinion / essay / journalism | 25% | 50 |
| academic / formal | 15% | 30 |
| creative / expressive | 10% | 20 |
| **Total** | **100%** | **200** |

### 2.2 Raw candidate target: 450-550

Assume roughly **35-45% acceptance** after quality filtering.

---

## 3. Source-to-Domain Mapping

Only use sources that satisfy the repo’s quality bar: verifiably human or strongly human-curated.

| Domain bucket | Primary sources | Candidate target | Expected accepted |
|---|---|---:|---:|
| email / professional | Enron, RAID human | 120 | 50 |
| instruction / technical | GoodWiki, FineWeb-Edu, RAID human | 120 | 50 |
| blog / opinion / essay / journalism | RAID human, CNN/DailyMail | 120 | 50 |
| academic / formal | peS2o | 90 | 30 |
| creative / expressive | WritingPrompts, PG-19 | 60 | 20 |
| **Total** |  | **510** | **200** |

### 3.1 Recommended source mix by accepted seed count

#### Email / professional (50)
- Enron: **40**
- RAID human: **10**

#### Instruction / technical (50)
- GoodWiki: **20**
- FineWeb-Edu: **20**
- RAID human: **10**

#### Blog / opinion / essay / journalism (50)
- RAID human: **30**
- CNN/DailyMail: **20**

#### Academic / formal (30)
- peS2o: **30**

#### Creative / expressive (20)
- WritingPrompts: **15**
- PG-19: **5**

### 3.2 Use of the current curated seeds

Current `seeds/human_seeds_v01.jsonl` should be treated as:

- **diagnostic / seed inspiration only**
- not the backbone of the v03 matched benchmark

Reason: the current set is short, stylized, and heavily skewed toward tech/blog voice.

---

## 4. Length and Form Constraints

The current global rule of 100-600 words is directionally right, but v03 should be more domain-specific.

| Domain bucket | Preferred length | Hard min | Hard max |
|---|---:|---:|---:|
| email / professional | 90-180 | 70 | 220 |
| instruction / technical | 120-240 | 100 | 300 |
| blog / opinion / essay / journalism | 120-240 | 100 | 300 |
| academic / formal | 120-260 | 100 | 320 |
| creative / expressive | 100-220 | 90 | 260 |

### Why

This avoids three current problems:

1. tiny samples that make sentence rhythm noisy
2. giant AI explainers that are easy to catch for the wrong reasons
3. domain mismatch where, for example, a terse email is compared to a sprawling explainer

---

## 5. Seed Acceptance Rubric

A candidate human seed is accepted only if it passes the hard filters and scores well on the soft filters.

### 5.1 Hard filters

Reject if any of the following are true:

- not clearly human-sourced / human-curated
- non-English
- duplicate or near-duplicate
- below hard minimum length or above hard maximum
- list-heavy unless the genre naturally requires a list
- FAQ / SEO / marketing sludge
- instructional “overview of X” boilerplate with no specifics
- text fragment depends on missing context
- obvious AI artifacts already present

### 5.2 Soft quality signals

Prefer candidates with most of the following:

- **2+ concrete anchors**
  - names, dates, commands, metrics, versions, citations, quoted phrases
- **native register**
  - genuinely sounds like its domain
- **local asymmetry**
  - not a perfect 3-part template
- **speaker position**
  - some identifiable stance, even if formal
- **preservable invariants**
  - there are facts/style commitments a rewrite must preserve
- **non-generic specificity**
  - concrete enough that generic rewriting would visibly damage it

### 5.3 Scored acceptance checklist

Use a lightweight 0/1 screening checklist and keep only seeds with **5/6 or better**:

1. Contains at least 2 concrete anchors
2. Domain/register is obvious within 2 sentences
3. Text is self-contained
4. Not already templated or listified unless genre-native
5. Has a clear discourse role
6. A generic rewrite would obviously lose something important

---

## 6. Discourse Role Taxonomy

Every accepted seed should be tagged with a discourse role. This is critical because current humanized outputs sometimes drift into a different role.

### 6.1 Allowed roles

- `status_update`
- `request`
- `decision_rationale`
- `troubleshooting`
- `instructional_explanation`
- `argument_opinion`
- `anecdote`
- `reflection`
- `methods`
- `literature_review`
- `discussion_limitations`
- `reported_summary`
- `scene`
- `narrative_reflection`

### 6.2 Non-negotiable rule

**AIify and Humanize must preserve discourse role exactly.**

Examples:

- question stays question
- status update stays status update
- anecdote stays anecdote
- methods paragraph stays methods paragraph
- short internal note does not become a blog paragraph

---

## 7. What a Good Seed Looks Like

### 7.1 Email / professional

**Strong archetype**
- real blocker
- named person/team
- next step
- one practical judgment call

**Bad archetype**
- polished PM boilerplate with no concrete stakes

### 7.2 Instruction / technical

**Strong archetype**
- exact symptom
- exact tool/config/file/command
- one caveat or gotcha
- prose, not already bullet sludge

**Bad archetype**
- generic explainer like “what is caching” or “benefits of Docker”

### 7.3 Blog / opinion / essay / journalism

**Strong archetype**
- strong claim
- one anecdote or reported example
- one concrete detail or number
- non-generic voice

**Bad archetype**
- slogan or tweet-sized hot take

### 7.4 Academic / formal

**Strong archetype**
- methods paragraph
- discussion paragraph
- limitations paragraph
- literature review paragraph with named citations

**Bad archetype**
- only polished abstracts

### 7.5 Creative / expressive

**Strong archetype**
- scene or reflective paragraph
- sensory detail
- uneven rhythm
- implied narrator

**Bad archetype**
- cliché purple prose with no concrete scene

---

## 8. AIify Generation Spec

The current AIify approach (“add all 8 patterns”) is too synthetic.

### 8.1 New AIify policy

For each accepted human seed:

- generate **2 AIified candidates**
- choose the one that best matches the target score band and semantic preservation rules

### 8.2 Pattern injection policy

Do **not** add all 8 patterns every time.

Instead sample:

- **intensity**: `subtle`, `medium`, `heavy`
- **target dimensions**: choose **2-4** dimensions

### 8.3 Intensity mix across the 200-seed pool

| Intensity | Share | Seeds |
|---|---:|---:|
| subtle | 35% | 70 |
| medium | 45% | 90 |
| heavy | 20% | 40 |

### 8.4 Allowed AIify dimensions by domain

#### Email / professional
Prefer:
- opener_pattern
- hedging_density
- closing_pattern
- transition_overuse
- mild formality drift

Avoid by default:
- forced bullet conversion

#### Instruction / technical
Prefer:
- structural_symmetry
- padding_density
- transition_overuse
- generic specificity loss
- occasional listification

#### Blog / opinion / essay / journalism
Prefer:
- opener_pattern
- hedging_density
- rhetorical_sophistication
- personality flattening
- uniform sentence rhythm

#### Academic / formal
Prefer:
- genericity / specificity loss
- rhetorical inflation
- padding_density
- copula avoidance
- structural symmetry

Avoid:
- cheery customer-support opener unless the seed itself is Q&A-like

#### Creative / expressive
Prefer:
- rhythm flattening
- generic abstraction
- over-explanation
- formality shift
- personality removal

Avoid:
- bullet points except in intentionally adversarial cases

### 8.5 Target AI score band

Accept an AIified candidate only if:

- `combined_score <= 0.45`
- preferred band: **0.20-0.40**
- semantic content preserved
- discourse role preserved

If both candidates fail, regenerate.

---

## 9. Humanize Generation Spec

The current humanize instruction is too loose and allows role drift.

### 9.1 New Humanize policy

For each accepted AIified input:

- generate **2 humanized candidates**
- keep the candidate that best restores the original without overshooting it unnaturally

### 9.2 Humanize target behavior

A good humanizer should:

- remove AI markers
- preserve meaning
- preserve discourse role
- preserve stance and speaker position
- preserve numbers/entities
- restore local texture, not replace the text with a different genre

### 9.3 Target humanized score band

Accept a humanized candidate only if:

- `combined_score >= 0.75`
- `humanized_score - aiified_score >= 0.30`
- `humanized_score <= original_score + 0.05` preferred
- if `humanized_score > original_score + 0.08`, flag for manual review

This is specifically meant to stop “humanized beats human by becoming a generic high-scoring explainer”.

---

## 10. Pair Acceptance Rules for SFT

A triple is eligible for SFT export only if all checks pass.

### 10.1 Required score thresholds

- `aiified_score <= 0.45`
- `humanized_score >= 0.75`
- `humanize_delta >= 0.30`
- `aiify_delta >= 0.25`
- at least **2 humanness dimensions improve** from AIified → humanized

### 10.2 Length preservation

#### AIify vs original
- preferred ratio: **0.90-1.20**
- hard max: **1.25**

#### Humanized vs AIified
- preferred ratio: **0.85-1.15**
- hard max: **1.20**

#### Humanized vs original
- preferred ratio: **0.85-1.15**
- hard max: **1.20**

### 10.3 Preservation checks

Must preserve exactly or near-exactly:

- named entities
- numbers / dates / metrics / versions
- tense
- person (`I` vs `we` vs `you`)
- polarity / stance
- discourse role
- domain tag

### 10.4 Suspicion flags

Flag for review or reject if any of these occur:

- `recovery_ratio > 1.25`
- anecdote becomes advice
- question becomes essay
- email becomes blog paragraph
- humanized output adds broader claims not in the source
- humanized output sounds more generic than the original human
- humanized output introduces extra motivational emphasis or fake certainty

### 10.5 Manual review priority queue

Manual review should focus first on:

- highest recovery ratios
- biggest length expansions
- biggest score gaps where humanized > original
- academic and technical samples with zero contractions
- any sample with em-dashes in the human original

---

## 11. Prompt Revisions

## 11.1 Proposed AIify prompt v02

```text
You are rewriting human-written text to sound more like typical LLM output.

Your job is NOT to change the task, genre, or meaning.
Preserve exactly:
- discourse role
- named entities
- numbers, dates, versions, metrics
- stance and conclusion
- person and tense

Target domain: {domain}
Discourse role: {discourse_role}
Intensity: {intensity}
Patterns to add: {target_dimensions}

Rules:
- Keep the same information
- Keep roughly the same length (within 20%)
- Do not add facts, examples, or advice
- Do not convert to bullets unless listification is one of the requested patterns
- Make the text feel AI-written in the requested ways, but keep it plausible for this domain

Text:
{text}
```

### 11.2 Proposed Humanize prompt v02

```text
Rewrite the following text so it reads like natural human writing.

Remove only the AI patterns that are present.
Do NOT change the writing act.

Preserve exactly:
- domain
- discourse role
- named entities
- numbers, dates, versions, metrics
- person and tense
- stance and conclusion

Detected AI patterns:
{detected_dimensions_with_evidence}

Important constraints:
- If the source is a question, keep it a question
- If the source is a status update, keep it a status update
- If the source is an anecdote, keep it an anecdote
- If the source is a methods paragraph, keep it a methods paragraph
- Do not add advice, new examples, stronger emotion, or broader claims
- Keep length within 15% of the input unless the input is obviously padded
- End naturally; do not summarize unless the original text summarized

Text:
{text}
```

---

## 12. Reporting Rules

### 12.1 Core benchmark reports must include

- overall class means
- AUROC by pair:
  - human vs AI
  - humanized vs AI
  - human vs humanized
- per-domain metrics
- length-banded metrics
- per-dimension metrics
- manual review notes for flagged cases

### 12.2 Report the OOD AI set separately

Do not mix into the core matched benchmark.

Report as:
- OOD AI mean score
- OOD AI false-negative rate
- which dimensions fail most often on long explainers

---

## 13. Immediate Next Actions

### Phase 1: freeze and relabel
1. Freeze `test_set_v02` as exploratory.
2. Mark existing external AI rows as `ood_ai` split.
3. Keep existing curated seeds only as diagnostic material.

### Phase 2: build the v03 human pool
4. Collect ~510 raw human candidates from the mapped corpora.
5. Apply hard filters.
6. Tag accepted seeds with:
   - domain
   - source
   - discourse role
   - length band
7. Accept exactly 200 seeds using the target distribution.

### Phase 3: rebuild generated data
8. Generate 2 AIified candidates per seed with subset-based/domain-aware AIify.
9. Keep the best AIified candidate per seed.
10. Generate 2 humanized candidates per seed.
11. Keep only triples that satisfy the new pair rules.

### Phase 4: benchmark and train
12. Export `benchmark_v03_core`.
13. Export `benchmark_v03_ood_ai` separately.
14. Export `sft_pairs_v02` from accepted triples only.
15. Manually inspect all flagged pairs before using them for SFT.

---

## 14. Non-Goals for V03

Do not do these in the same pass:

- redesign Layer 2 dimensions
- add non-Google generation models to repo configs
- expand the benchmark with arbitrary scraped web text
- optimize for scale before the matched benchmark is credible

---

## 15. Success Criteria

The v03 rebuild is successful if:

1. the human seed pool hits the target domain distribution exactly
2. median human seed length is >100 words in every bucket except possibly email
3. humanized outputs no longer systematically exceed original humans
4. pair acceptance rejects role drift automatically
5. the core benchmark and OOD AI benchmark are reported separately
6. low-performing dimensions are stressed by real domain diversity, not just more tech blog prose
