# humanize_tasks_v03 — verification report

- tasks: **492**
- rollouts: **2576**
- hard gates: 5 PASS / 4 WARN / 5 FAIL

| gate | status | value | hard | soft | note |
|---|---|---|---|---|---|
| A1_instruction_length | **FAIL** | `{'p10': 81, 'p50': 120, 'p90': 180}` | `{'p90_min': 200, 'p10_max': 80}` | `{'p50_min': 110, 'p50_max': 170}` |  |
| A2_constraint_density | **PASS** | `8.44` | `4` | `5` |  |
| A3_cell_coverage | **FAIL** | `{'filled': 8, 'cells_total': 40, 'unique_cells': 21}` | `25` | `30` |  |
| A4_embedding_distance | **WARN** | `0.573` | `0.55` | `0.62` |  |
| A5_author_balance | **PASS** | `2.32` | `10` | `5` | observed={'openai/gpt-5.4-mini': 0.573, 'google/gemini-3.1-flash-lite-preview': 0.427} |
| B1_strong_reward_band | **WARN** | `0.807` | `{'median_min': 0.4, 'median_max': 0.85}` | `{'median_min': 0.5, 'median_max': 0.8}` |  |
| B2_strong_weak_gap | **FAIL** | `0.065` | `0.1` | `0.15` | strong=openai/gpt-5.4-mini mean=0.780; weak=google/gemini-3.1-flash-lite-preview mean=0.715 |
| B3_per_mode_variance | **FAIL** | `0.086` | `0.12` | `0.18` | per_mode={'compression': 0.086, 'expansion': 0.116, 'long_form_generate': 0.146, 'multi_constraint_compose': 0.091, 'rewrite_humanize': 0.087, 'tone_shift': 0.107} |
| B4_check_firing_rate | **FAIL** | `0.003` | `0.15` | `0.25` | per_check={'ai_tell_phrase': 0.046, 'placeholder_disallowed': 0.042, 'subject_line': 0.014, 'invented_detail': 0.005, 'wrong_format_markdown': 0.106, 'signoff': 0.01, 'option_menu': 0.003, 'wrapper_phrase': 0.004} |
| B5_penalty_concentration | **PASS** | `0.378` | `0.5` | `0.4` |  |
| C1_rollout_std | **PASS** | `0.032` | `0.25` | `0.18` |  |
| C2_cross_model_rho | **PASS** | `0.685` | `0.5` | `0.65` | primary=openai/gpt-5.4-mini; second=google/gemini-3-flash-preview; n_paired=200 |
| C3_judge_kappa | **WARN** | `None` | `0.55` | `0.7` | judge re-score requires a separate audit run; not computed here |
| C4_check_determinism | **WARN** | `None` | `bit_exact` | `bit_exact` | deterministic-check parity test requires 2 runs; see slice gate |
