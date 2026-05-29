# humanize_tasks_v03 — verification report

- tasks: **594**
- rollouts: **0**
- hard gates: 1 PASS / 10 WARN / 3 FAIL

| gate | status | value | hard | soft | note |
|---|---|---|---|---|---|
| A1_instruction_length | **FAIL** | `{'p10': 82, 'p50': 123, 'p90': 182}` | `{'p90_min': 200, 'p10_max': 80}` | `{'p50_min': 110, 'p50_max': 170}` |  |
| A2_constraint_density | **PASS** | `8.49` | `4` | `5` |  |
| A3_cell_coverage | **FAIL** | `{'filled': 8, 'cells_total': 40, 'unique_cells': 21}` | `25` | `30` |  |
| A4_embedding_distance | **WARN** | `0.578` | `0.55` | `0.62` |  |
| A5_author_balance | **FAIL** | `40.0` | `10` | `5` | observed={'openai/gpt-5.4-mini': 0.551, 'google/gemini-3.1-flash-lite-preview': 0.449} |
| B1_strong_reward_band | **WARN** | `None` | `{'median_min': 0.4, 'median_max': 0.85}` | `{'median_min': 0.5, 'median_max': 0.8}` | no strong-model rollouts |
| B2_strong_weak_gap | **WARN** | `None` | `0.1` | `0.15` | missing rollouts |
| B3_per_mode_variance | **WARN** | `None` | `0.12` | `0.18` | no rollouts |
| B4_check_firing_rate | **WARN** | `None` | `0.15` | `0.25` | no penalties recorded |
| B5_penalty_concentration | **WARN** | `None` | `0.5` | `0.4` | no penalty mass |
| C1_rollout_std | **WARN** | `None` | `0.25` | `0.18` | not enough rollouts per task |
| C2_cross_model_rho | **WARN** | `None` | `0.5` | `0.65` | only 0 paired tasks (<30); skipped |
| C3_judge_kappa | **WARN** | `None` | `0.55` | `0.7` | judge re-score requires a separate audit run; not computed here |
| C4_check_determinism | **WARN** | `None` | `bit_exact` | `bit_exact` | deterministic-check parity test requires 2 runs; see slice gate |
