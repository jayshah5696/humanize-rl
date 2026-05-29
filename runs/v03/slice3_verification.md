# humanize_tasks_v03 — verification report

- tasks: **99**
- rollouts: **0**
- hard gates: 1 PASS / 9 WARN / 4 FAIL

| gate | status | value | hard | soft | note |
|---|---|---|---|---|---|
| A1_instruction_length | **FAIL** | `{'p10': 74, 'p50': 92, 'p90': 159}` | `{'p90_min': 200, 'p10_max': 80}` | `{'p50_min': 110, 'p50_max': 170}` |  |
| A2_constraint_density | **PASS** | `8.6` | `4` | `5` |  |
| A3_cell_coverage | **FAIL** | `{'filled': 5, 'cells_total': 40, 'unique_cells': 10}` | `25` | `30` |  |
| A4_embedding_distance | **FAIL** | `0.485` | `0.55` | `0.62` |  |
| A5_author_balance | **FAIL** | `59.85` | `10` | `5` | observed={'google/gemini-3.1-flash-lite-preview': 0.848, 'google/gemini-3-flash-preview': 0.152} |
| B1_strong_reward_band | **WARN** | `None` | `{'median_min': 0.4, 'median_max': 0.85}` | `{'median_min': 0.5, 'median_max': 0.8}` | no strong-model rollouts |
| B2_strong_weak_gap | **WARN** | `None` | `0.1` | `0.15` | missing rollouts |
| B3_per_mode_variance | **WARN** | `None` | `0.12` | `0.18` | no rollouts |
| B4_check_firing_rate | **WARN** | `None` | `0.15` | `0.25` | no penalties recorded |
| B5_penalty_concentration | **WARN** | `None` | `0.5` | `0.4` | no penalty mass |
| C1_rollout_std | **WARN** | `None` | `0.25` | `0.18` | not enough rollouts per task |
| C2_cross_model_rho | **WARN** | `None` | `0.5` | `0.65` | only 0 paired tasks (<30); skipped |
| C3_judge_kappa | **WARN** | `None` | `0.55` | `0.7` | judge re-score requires a separate audit run; not computed here |
| C4_check_determinism | **WARN** | `None` | `bit_exact` | `bit_exact` | deterministic-check parity test requires 2 runs; see slice gate |
