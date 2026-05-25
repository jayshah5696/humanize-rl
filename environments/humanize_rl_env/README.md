# humanize-rl-env

Prime Verifiers single-turn environment for the Humanize-RL reward suite.

## Shape

Each task is one episode:

1. dataset row supplies `prompt`, `task_id`, `task`, and JSON `info`;
2. model produces one completion;
3. `humanize_reward(completion, task, state)` calls `humanize_rl.reward.score_response`;
4. the full `RewardResult` is cached in rollout `state`;
5. Verifiers returns the scalar reward and logs zero-weight rubric metrics.

This follows the current Prime Verifiers simple environment pattern:

```python
rubric = vf.Rubric(funcs=[humanize_reward])
rubric.add_metric(style_metric)
rubric.add_metric(task_following_metric)
rubric.add_metric(faithfulness_metric)
rubric.add_metric(length_metric)
rubric.add_metric(format_metric)
rubric.add_metric(risk_penalty_metric)
return vf.SingleTurnEnv(dataset=dataset, rubric=rubric)
```

Metrics exposed:

- `style_metric`
- `task_following_metric`
- `faithfulness_metric`
- `length_metric`
- `format_metric`
- `clarity_metric`
- `placeholder_metric`
- `risk_penalty_metric`
- `option_menu_penalty_metric`
- `wrapper_phrase_penalty_metric`
- `invented_detail_penalty_metric`

OpenEnv is intentionally not used here because this task has no multi-turn state,
external tools, WebSocket server, or action/observation protocol.

## Usage

```bash
prime eval run humanize-rl-env
```

Optional loader args:

```python
load_environment(split="validation", task_path="data/rl/humanize_tasks_v01_smoke.jsonl")
```

Splits: `train`, `validation`, `test`, `all`.
