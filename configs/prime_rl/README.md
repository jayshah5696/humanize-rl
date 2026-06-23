# Prime `prime-rl` SFT configs

These configs are for Prime's open `prime-rl` runtime, not Hosted Training
`prime train` env-RL TOMLs in `configs/prime/`.

Use this path first for dataset SFT:

```bash
uv run sft @ configs/prime_rl/qwen35_08b_sft_smoke_env0314.toml \
  --wandb.project humanize-rl \
  --wandb.name qwen35-08b-prime-sft-smoke-env0314
```

The SFT dataset is a dedicated one-file-per-split HF dataset so Prime does not
accidentally load older parquet/JSONL artifacts from the broader SFT repo:

```text
jayshah5696/humanize-rl-prime-sft-messages-env0314
```

Current split counts:

- train: `4313`
- validation: `239`
- test: `241`
- total accepted: `4793`
- upstream rejected by builder: `62`
- duplicate ids: `0`
- repair-reference rows: `20`

Important:

- Qwen3.5 configs set `[renderer] name = "qwen3.5"` based on the current
  `renderers` model map.
- These configs are schema-checked locally with `prime-rl-configs`; full
  `uv run sft --dry-run` needs Linux/CUDA because full `prime-rl` depends on
  CUDA torch wheels.
- Keep W&B enabled by TOML or CLI. Never commit W&B keys.
