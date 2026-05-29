#!/usr/bin/env bash
# Launcher for slice 4: 6 modes x 3 authors. Reads pre-split seed shards from
# data/raw/v03_seeds_by_author/{author_slug}__{mode}.jsonl.
#
# Pro uses --skip-errors + --extra-body to cap reasoning. Others run plain.
# Per-mode arka outputs land in data/processed/v03_{mode}_{author_slug}.jsonl.

set -uo pipefail

LOG_DIR="logs/v03_slice4_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR" data/processed

# Note: Gemini 3.1 Pro dropped from slice 4 because OpenRouter currently
# ignores reasoning.max_tokens caps for that model, leading to consistent
# LengthFinishReasonError. Revisit when fixed.
GPT="openai/gpt-5.4-mini"
FLASH="google/gemini-3.1-flash-lite-preview"
MODES=(rewrite_humanize compression tone_shift expansion long_form_generate multi_constraint_compose)
AUTHORS=("$GPT" "$FLASH")

start_ts=$(date +%s)
total=0
ok=0
fail=0
for mode in "${MODES[@]}"; do
  for author in "${AUTHORS[@]}"; do
    total=$((total+1))
    slug="${author//\//_}"
    seed="data/raw/v03_seeds_by_author/${slug}__${mode}.jsonl"
    out="data/processed/v03_${mode}_${slug}.jsonl"
    log="$LOG_DIR/${mode}__${slug}.log"
    if [ ! -f "$seed" ]; then
      echo "SKIP: missing $seed"
      continue
    fi
    # Always pass --skip-errors so per-row LLM failures don't kill a run.
    extra_flags=(--skip-errors)
    echo
    echo "====== [$total/12] $mode  author=$author  seeds=$(wc -l < "$seed") ======"
    if timeout 1800 uv run scripts/data/build/run_arka_pipeline.py \
        --config "configs/v03/v03_${mode}.yaml" \
        --author "$author" \
        --out "$out" \
        --seed-path "$seed" \
        --target-count 1 \
        --run-id "v03-s4-${mode}-${slug}" \
        "${extra_flags[@]}" 2>&1 | tee "$log" | tail -8
    then
      ok=$((ok+1))
    else
      fail=$((fail+1))
      echo "  ! FAILED — see $log"
    fi
  done
done

elapsed=$(( $(date +%s) - start_ts ))
echo
echo "============================================================"
echo "slice 4 complete: $ok/$total OK, $fail FAIL, ${elapsed}s elapsed (2 authors x 6 modes = 12)"
echo "logs in $LOG_DIR"
ls -la data/processed/v03_*.jsonl | tail -20
