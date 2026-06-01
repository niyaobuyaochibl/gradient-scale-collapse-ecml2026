#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset> [variant_csv]"
  echo "  dataset: ml1m | amazonff | yelp | mind"
  echo "  variant_csv (optional): comma-separated subset of {fixed,attention_pop,attention_pop_user,attention_concat,attention_ctx,attention_ctx_lora}"
  exit 1
fi

DATASET="$1"
VARIANT_CSV="${2:-fixed,attention_pop,attention_pop_user,attention_concat,attention_ctx,attention_ctx_lora}"

IFS=',' read -ra VARIANTS <<< "$VARIANT_CSV"

SEEDS_CSV="${SEEDS_CSV:-42,123,999,2024,2025}"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=$' 	
'  # reset IFS

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extended"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_fusion.py"
RESULT_ROOT="results/${DATASET}"
LOG_ROOT="data/extended_fusion/logs/${DATASET}"

declare -A CONFIG_MAP
CONFIG_MAP["ml1m:fixed"]="${CONFIG_DIR}/movielens_fixed_baseline.yaml"
CONFIG_MAP["ml1m:attention_pop"]="${CONFIG_DIR}/movielens_attention_pop.yaml"
CONFIG_MAP["ml1m:attention_pop_user"]="${CONFIG_DIR}/movielens_attention_pop_user.yaml"
CONFIG_MAP["ml1m:attention_concat"]="${CONFIG_DIR}/movielens_attention_concat.yaml"
CONFIG_MAP["ml1m:attention_ctx"]="${CONFIG_DIR}/movielens_attention_ctx.yaml"
CONFIG_MAP["ml1m:attention_ctx_lora"]="${CONFIG_DIR}/movielens_attention_ctx_lora.yaml"
CONFIG_MAP["amazonff:fixed"]="${CONFIG_DIR}/amazon_fixed_baseline.yaml"
CONFIG_MAP["amazonff:attention_pop"]="${CONFIG_DIR}/amazon_attention_pop.yaml"
CONFIG_MAP["amazonff:attention_pop_user"]="${CONFIG_DIR}/amazon_attention_pop_user.yaml"
CONFIG_MAP["amazonff:attention_concat"]="${CONFIG_DIR}/amazon_attention_concat.yaml"
CONFIG_MAP["amazonff:attention_ctx"]="${CONFIG_DIR}/amazon_attention_ctx.yaml"
CONFIG_MAP["amazonff:attention_ctx_lora"]="${CONFIG_DIR}/amazon_attention_ctx_lora.yaml"
CONFIG_MAP["yelp:fixed"]="${CONFIG_DIR}/yelp_fixed_baseline.yaml"
CONFIG_MAP["yelp:attention_pop"]="${CONFIG_DIR}/yelp_attention_pop.yaml"
CONFIG_MAP["yelp:attention_pop_user"]="${CONFIG_DIR}/yelp_attention_pop_user.yaml"
CONFIG_MAP["yelp:attention_concat"]="${CONFIG_DIR}/yelp_attention_concat.yaml"
CONFIG_MAP["yelp:attention_ctx"]="${CONFIG_DIR}/yelp_attention_ctx.yaml"
CONFIG_MAP["yelp:attention_ctx_lora"]="${CONFIG_DIR}/yelp_attention_ctx_lora.yaml"
CONFIG_MAP["mind:fixed"]="${CONFIG_DIR}/mind_fixed_baseline.yaml"
CONFIG_MAP["mind:attention_pop"]="${CONFIG_DIR}/mind_attention_pop.yaml"
CONFIG_MAP["mind:attention_pop_user"]="${CONFIG_DIR}/mind_attention_pop_user.yaml"
CONFIG_MAP["mind:attention_concat"]="${CONFIG_DIR}/mind_attention_concat.yaml"
CONFIG_MAP["mind:attention_ctx"]="${CONFIG_DIR}/mind_attention_ctx.yaml"
CONFIG_MAP["mind:attention_ctx_lora"]="${CONFIG_DIR}/mind_attention_ctx_lora.yaml"

if [[ ! -v CONFIG_MAP["${DATASET}:fixed"] ]]; then
  echo "Unknown dataset: ${DATASET}"
  exit 1
fi

mkdir -p "$RESULT_ROOT" "$LOG_ROOT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

for variant in "${VARIANTS[@]}"; do
  key="${DATASET}:${variant}"
  config_path="${CONFIG_MAP[$key]:-}" ;
  if [[ -z "$config_path" ]]; then
    echo "Skipping unsupported variant '${variant}' for dataset '${DATASET}'."
    continue
  fi

  variant_result_dir="${RESULT_ROOT}/${variant}"
  variant_log_dir="${LOG_ROOT}/${variant}"
  mkdir -p "$variant_result_dir" "$variant_log_dir"

  echo "============================================"
  echo "Dataset: ${DATASET} | Variant: ${variant}"
  echo "Config: ${config_path}"
  echo "Result dir: ${variant_result_dir}"
  echo "Log dir: ${variant_log_dir}"
  echo "============================================"

  for seed in "${SEEDS[@]}"; do
    log_file="${variant_log_dir}/${variant}_seed${seed}.log"
    echo "[$(timestamp)] ▶️  Running seed ${seed} (log: ${log_file})"
    python -u "$TRAIN_SCRIPT" \
      --config "$config_path" \
      --seed "$seed" \
      --output_dir "$variant_result_dir" \
      >"$log_file" 2>&1
    echo "[$(timestamp)] ✅  Completed seed ${seed}"
  done

  echo "[$(timestamp)] 🎯 Variant ${variant} completed."
done

echo "[$(timestamp)] All requested runs finished."


