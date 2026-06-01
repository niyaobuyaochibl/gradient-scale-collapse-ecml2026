#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extended"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_fusion.py"
RESULT_ROOT="data/extended_fusion/results"
LOG_ROOT="data/extended_fusion/logs/route_c_core"

SEEDS_CSV="${SEEDS_CSV:-42,123,999,2024,2025}"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=$' \t\n'

declare -A CFG
declare -A OUT

# MovieLens-1M
CFG["ml1m:lightgcn_only"]="${CONFIG_DIR}/movielens_lightgcn_only.yaml"
OUT["ml1m:lightgcn_only"]="${RESULT_ROOT}/movielens_lightgcn_only"
CFG["ml1m:fixed"]="${CONFIG_DIR}/movielens_fixed_baseline.yaml"
OUT["ml1m:fixed"]="${RESULT_ROOT}/ml1m/fixed"
CFG["ml1m:gbaf"]="${CONFIG_DIR}/movielens_gbaf.yaml"
OUT["ml1m:gbaf"]="${RESULT_ROOT}/movielens_gbaf"
CFG["ml1m:gbaf_adaptive"]="${CONFIG_DIR}/movielens_gbaf_adaptive.yaml"
OUT["ml1m:gbaf_adaptive"]="${RESULT_ROOT}/movielens_gbaf_adaptive"

# Amazon Fine Food
CFG["amazonff:lightgcn_only"]="${CONFIG_DIR}/amazon_lightgcn_only.yaml"
OUT["amazonff:lightgcn_only"]="${RESULT_ROOT}/amazon_lightgcn_only"
CFG["amazonff:fixed"]="${CONFIG_DIR}/amazon_fixed_baseline.yaml"
OUT["amazonff:fixed"]="${RESULT_ROOT}/amazonff/fixed"
CFG["amazonff:gbaf"]="${CONFIG_DIR}/amazon_gbaf.yaml"
OUT["amazonff:gbaf"]="${RESULT_ROOT}/amazon_gbaf"
CFG["amazonff:gbaf_adaptive"]="${CONFIG_DIR}/amazon_gbaf_adaptive.yaml"
OUT["amazonff:gbaf_adaptive"]="${RESULT_ROOT}/amazon_gbaf_adaptive"

# Yelp
CFG["yelp:lightgcn_only"]="${CONFIG_DIR}/yelp_lightgcn_only.yaml"
OUT["yelp:lightgcn_only"]="${RESULT_ROOT}/yelp_lightgcn_only"
CFG["yelp:fixed"]="${CONFIG_DIR}/yelp_fixed_baseline.yaml"
OUT["yelp:fixed"]="${RESULT_ROOT}/yelp/fixed"
CFG["yelp:gbaf"]="${CONFIG_DIR}/yelp_gbaf.yaml"
OUT["yelp:gbaf"]="${RESULT_ROOT}/yelp/gbaf"
CFG["yelp:gbaf_adaptive"]="${CONFIG_DIR}/yelp_gbaf_adaptive.yaml"
OUT["yelp:gbaf_adaptive"]="${RESULT_ROOT}/yelp/gbaf_adaptive"

# MIND
CFG["mind:lightgcn_only"]="${CONFIG_DIR}/mind_lightgcn_only.yaml"
OUT["mind:lightgcn_only"]="${RESULT_ROOT}/mind_lightgcn_only"
CFG["mind:fixed"]="${CONFIG_DIR}/mind_fixed_baseline.yaml"
OUT["mind:fixed"]="${RESULT_ROOT}/mind/fixed"
CFG["mind:gbaf"]="${CONFIG_DIR}/mind_gbaf.yaml"
OUT["mind:gbaf"]="${RESULT_ROOT}/mind/gbaf"
CFG["mind:gbaf_adaptive"]="${CONFIG_DIR}/mind_gbaf_adaptive.yaml"
OUT["mind:gbaf_adaptive"]="${RESULT_ROOT}/mind/gbaf_adaptive"

DATASETS=("ml1m" "amazonff" "yelp" "mind")
METHODS=("lightgcn_only" "fixed" "gbaf" "gbaf_adaptive")

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

mkdir -p "${LOG_ROOT}"

for ds in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    key="${ds}:${method}"
    cfg="${CFG[$key]}"
    out_dir="${OUT[$key]}"
    log_dir="${LOG_ROOT}/${ds}/${method}"
    mkdir -p "${out_dir}" "${log_dir}"

    echo "=================================================="
    echo "[$(timestamp)] Dataset=${ds} Method=${method}"
    echo "Config: ${cfg}"
    echo "Output: ${out_dir}"
    echo "Logs  : ${log_dir}"
    echo "=================================================="

    for seed in "${SEEDS[@]}"; do
      log_file="${log_dir}/seed${seed}.log"
      echo "[$(timestamp)] ▶ Running seed ${seed}"
      python -u "${TRAIN_SCRIPT}" \
        --config "${cfg}" \
        --seed "${seed}" \
        --output_dir "${out_dir}" >"${log_file}" 2>&1
      echo "[$(timestamp)] ✅ Done seed ${seed}"
    done
  done
done

echo "[$(timestamp)] All Route-C core runs finished."
