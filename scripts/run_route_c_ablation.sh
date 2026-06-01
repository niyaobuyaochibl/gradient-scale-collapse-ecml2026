#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extended"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_fusion.py"
RESULT_ROOT="data/extended_fusion/results"
LOG_ROOT="data/extended_fusion/logs/route_c_ablation/ml1m"

SEEDS_CSV="${SEEDS_CSV:-42,123,999,2024,2025}"
IFS=',' read -ra SEEDS <<< "$SEEDS_CSV"
IFS=$' \t\n'

declare -A CFG
declare -A OUT

CFG["gbaf_no_grad"]="${CONFIG_DIR}/movielens_gbaf_no_grad.yaml"
OUT["gbaf_no_grad"]="${RESULT_ROOT}/movielens_gbaf_no_grad"
CFG["gbaf_no_reg"]="${CONFIG_DIR}/movielens_gbaf_no_reg.yaml"
OUT["gbaf_no_reg"]="${RESULT_ROOT}/movielens_gbaf_no_reg"
CFG["gbaf_no_conf"]="${CONFIG_DIR}/movielens_gbaf_no_conf.yaml"
OUT["gbaf_no_conf"]="${RESULT_ROOT}/movielens_gbaf_no_conf"
CFG["gbaf_adaptive_no_adaptive"]="${CONFIG_DIR}/movielens_gbaf_adaptive_no_adaptive.yaml"
OUT["gbaf_adaptive_no_adaptive"]="${RESULT_ROOT}/movielens_gbaf_adaptive_no_adaptive"

VARIANTS=("gbaf_no_grad" "gbaf_no_reg" "gbaf_no_conf" "gbaf_adaptive_no_adaptive")

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

for variant in "${VARIANTS[@]}"; do
  cfg="${CFG[$variant]}"
  out_dir="${OUT[$variant]}"
  log_dir="${LOG_ROOT}/${variant}"
  mkdir -p "${out_dir}" "${log_dir}"
  echo "=================================================="
  echo "[$(timestamp)] Variant=${variant}"
  echo "Config: ${cfg}"
  echo "Output: ${out_dir}"
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

echo "[$(timestamp)] Route-C ablation runs finished."
