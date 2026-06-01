#!/usr/bin/env bash
# Re-run Yelp + MIND with 60 epochs (updated configs)
# ML-1M and Amazon results are already complete -- skip them.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONFIG_DIR="${PROJECT_ROOT}/configs/extended"
TRAIN_SCRIPT="${PROJECT_ROOT}/train_fusion.py"
RESULT_ROOT="data/extended_fusion/results"
LOG_ROOT="data/extended_fusion/logs/yelp_mind_60ep"

SEEDS=(42 123 999 2024 2025)

declare -A CFG
declare -A OUT

# ---------- Yelp ----------
CFG["yelp:lightgcn_only"]="${CONFIG_DIR}/yelp_lightgcn_only.yaml"
OUT["yelp:lightgcn_only"]="${RESULT_ROOT}/yelp_lightgcn_only"

CFG["yelp:fixed"]="${CONFIG_DIR}/yelp_fixed_baseline.yaml"
OUT["yelp:fixed"]="${RESULT_ROOT}/yelp/fixed"

CFG["yelp:gbaf"]="${CONFIG_DIR}/yelp_gbaf.yaml"
OUT["yelp:gbaf"]="${RESULT_ROOT}/yelp/gbaf"

CFG["yelp:gbaf_adaptive"]="${CONFIG_DIR}/yelp_gbaf_adaptive.yaml"
OUT["yelp:gbaf_adaptive"]="${RESULT_ROOT}/yelp/gbaf_adaptive"

# ---------- MIND ----------
CFG["mind:lightgcn_only"]="${CONFIG_DIR}/mind_lightgcn_only.yaml"
OUT["mind:lightgcn_only"]="${RESULT_ROOT}/mind_lightgcn_only"

CFG["mind:fixed"]="${CONFIG_DIR}/mind_fixed_baseline.yaml"
OUT["mind:fixed"]="${RESULT_ROOT}/mind/fixed"

CFG["mind:gbaf"]="${CONFIG_DIR}/mind_gbaf.yaml"
OUT["mind:gbaf"]="${RESULT_ROOT}/mind/gbaf"

CFG["mind:gbaf_adaptive"]="${CONFIG_DIR}/mind_gbaf_adaptive.yaml"
OUT["mind:gbaf_adaptive"]="${RESULT_ROOT}/mind/gbaf_adaptive"

# ---------- Ordered runs: Yelp first, then MIND ----------
DATASETS=("yelp" "mind")
METHODS=("lightgcn_only" "fixed" "gbaf" "gbaf_adaptive")

mkdir -p "${LOG_ROOT}"

TOTAL=$((${#DATASETS[@]} * ${#METHODS[@]} * ${#SEEDS[@]}))
COUNT=0

for ds in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    key="${ds}:${method}"
    config="${CFG[$key]}"
    outdir="${OUT[$key]}"
    
    for seed in "${SEEDS[@]}"; do
      COUNT=$((COUNT + 1))
      result_file="${outdir}/results_seed${seed}.json"
      log_file="${LOG_ROOT}/${ds}_${method}_seed${seed}.log"
      
      echo ""
      echo "============================================"
      echo "[${COUNT}/${TOTAL}] ${ds} / ${method} / seed=${seed}"
      echo "  config : ${config}"
      echo "  output : ${outdir}"
      echo "  log    : ${log_file}"
      echo "============================================"
      
      python -u "${TRAIN_SCRIPT}" \
        --config "${config}" \
        --seed "${seed}" \
        --output_dir "${outdir}" \
        2>&1 | tee "${log_file}"
      
      echo "[DONE] ${ds}/${method}/seed${seed} -> exit $?"
    done
  done
done

echo ""
echo "========================================"
echo "ALL Yelp + MIND experiments completed!"
echo "Total runs: ${TOTAL}"
echo "========================================"
