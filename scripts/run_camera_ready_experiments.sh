#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_DIR="${ROOT_DIR}/configs/final"
TRAIN_SCRIPT="${ROOT_DIR}/code/train_fusion.py"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/results/reruns}"
IFS=',' read -r -a SEEDS <<< "${SEEDS_CSV:-42,123,999,2024,2025}"
IFS=',' read -r -a DATASETS <<< "${DATASETS_CSV:-movielens,amazon_ff,amazon_books,amazon_cds,yelp,mind}"
IFS=',' read -r -a METHODS <<< "${METHODS_CSV:-cf_only,fixed,branch_norm,gbaf,concat_mlp}"

for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do
    config="${CONFIG_DIR}/${dataset}_${method}.yaml"
    if [[ ! -f "${config}" ]]; then
      echo "[SKIP] no final config: ${dataset}/${method}"
      continue
    fi
    for seed in "${SEEDS[@]}"; do
      output="${OUTPUT_ROOT}/${dataset}/${method}"
      mkdir -p "${output}"
      echo "[RUN] ${dataset}/${method}/seed=${seed}"
      python "${TRAIN_SCRIPT}" --config "${config}" --seed "${seed}" --output_dir "${output}"
    done
  done
done
