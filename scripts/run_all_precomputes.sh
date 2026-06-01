#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="${ROOT_DIR}/experiments/extended/scripts/precompute_llm_embeddings.py"
OUTPUT_ROOT="data/extended_fusion/embeddings"

ENCODERS=(minilm mpnet roberta simcse)
DATASETS=(ml1m amazonff)

echo "== Precomputing text embeddings =="

for dataset in "${DATASETS[@]}"; do
  for encoder in "${ENCODERS[@]}"; do
    echo "--> Dataset: ${dataset}, Encoder: ${encoder}"
    python "$SCRIPT" "$dataset" "$encoder" --batch-size 64 --device cpu --output-dir "$OUTPUT_ROOT" "$@"
  done
done

echo "== Done =="


