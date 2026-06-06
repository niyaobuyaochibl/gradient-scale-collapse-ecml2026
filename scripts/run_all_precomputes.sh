#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="${ROOT_DIR}/scripts/precompute_llm_embeddings.py"
DATASETS=(ml1m amazonff amazon_books amazon_cds yelp mind)

cd "${ROOT_DIR}"
for dataset in "${DATASETS[@]}"; do
  python "${SCRIPT}" "${dataset}" minilm --device "${DEVICE:-cpu}" --output-dir "${ROOT_DIR}/embeddings" "$@"
done
