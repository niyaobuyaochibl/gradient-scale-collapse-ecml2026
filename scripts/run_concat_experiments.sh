#!/bin/bash
# Run ConcatMLP fusion on all 6 datasets x 5 seeds for Direction A+ architecture contrast
# Full-ranking evaluation matching main paper protocol

SEEDS=(42 123 999 2024 2025)
OUTPUT_BASE="results/concat_mlp"

CONFIGS=(
    "configs/extended/movielens_concat_mlp.yaml"
    "configs/extended/amazon_concat_mlp.yaml"
    "configs/extended/amazon_books_concat_mlp.yaml"
    "configs/extended/amazon_cds_concat_mlp.yaml"
    "configs/extended/yelp_concat_mlp.yaml"
    "configs/extended/mind_concat_mlp.yaml"
)

DATASET_NAMES=(
    "ml1m"
    "amazon_ff"
    "amazon_books"
    "amazon_cds"
    "yelp"
    "mind"
)

echo "========================================="
echo "ConcatMLP Full-Ranking Experiments"
echo "6 datasets x 5 seeds = 30 runs"
echo "========================================="

TOTAL=0
DONE=0
FAIL=0

for idx in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$idx]}"
    ds="${DATASET_NAMES[$idx]}"

    for seed in "${SEEDS[@]}"; do
        TOTAL=$((TOTAL + 1))
        outdir="${OUTPUT_BASE}/${ds}/seed_${seed}"

        if [ -f "${outdir}/results_seed${seed}.json" ]; then
            echo "[SKIP] ${ds} seed=${seed} — already done"
            DONE=$((DONE + 1))
            continue
        fi

        echo ""
        echo ">>> [${TOTAL}/30] Running ${ds} seed=${seed}"
        echo "    Config: ${config}"
        echo "    Output: ${outdir}"

        python train_fusion.py \
            --config "${config}" \
            --seed "${seed}" \
            --output_dir "${outdir}" \
            2>&1 | tail -5

        if [ $? -eq 0 ]; then
            DONE=$((DONE + 1))
            echo "    [OK] ${ds} seed=${seed} done"
        else
            FAIL=$((FAIL + 1))
            echo "    [FAIL] ${ds} seed=${seed}"
        fi
    done
done

echo ""
echo "========================================="
echo "Summary: ${DONE} done, ${FAIL} failed, out of ${TOTAL} total"
echo "========================================="
