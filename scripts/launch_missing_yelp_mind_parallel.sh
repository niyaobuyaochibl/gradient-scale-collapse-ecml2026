#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="outputs/"
TRAIN_SCRIPT="$PROJECT_ROOT/train_fusion.py"
RESULT_ROOT="data/extended_fusion/results"
LOG_ROOT="data/extended_fusion/logs/yelp_mind_60ep_parallel"
mkdir -p "$LOG_ROOT"

launch(){
  local cfg="$1"; local out="$2"; local seed="$3";
  local outdir="$RESULT_ROOT/$out"
  local result="$outdir/results_seed${seed}.json"
  local logf="$LOG_ROOT/$(echo "$out" | tr '/' '_')_seed${seed}.log"
  mkdir -p "$outdir"
  if [ -f "$result" ]; then
    echo "[SKIP done] $out seed=$seed"
    return
  fi
  if pgrep -af "train_fusion.py.*--config .*${cfg}.*--seed ${seed}.*--output_dir ${outdir}" >/dev/null; then
    echo "[SKIP running] $out seed=$seed"
    return
  fi
  echo "[LAUNCH] $out seed=$seed"
  python -u "$TRAIN_SCRIPT" --config "$PROJECT_ROOT/$cfg" --seed "$seed" --output_dir "$outdir" > "$logf" 2>&1 &
}

# Yelp GBAF
for s in 42 123 999 2024 2025; do launch "configs/extended/yelp_gbaf.yaml" "yelp/gbaf" "$s"; done
# Yelp GBAF-Adaptive (seed42 already running separately)
for s in 123 999 2024 2025; do launch "configs/extended/yelp_gbaf_adaptive.yaml" "yelp/gbaf_adaptive" "$s"; done
# MIND all methods remaining seeds
for s in 123 999 2024 2025; do launch "configs/extended/mind_lightgcn_only.yaml" "mind_lightgcn_only" "$s"; done
for s in 123 999 2024 2025; do launch "configs/extended/mind_fixed_baseline.yaml" "mind/fixed" "$s"; done
for s in 123 999 2024 2025; do launch "configs/extended/mind_gbaf.yaml" "mind/gbaf" "$s"; done
for s in 123 999 2024 2025; do launch "configs/extended/mind_gbaf_adaptive.yaml" "mind/gbaf_adaptive" "$s"; done

echo "Launch dispatch finished"
