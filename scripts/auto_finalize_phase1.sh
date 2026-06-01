#!/usr/bin/env bash
set -euo pipefail

RESULT_ROOT="data/extended_fusion/results"
ANALYSIS_ROOT="experiments/extended/analysis"
LOG_FILE="data/extended_fusion/logs/auto_finalize_phase1.log"

mkdir -p "$(dirname "$LOG_FILE")"

count_jsons(){
  local dir="$1"
  shopt -s nullglob
  local arr=("$RESULT_ROOT/$dir"/results_seed*.json)
  shopt -u nullglob
  echo "${#arr[@]}"
}

mmrec_done_count(){
  python3 - <<'PY'
import glob,re
base='results/mmrec'
count=0
for ds in ['ecml_ml1m','ecml_amazonff','ecml_yelp','ecml_mind']:
  for m in ['BM3','LATTICE','FREEDOM']:
    for log in glob.glob(f'{base}/{ds}/{m}/seed*/train.log'):
      t=open(log,'r',errors='ignore').read()
      if (re.search(r'BEST.*Test:',t,re.S) is not None) or ('best test:' in t.lower()):
        count += 1
print(count)
PY
}

echo "[$(date)] auto finalize monitor started" | tee -a "$LOG_FILE"

while true; do
  y1=$(count_jsons yelp_lightgcn_only)
  y2=$(count_jsons yelp/fixed)
  y3=$(count_jsons yelp/gbaf)
  y4=$(count_jsons yelp/gbaf_adaptive)
  m1=$(count_jsons mind_lightgcn_only)
  m2=$(count_jsons mind/fixed)
  m3=$(count_jsons mind/gbaf)
  m4=$(count_jsons mind/gbaf_adaptive)
  mm=$(mmrec_done_count)

  echo "[$(date)] yelp=($y1,$y2,$y3,$y4) mind=($m1,$m2,$m3,$m4) mmrec_done=$mm/60" | tee -a "$LOG_FILE"

  if [ "$y1" -ge 5 ] && [ "$y2" -ge 5 ] && [ "$y3" -ge 5 ] && [ "$y4" -ge 5 ] \
     && [ "$m1" -ge 5 ] && [ "$m2" -ge 5 ] && [ "$m3" -ge 5 ] && [ "$m4" -ge 5 ] \
     && [ "$mm" -ge 60 ]; then
    echo "[$(date)] all phase1 runs complete, launching final analyses" | tee -a "$LOG_FILE"

    CUDA_VISIBLE_DEVICES='' python experiments/extended/analysis/stratified_eval.py \
      --datasets ml1m amazonff yelp mind \
      --variants lightgcn_only fixed gbaf gbaf_adaptive \
      --seeds 42 123 999 2024 2025 \
      --limit-test 5000 \
      --output-dir "$ANALYSIS_ROOT" | tee -a "$LOG_FILE"

    python experiments/extended/analysis/plot_gradient_ratio.py | tee -a "$LOG_FILE"

    echo "[$(date)] auto finalize completed" | tee -a "$LOG_FILE"
    exit 0
  fi

  sleep 300
done
