#!/usr/bin/env bash
# Stagger-launch 12 D-series shards (GPU0: D0->D2, GPU1: D1->D3) with 120s offset.
# This avoids ViT-B-16.pt SHA256 race + GPU memory burst during model tracing.
set -uo pipefail

cd /inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap
export PATH=/root/miniconda3/envs/onemap/bin:$PATH  # tmux on PATH

# Interleave GPU 0 and GPU 1 shards so neither GPU gets all 6 launches in a row.
# Pairs (gpu, shard, delay_a, delay_b):
LAUNCHES=(
  "0 0 0 10"
  "1 0 5 20"
  "0 1 0 10"
  "1 1 5 20"
  "0 2 0 10"
  "1 2 5 20"
  "0 3 0 10"
  "1 3 5 20"
  "0 4 0 10"
  "1 4 5 20"
  "0 5 0 10"
  "1 5 5 20"
)

for i in "${!LAUNCHES[@]}"; do
  read -r GPU SHARD DA DB <<<"${LAUNCHES[$i]}"
  SESSION="g${GPU}_s${SHARD}"
  echo "[$(date '+%F %T')] launching ${SESSION} (gpu=${GPU} shard=${SHARD} ${DA}->${DB})"
  tmux new-session -d -s "${SESSION}" \
    "bash scripts/run_d_shard.sh ${GPU} ${SHARD} ${DA} ${DB}"
  if [ $((i+1)) -lt ${#LAUNCHES[@]} ]; then
    sleep 120
  fi
done
echo "[$(date '+%F %T')] all 12 sessions launched"
tmux ls 2>&1 | head -20
