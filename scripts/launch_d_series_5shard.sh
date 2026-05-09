#!/usr/bin/env bash
set -uo pipefail

cd /inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap
export PATH=/root/miniconda3/envs/onemap/bin:$PATH

# 10 shards: GPU0 s0-s4 (delay0->10), GPU1 s0-s4 (delay5->20), interleaved 120s
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
echo "[$(date '+%F %T')] all 10 sessions launched"
tmux ls 2>&1 | head -15
