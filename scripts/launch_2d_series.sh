#!/usr/bin/env bash
# Instance 2d: stagger-launch 10 shards (OACP + argmax, both open-vocab + 5 holdout) on val_ablation.
# GPU 0 runs OACP s0..s4, GPU 1 runs argmax s0..s4. 120s offset to avoid model-tracing OOM.
set -uo pipefail

cd /inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap
export PATH=/root/miniconda3/envs/onemap/bin:$PATH  # tmux on PATH

CONDA_INIT="source /root/miniconda3/etc/profile.d/conda.sh && conda activate onemap"
ENV_VARS="PYTORCH_NO_NVML=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

# Pairs (gpu, shard, method):
LAUNCHES=(
  "0 0 oacp"
  "1 0 argmax"
  "0 1 oacp"
  "1 1 argmax"
  "0 2 oacp"
  "1 2 argmax"
  "0 3 oacp"
  "1 3 argmax"
  "0 4 oacp"
  "1 4 argmax"
)

for i in "${!LAUNCHES[@]}"; do
  read -r GPU SHARD METHOD <<<"${LAUNCHES[$i]}"
  CFG="config/mon/eval_${METHOD}_ablation.yaml"
  OUT="results/mp3d_${METHOD}_ablation/s${SHARD}"
  SESSION="2d_g${GPU}_s${SHARD}_${METHOD}"
  mkdir -p "${OUT}"
  echo "[$(date '+%F %T')] launching ${SESSION}"
  tmux new-session -d -s "${SESSION}" \
    "${CONDA_INIT} && CUDA_VISIBLE_DEVICES=${GPU} ${ENV_VARS} \
     xvfb-run -a python -u eval_habitat.py -c ${CFG} \
       --EvalConf.object_nav_path datasets/objectnav_mp3d_v1/val_ablation_s${SHARD}/content/ \
       --EvalConf.results_path ${OUT} \
       2>&1 | tee ${OUT}/log.txt"
  if [ $((i+1)) -lt ${#LAUNCHES[@]} ]; then
    sleep 120
  fi
done
echo "[$(date '+%F %T')] all ${#LAUNCHES[@]} sessions launched"
tmux ls 2>&1 | head -20
