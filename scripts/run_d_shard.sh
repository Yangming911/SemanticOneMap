#!/usr/bin/env bash
# Run two delay-ablation phases sequentially on one GPU+shard pair.
# Usage: run_d_shard.sh <gpu_id> <shard_id> <delay_a> <delay_b>
set -uo pipefail

GPU=$1
SHARD=$2
DELAY_A=$3
DELAY_B=$4

cd /inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap

source /root/miniconda3/etc/profile.d/conda.sh
conda activate onemap

export CUDA_VISIBLE_DEVICES=${GPU}
export PYTORCH_NO_NVML=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

run_one() {
  local DELAY=$1
  local OUT="results/mp3d_oacp_delay${DELAY}/s${SHARD}"
  mkdir -p "${OUT}"
  echo "[$(date '+%F %T')] >>> START GPU${GPU} shard${SHARD} delay${DELAY}"
  xvfb-run -a python -u eval_habitat.py -c config/mon/eval_oacp_delay${DELAY}.yaml \
    --EvalConf.object_nav_path datasets/objectnav_mp3d_v1/val_ablation_s${SHARD}/content/ \
    --EvalConf.results_path "${OUT}" \
    2>&1 | tee "${OUT}/log.txt"
  local STATUS=${PIPESTATUS[0]}
  echo "[$(date '+%F %T')] <<< END GPU${GPU} shard${SHARD} delay${DELAY} (exit=${STATUS})"
  return ${STATUS}
}

# Phase 1
run_one "${DELAY_A}" || { echo "Phase A (delay=${DELAY_A}) failed; aborting Phase B"; exit 1; }
# Phase 2 (only if Phase 1 succeeded)
run_one "${DELAY_B}"
