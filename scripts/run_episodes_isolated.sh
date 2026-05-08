#!/bin/bash
source /root/miniconda3/etc/profile.d/conda.sh && conda activate onemap
WD="/inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap"
cd $WD

EVAL_YAML="$1"
RESULTS_DIR="$2"
shift 2

for ep in "$@"; do
  end=$((ep + 1))
  state_file="$RESULTS_DIR/state/state_${ep}.txt"
  if [ -f "$state_file" ]; then
    echo "ep $ep already done, skip"
    continue
  fi
  echo "$(date +%H:%M:%S) Running ep $ep..."
  timeout 1800 xvfb-run -a python -u eval_habitat.py \
    -c "$EVAL_YAML" \
    --EvalConf.ep_start $ep --EvalConf.ep_end $end \
    2>&1 | tee "$RESULTS_DIR/log_ep${ep}.txt"
  ret=$?
  if [ $ret -ne 0 ]; then
    echo "ep $ep CRASHED (exit=$ret), skipping"
  fi
done
echo "ALL DONE"
