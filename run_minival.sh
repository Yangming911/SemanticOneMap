#!/usr/bin/env bash
# Run 3 minival experiments in parallel and merge logs when done.
set -euo pipefail

LOGDIR="/tmp/minival_logs"
SUMMARY="/tmp/minival_summary.log"
mkdir -p "$LOGDIR"

cd /opt/data/private/onemap/OneMap_obstacle

run_exp() {
    local name="$1"
    local config="$2"
    local logfile="$LOGDIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting $name ..."
    conda run -n onemap bash -c \
        "PYTORCH_NO_NVML=1 xvfb-run -a python -u eval_habitat.py -c $config" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] Done: $name"
}

export -f run_exp

# Launch all 3 in parallel
run_exp minival_baseline  config/mon/eval_conf_minival_baseline.yaml  &
PID0=$!
run_exp minival_yolo      config/mon/eval_conf_minival_yolo.yaml      &
PID1=$!
run_exp minival_yolocp    config/mon/eval_conf_minival_yolocp.yaml    &
PID2=$!

echo "PIDs: baseline=$PID0  yolo=$PID1  yolocp=$PID2"
echo "Logs in $LOGDIR"

wait $PID0; S0=$?
wait $PID1; S1=$?
wait $PID2; S2=$?

echo ""
echo "========================================"
echo "All done. Exit codes: baseline=$S0 yolo=$S1 yolocp=$S2"
echo "========================================"

# Merge into one summary file
{
    echo "========================================"
    echo "  MINIVAL SUMMARY — $(date)"
    echo "========================================"
    for name in minival_baseline minival_yolo minival_yolocp; do
        echo ""
        echo "----------------------------------------"
        echo "  EXPERIMENT: $name"
        echo "----------------------------------------"
        cat "$LOGDIR/${name}.log"
    done
    echo ""
    echo "========================================"
    echo "  END OF SUMMARY"
    echo "========================================"
} > "$SUMMARY"

echo "Summary written to: $SUMMARY"
