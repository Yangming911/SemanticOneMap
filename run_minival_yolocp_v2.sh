#!/usr/bin/env bash
# Run yolocp v2 (Method B + sliding window) as 3 parallel shards of 10 episodes.
set -euo pipefail

LOGDIR="/tmp/minival_v2_logs"
SUMMARY="/tmp/minival_v2_summary.log"
mkdir -p "$LOGDIR"

run_exp() {
    local name="$1"
    local config="$2"
    local logfile="$LOGDIR/${name}.log"
    echo "[$(date '+%H:%M:%S')] Starting $name ..."
    conda run -n onemap bash -c \
        "cd /opt/data/private/onemap/OneMap_obstacle && PYTORCH_NO_NVML=1 xvfb-run -a python -u eval_habitat.py -c $config" \
        > "$logfile" 2>&1
    echo "[$(date '+%H:%M:%S')] Done: $name"
}

export -f run_exp

run_exp yolocp_s0 config/mon/eval_conf_minival_yolocp_s0.yaml &
PID0=$!
run_exp yolocp_s1 config/mon/eval_conf_minival_yolocp_s1.yaml &
PID1=$!
run_exp yolocp_s2 config/mon/eval_conf_minival_yolocp_s2.yaml &
PID2=$!

echo "PIDs: s0=$PID0  s1=$PID1  s2=$PID2"
echo "Logs in $LOGDIR"

wait $PID0; S0=$?
wait $PID1; S1=$?
wait $PID2; S2=$?

echo ""
echo "========================================"
echo "All done. Exit codes: s0=$S0 s1=$S1 s2=$S2"
echo "========================================"

# Merge into one summary
{
    echo "========================================"
    echo "  MINIVAL V2 YOLOCP SUMMARY — $(date)"
    echo "========================================"
    for name in yolocp_s0 yolocp_s1 yolocp_s2; do
        echo ""
        echo "----------------------------------------"
        echo "  SHARD: $name"
        echo "----------------------------------------"
        cat "$LOGDIR/${name}.log"
    done
    echo ""
    echo "========================================"
    echo "  END OF SUMMARY"
    echo "========================================"
} > "$SUMMARY"

echo "Summary written to: $SUMMARY"

# Extract and aggregate key metrics
echo ""
echo "=== AGGREGATED METRICS ==="
grep -h "SR\|SPL\|SEMANTIC_COLLISION\|FAILURE" "$LOGDIR"/yolocp_s*.log | grep -v "^Result" | sort | uniq -c | sort -rn | head -20
