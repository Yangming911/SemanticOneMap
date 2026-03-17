#!/usr/bin/env bash
# Run 3 variants × 3 shards = 9 jobs (3 at a time, one per variant)
# Usage: bash run_minival_ov.sh [baseline_ov|yolo_ov|yolocp_ov]
# With no arg, runs all 3 variants sequentially (3 shards in parallel each)

set -euo pipefail

LOGDIR="/tmp/minival_ov_logs"
mkdir -p "$LOGDIR"
WDIR="/opt/data/private/onemap/OneMap_obstacle"

run_variant() {
    local variant="$1"
    echo "[$(date '+%H:%M:%S')] Starting variant: $variant"
    local pids=()
    for s in 0 1 2; do
        local config="config/mon/eval_conf_minival_${variant}_s${s}.yaml"
        local logfile="$LOGDIR/${variant}_s${s}.out"
        conda run -n onemap bash -c \
            "cd $WDIR && PYTORCH_NO_NVML=1 xvfb-run -a python -u eval_habitat.py -c $config" \
            > "$logfile" 2>&1 &
        pids+=($!)
        echo "  shard s${s} → PID ${pids[-1]}, log: $logfile"
    done
    for pid in "${pids[@]}"; do
        wait "$pid" && echo "  PID $pid done OK" || echo "  PID $pid FAILED (exit $?)"
    done
    echo "[$(date '+%H:%M:%S')] Variant $variant finished."
}

VARIANTS=("baseline_ov" "yolo_ov" "yolocp_ov")
if [ $# -ge 1 ]; then
    VARIANTS=("$1")
fi

for variant in "${VARIANTS[@]}"; do
    run_variant "$variant"
done

# Merge results per variant
echo ""
echo "=== Summary ==="
SUMMARY="$LOGDIR/summary_ov.log"
> "$SUMMARY"
for variant in "${VARIANTS[@]}"; do
    echo "--- $variant ---" | tee -a "$SUMMARY"
    for s in 0 1 2; do
        logfile="$LOGDIR/${variant}_s${s}.out"
        if [ -f "$logfile" ]; then
            grep -E "SR|SPL|SEMANTIC_COLLISION|MISDETECT|Success Rate|shard" "$logfile" | tail -10 | tee -a "$SUMMARY" || true
        fi
    done
    echo "" | tee -a "$SUMMARY"
done
echo "Summary written to $SUMMARY"
