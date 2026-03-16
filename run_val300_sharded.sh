#!/bin/bash
# Sharded val_300 eval: 3 baseline shards, then 3 yolo shards (3 GPU slots each).
# Launch: nohup bash run_val300_sharded.sh > log/val300_sharded_TIMESTAMP.out 2>&1 &
set -e
cd /opt/data/private/onemap/OneMap_obstacle
LOG_DIR="log"

echo "========================================"
echo "val_300 sharded experiments: $(date)"
echo "========================================"

# Helper: find the most recently written eval_*.log and rename it
rename_latest_log() {
    local new_name="$1"
    local latest
    latest=$(ls -t ${LOG_DIR}/eval_*.log 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        cp "$latest" "${LOG_DIR}/${new_name}"
        echo "  -> log saved as ${new_name}"
    fi
}

# ── Phase 1: 3 baseline shards in parallel ──────────────────────────
echo ""
echo "[Phase 1/2] Launching 3 BASELINE shards in parallel ..."
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_baseline_s0.yaml > ${LOG_DIR}/val300_baseline_s0_stdout.out 2>&1 &
PID0=$!
sleep 5
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_baseline_s1.yaml > ${LOG_DIR}/val300_baseline_s1_stdout.out 2>&1 &
PID1=$!
sleep 5
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_baseline_s2.yaml > ${LOG_DIR}/val300_baseline_s2_stdout.out 2>&1 &
PID2=$!

echo "  PIDs: $PID0 $PID1 $PID2 — waiting ..."
wait $PID0; rename_latest_log "val300_baseline_s0_$(date +%Y%m%d_%H%M%S).log"
wait $PID1; rename_latest_log "val300_baseline_s1_$(date +%Y%m%d_%H%M%S).log"
wait $PID2; rename_latest_log "val300_baseline_s2_$(date +%Y%m%d_%H%M%S).log"
echo "[Phase 1/2] BASELINE done: $(date)"

# ── Phase 2: 3 yolo shards in parallel ──────────────────────────────
echo ""
echo "[Phase 2/2] Launching 3 YOLO shards in parallel ..."
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_yolo_s0.yaml > ${LOG_DIR}/val300_yolo_s0_stdout.out 2>&1 &
PID0=$!
sleep 5
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_yolo_s1.yaml > ${LOG_DIR}/val300_yolo_s1_stdout.out 2>&1 &
PID1=$!
sleep 5
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_yolo_s2.yaml > ${LOG_DIR}/val300_yolo_s2_stdout.out 2>&1 &
PID2=$!

echo "  PIDs: $PID0 $PID1 $PID2 — waiting ..."
wait $PID0; rename_latest_log "val300_yolo_s0_$(date +%Y%m%d_%H%M%S).log"
wait $PID1; rename_latest_log "val300_yolo_s1_$(date +%Y%m%d_%H%M%S).log"
wait $PID2; rename_latest_log "val300_yolo_s2_$(date +%Y%m%d_%H%M%S).log"
echo "[Phase 2/2] YOLO done: $(date)"

# ── Combine ──────────────────────────────────────────────────────────
echo ""
echo "Combining results ..."
conda run -n onemap python combine_val300_results.py

echo ""
echo "All done: $(date)"
