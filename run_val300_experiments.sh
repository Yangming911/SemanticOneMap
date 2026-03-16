#!/bin/bash
# Run baseline and YOLO val_300 experiments sequentially.
# Launch with: nohup bash run_val300_experiments.sh > log/val300_run.out 2>&1 &
set -e
cd /opt/data/private/onemap/OneMap_obstacle

TS=$(date +%Y%m%d_%H%M%S)
echo "========================================" | tee -a log/val300_run_${TS}.log
echo "val_300 experiment sequence started: $(date)" | tee -a log/val300_run_${TS}.log
echo "========================================" | tee -a log/val300_run_${TS}.log

echo "" | tee -a log/val300_run_${TS}.log
echo "[1/2] Running BASELINE (yolo=False) ..." | tee -a log/val300_run_${TS}.log
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_baseline.yaml \
    2>&1 | tee -a log/val300_run_${TS}.log
echo "[1/2] BASELINE done: $(date)" | tee -a log/val300_run_${TS}.log

echo "" | tee -a log/val300_run_${TS}.log
echo "[2/2] Running YOLO (yolo=True, window=50) ..." | tee -a log/val300_run_${TS}.log
conda run -n onemap xvfb-run -a python -u eval_habitat.py \
    -c config/mon/eval_conf_val300_yolo.yaml \
    2>&1 | tee -a log/val300_run_${TS}.log
echo "[2/2] YOLO done: $(date)" | tee -a log/val300_run_${TS}.log

echo "" | tee -a log/val300_run_${TS}.log
echo "All experiments finished: $(date)" | tee -a log/val300_run_${TS}.log
