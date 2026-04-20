#!/usr/bin/env bash
# Generate visualization videos for all 33 mini-val MP3D episodes using V5e config.
# Output: outputs/v5e_all/<...>_ep<N>_<date>.mp4
# Log:    results/v5e_vis_all.log

set -u
set +e  # continue on per-episode errors

CFG="config/mon/eval_conf_mp3d_wgate_mp3d_v5e_full_mini.yaml"
OUT_DIR="outputs/v5e_all"
LOG="results/v5e_vis_all.log"

mkdir -p "$OUT_DIR"
: > "$LOG"

for ep in $(seq 0 32); do
    echo "=== [$(date '+%H:%M:%S')] Episode $ep ===" | tee -a "$LOG"
    PYTORCH_NO_NVML=1 xvfb-run -a python -u visualize_single_scene.py \
        --episode-id "$ep" \
        --no-display \
        --output "$OUT_DIR/v5e.mp4" \
        -c "$CFG" 2>&1 | tee -a "$LOG"
    echo "=== [$(date '+%H:%M:%S')] Episode $ep done ===" | tee -a "$LOG"
done

echo "All episodes processed. Videos in $OUT_DIR" | tee -a "$LOG"
