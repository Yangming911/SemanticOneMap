#!/bin/bash
WD="/inspire/hdd/global_user/wangshuqi-253208110272/SemanticOneMap"
RESULTS_DIR="$1"
TOTAL_EP="${2:-330}"

while true; do
    n_state=$(ls $RESULTS_DIR/state/state_*.txt 2>/dev/null | wc -l)
    n_sessions=$(tmux ls 2>/dev/null | wc -l)
    
    if [ "$n_state" -ge "$TOTAL_EP" ]; then
        echo "$(date +%H:%M:%S) COMPLETE: $n_state/$TOTAL_EP episodes"
        break
    fi
    
    if [ "$n_sessions" -eq 0 ] && [ "$n_state" -lt "$TOTAL_EP" ]; then
        # All sessions dead but not complete - find missing
        missing=$(python3 -c "
import os
existing = {int(f[6:-4]) for f in os.listdir('$RESULTS_DIR/state/') if f.startswith('state_') and f.endswith('.txt')}
missing = sorted(set(range($TOTAL_EP)) - existing)
if missing: print(f'MISSING {len(missing)} eps: {missing[:10]}...' if len(missing)>10 else f'MISSING {len(missing)} eps: {missing}')
else: print('NONE')
")
        echo "$(date +%H:%M:%S) ALL SESSIONS DEAD | $n_state/$TOTAL_EP | $missing"
        break
    fi
    
    echo "$(date +%H:%M:%S) $n_state/$TOTAL_EP | sessions=$n_sessions"
    sleep 60
done
