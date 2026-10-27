#!/bin/bash
# chain 24: unlock-calendar constraint test after the final combos experiment (memory: one heavy process at a time)
cd /agent/workspace/proj
while pgrep -f "exp24f_final_combo[s]" > /dev/null; do sleep 15; done
echo "=== $(date -u +%H:%M:%S) exp25b"; python3 -u exp25b_unlock_filter.py > logs/exp25b.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
