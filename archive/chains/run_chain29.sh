#!/bin/bash
# chain 29: exp26 (Opus round 3 research) after the float64 re-runs of exp24f/exp25b
cd /agent/workspace/proj
while pgrep -f "exp24f_final_combo[s]|exp25b_unlock_filte[r]" > /dev/null; do sleep 15; done
echo "=== $(date -u +%H:%M:%S) exp26"; python3 -u exp26_opus3.py > logs/exp26.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
