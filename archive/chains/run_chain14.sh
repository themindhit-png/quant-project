#!/bin/bash
# chain 14 (after chain 13): corrected lag test + beta neutralisation windows + concentration
cd /agent/workspace/proj
while pgrep -f "run_chain1[23].sh" > /dev/null; do sleep 30; done
echo "=== $(date -u +%H:%M:%S) exp21e"; python3 -u exp21e_lag_fix.py > logs/exp21e.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
