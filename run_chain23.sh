#!/bin/bash
# chain 23: token-unlock sleeve research after chain 22 (memory: one heavy process at a time)
cd /agent/workspace/proj
while pgrep -f "run_chain2[2].sh" > /dev/null; do sleep 20; done
echo "=== $(date -u +%H:%M:%S) exp25"; python3 -u exp25_unlocks.py > logs/exp25.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
