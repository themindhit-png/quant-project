#!/bin/bash
# chain 11: exp22b (simplified books + alt-season overlay) after chain10 finishes (memory: one heavy process at a time)
cd /agent/workspace/proj
while pgrep -f "run_chain1[0].sh" > /dev/null; do sleep 30; done
echo "=== $(date -u +%H:%M:%S) exp22b"; python3 -u exp22b_simplify.py > logs/exp22b.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
