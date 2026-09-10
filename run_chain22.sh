#!/bin/bash
# chain 22: after exp24c -> exp24d (dynamic/kill) -> exp24e (funding sleeves) -> exp24f (final combos). Harness block later, once the book is chosen.
cd /agent/workspace/proj
while pgrep -f "exp24c_ml8_noag[e]" > /dev/null; do sleep 15; done
echo "=== $(date -u +%H:%M:%S) exp24d"; python3 -u exp24d_dynamic.py > logs/exp24d.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp24e"; python3 -u exp24e_funding_sleeves.py > logs/exp24e.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp24f"; python3 -u exp24f_final_combos.py > logs/exp24f.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
