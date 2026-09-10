#!/bin/bash
# chain 8: age-free ML retrain (exp21c, ~1h) then its portfolio impact (exp21d). Waits for chain7 (memory).
cd /agent/workspace/proj
while pgrep -f "run_chain[7].sh" > /dev/null; do sleep 30; done
echo "=== $(date -u +%H:%M:%S) exp21c (retrain without age_d, universe >= 90d)"
python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d (portfolio impact)"
python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
