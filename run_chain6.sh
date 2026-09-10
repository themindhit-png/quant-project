#!/bin/bash
# chain 6: reconciliation harness vs research — per-attempt maker fill 0.33 (3 attempts -> ~70% effective maker share,
# as in the research engine's maker_share=0.7). Waits for chain5 (one harness process at a time).
cd /agent/workspace/proj
while pgrep -f "run_chain[5].sh" > /dev/null; do sleep 30; done
export ACCOUNT_SIZE=10000
echo "=== $(date -u +%H:%M:%S) 5-sleeve maker-fill 0.33 (eff ~70%)"
env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --pure --mode funded --maker-fill 0.33 2>&1 | grep -v "^  20" > logs/h5_maker33.log
echo "=== $(date -u +%H:%M:%S) daily maker-fill 0.33 (eff ~70%)"
env FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --pure --mode funded --maker-fill 0.33 2>&1 | grep -v "^  20" > logs/hd_maker33.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
