#!/bin/bash
# chain 13 (after chain 12): harness dispersion across fill-RNG seeds and the simplified book (no listing sleeve)
cd /agent/workspace/proj
while pgrep -f "run_chain1[2].sh" > /dev/null; do sleep 30; done
export ACCOUNT_SIZE=10000
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
for s in 2 3; do
  echo "=== $(date -u +%H:%M:%S) 5-sleeve pure seed $s"
  env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed $s 2>&1 | grep -v "^  20" > logs/h5_seed$s.log
done
echo "=== $(date -u +%H:%M:%S) 4-sleeve (no listing) pure seed 1"
env SLEEVES=core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h4nl_pure.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
