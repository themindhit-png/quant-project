#!/bin/bash
# chain 15 (master, replaces 12/13/14): wait for the running exp22b, then sequentially:
# exp22c combos -> harness seeds 2,3 (5-sleeve) -> harness 4-sleeve no-listing -> exp21e lag fix -> exp21c age-free ML -> exp21d
cd /agent/workspace/proj
while pgrep -f "exp22b_simplif[y]" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
echo "=== $(date -u +%H:%M:%S) exp22c"; python3 -u exp22c_combos.py > logs/exp22c.log 2>&1
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
for s in 2 3; do
  echo "=== $(date -u +%H:%M:%S) 5-sleeve pure seed $s"
  env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed $s 2>&1 | grep -v "^  20" > logs/h5_seed$s.log
done
echo "=== $(date -u +%H:%M:%S) 4-sleeve (no listing) pure"
env SLEEVES=core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h4nl_pure.log
echo "=== $(date -u +%H:%M:%S) exp21e"; python3 -u exp21e_lag_fix.py > logs/exp21e.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21c (age-free ML retrain)"; python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d"; python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
