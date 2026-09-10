#!/bin/bash
# chain 16 (master, replaces 15): wait for exp22c, then exp22d (cost robustness) -> harness runs (seeds, simplified books)
# -> exp21e (lag fix) -> exp21c (age-free ML, ~1h) -> exp21d
cd /agent/workspace/proj
while pgrep -f "exp22c_combo[s]" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
echo "=== $(date -u +%H:%M:%S) exp22d"; python3 -u exp22d_cost_robust.py > logs/exp22d.log 2>&1
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
for s in 2 3; do
  echo "=== $(date -u +%H:%M:%S) 5-sleeve pure seed $s"
  env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed $s 2>&1 | grep -v "^  20" > logs/h5_seed$s.log
done
echo "=== $(date -u +%H:%M:%S) K7 harness: core,ml3,ml7,ml8 q.2 band .4"
env SLEEVES=core,ml3,ml7,ml8 ML8_TOP_FRAC=0.2 REBAL_BAND=0.4 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hK7_pure.log
echo "=== $(date -u +%H:%M:%S) R3 harness: core,ml3,ml8 q.2 band .4 (scales core:1,ml3:1,ml8:2)"
env SLEEVES=core,ml3,ml8 SLEEVE_SCALES=core:1,ml3:1,ml8:2 ML8_TOP_FRAC=0.2 REBAL_BAND=0.4 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hR3_pure.log
echo "=== $(date -u +%H:%M:%S) exp21e"; python3 -u exp21e_lag_fix.py > logs/exp21e.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21c (age-free ML retrain)"; python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d"; python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
