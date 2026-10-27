#!/bin/bash
# chain 17 (replaces the tail of chain 16 after exp22d): harness runs incl. the K14 candidate (no listing, ml8 q.2, band .5,
# beta-neutral), rolling challenge/funded for K14, exp21e, exp21c (age-free ML), exp21d
cd /agent/workspace/proj
while pgrep -f "exp22d_cost_robus[t]" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
K14="SLEEVES=core,ml3,ml7,ml8 ML8_TOP_FRAC=0.2 REBAL_BAND=0.5 BETA_CAP=0 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) K14 harness pure"
env $K14 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hK14_pure.log
echo "=== $(date -u +%H:%M:%S) K14 harness pure seed 2"
env $K14 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed 2 2>&1 | grep -v "^  20" > logs/hK14_pure_s2.log
echo "=== $(date -u +%H:%M:%S) K3 harness pure (listing kept, q.2 band .4)"
env SLEEVES=listing,core,ml3,ml7,ml8 ML8_TOP_FRAC=0.2 REBAL_BAND=0.4 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hK3_pure.log
echo "=== $(date -u +%H:%M:%S) 5-sleeve baseline pure seed 2"
env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed 2 2>&1 | grep -v "^  20" > logs/h5_seed2.log
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
R="$H --firm mubite_8_dd10 --starts $STARTS"
echo "=== $(date -u +%H:%M:%S) K14 rolling challenge bold (base 2.0 c.25 abandon .3)"
env $K14 BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollK14_bold20.log
echo "=== $(date -u +%H:%M:%S) K14 rolling challenge calm (base 1.0)"
env $K14 BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollK14_flat10.log
echo "=== $(date -u +%H:%M:%S) K14 rolling funded base 1.25 payouts 30d"
env $K14 BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollK14_funded125.log
echo "=== $(date -u +%H:%M:%S) exp21e"; python3 -u exp21e_lag_fix.py > logs/exp21e.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21c (age-free ML retrain)"; python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d"; python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
