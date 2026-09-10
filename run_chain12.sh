#!/bin/bash
# chain 12 (replaces chains 10/11): wait for the running harness (H1) to finish, then strictly sequential:
# H2..H5 harness diagnostics -> exp21a (Opus battery) -> exp22 (squeeze) -> exp22b (simplify) -> exp21c (age-free ML) -> exp21d
cd /agent/workspace/proj
while pgrep -f "harness_v[4].py" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
echo "=== $(date -u +%H:%M:%S) H2 5-sleeve pure NOTIONAL_BASE=equity"
env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=equity $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h5_nb_equity.log
echo "=== $(date -u +%H:%M:%S) H3 daily pure NOTIONAL_BASE=equity"
env FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=equity $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hd_nb_equity.log
echo "=== $(date -u +%H:%M:%S) H4 daily funded preset hyrotrader_2step_swing (BASE 1.0, CPPI .5)"
env FIRM=hyrotrader_2step_swing MODE=funded BASE_SCALE=1.0 CPPI_FRAC=0.5 $H --mode funded --firm hyrotrader_2step_swing 2>&1 | grep -v "^  20" > logs/hd_swing_funded.log
echo "=== $(date -u +%H:%M:%S) H5 daily pure hyrotrader_2step_swing NOTIONAL_BASE=initial"
env FIRM=hyrotrader_2step_swing MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=initial $H --pure --mode funded --firm hyrotrader_2step_swing 2>&1 | grep -v "^  20" > logs/hd_swing_pure.log
echo "=== $(date -u +%H:%M:%S) exp21a"; python3 -u exp21a_opus_tests.py > logs/exp21a.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp22";  python3 -u exp22_squeeze.py > logs/exp22.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp22b"; python3 -u exp22b_simplify.py > logs/exp22b.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21c (age-free ML retrain)"; python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d"; python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
