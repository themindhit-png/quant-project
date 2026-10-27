#!/bin/bash
# chain 10 (strictly sequential, one heavy process at a time):
#  1 exp21a Opus battery  2 exp21b 2021 regime  3 exp22 squeeze  4 harness diagnostics (notional base, funded vs pure)
#  5 exp21c age-free ML retrain (~1h)  6 exp21d its portfolio impact
cd /agent/workspace/proj
export ACCOUNT_SIZE=10000
echo "=== $(date -u +%H:%M:%S) exp21a"; python3 -u exp21a_opus_tests.py > logs/exp21a.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21b"; python3 -u exp21b_2021.py > logs/exp21b.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp22";  python3 -u exp22_squeeze.py > logs/exp22.log 2>&1
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
echo "=== $(date -u +%H:%M:%S) H1 5-sleeve pure NOTIONAL_BASE=initial"
env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=initial $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h5_nb_initial.log
echo "=== $(date -u +%H:%M:%S) H2 5-sleeve pure NOTIONAL_BASE=equity"
env SLEEVES=listing,core,ml3,ml7,ml8 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=equity $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h5_nb_equity.log
echo "=== $(date -u +%H:%M:%S) H3 daily pure NOTIONAL_BASE=equity"
env FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=equity $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hd_nb_equity.log
echo "=== $(date -u +%H:%M:%S) H4 daily funded preset hyrotrader_2step_swing (BASE 1.0, CPPI .5), fixed VT"
env FIRM=hyrotrader_2step_swing MODE=funded BASE_SCALE=1.0 CPPI_FRAC=0.5 $H --mode funded --firm hyrotrader_2step_swing 2>&1 | grep -v "^  20" > logs/hd_swing_funded.log
echo "=== $(date -u +%H:%M:%S) H5 daily pure hyrotrader_2step_swing NOTIONAL_BASE=initial (for the pure-vs-funded correlation)"
env FIRM=hyrotrader_2step_swing MODE=funded BASE_SCALE=1.0 NOTIONAL_BASE=initial $H --pure --mode funded --firm hyrotrader_2step_swing 2>&1 | grep -v "^  20" > logs/hd_swing_pure.log
echo "=== $(date -u +%H:%M:%S) exp21c (age-free ML retrain)"; python3 -u exp21c_ml_noage.py > logs/exp21c.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21d"; python3 -u exp21d_noage_portfolio.py > logs/exp21d.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
