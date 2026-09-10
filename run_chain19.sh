#!/bin/bash
# chain 19 (Opus round 2): F pure with NOTIONAL_BASE=equity (2 seeds); pessimistic real paths — taker-only, 1h execution lag,
# both; and the ml8-off (daily age-free) book as the kill-rule scenario. Waits for exp23a.
cd /agent/workspace/proj
while pgrep -f "exp23a_f_check[s]" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F="SLEEVES=listing,core,ml3,ml7,ml8 ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) F pure NOTIONAL_BASE=equity seed 1"
env $F NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF_pure_eq.log
echo "=== $(date -u +%H:%M:%S) F pure NOTIONAL_BASE=equity seed 2"
env $F NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded --seed 2 2>&1 | grep -v "^  20" > logs/hF_pure_eq_s2.log
echo "=== $(date -u +%H:%M:%S) F pure NOTIONAL_BASE=equity, exec lag 1h"
env $F NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded --exec-lag-h 1 2>&1 | grep -v "^  20" > logs/hF_pure_eq_lag1.log
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
R="$HP --firm mubite_8_dd10 --starts $STARTS"
BOLD="BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30"; FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"
echo "=== $(date -u +%H:%M:%S) F rolling bold, TAKER only"
env $F $BOLD $R --maker-fill 0.0 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF_bold20_taker.log
echo "=== $(date -u +%H:%M:%S) F rolling funded, TAKER only"
env $F $FUND $R --maker-fill 0.0 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF_funded125_taker.log
echo "=== $(date -u +%H:%M:%S) F rolling bold, exec lag 1h"
env $F $BOLD $R --maker-fill 0.33 --exec-lag-h 1 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF_bold20_lag1.log
echo "=== $(date -u +%H:%M:%S) F rolling funded, exec lag 1h"
env $F $FUND $R --maker-fill 0.33 --exec-lag-h 1 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF_funded125_lag1.log
echo "=== $(date -u +%H:%M:%S) F rolling bold, TAKER + lag 1h (worst case)"
env $F $BOLD $R --maker-fill 0.0 --exec-lag-h 1 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF_bold20_taker_lag1.log
D="SLEEVES=listing,core,ml3,ml7 ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d REBAL_BAND=0.3 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) ml8-off (daily age-free) rolling bold"
env $D $BOLD $R --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollD_bold20.log
echo "=== $(date -u +%H:%M:%S) ml8-off (daily age-free) rolling funded"
env $D $FUND $R --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollD_funded125.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
