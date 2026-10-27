#!/bin/bash
# FINAL measurement chain (priority order). Waits for any running harness first.
# 1 HyroTrader 2-step (10/5, both trailing) bold   2 HyroTrader 2-step SWING (static daily) bold   3 Mubite default (DD8) moderate
# 4 PTB Bybit (5% target / 5% static, no daily) bold   5 PTB moderate 1.25   6 HyroTrader 1-step (6/4) moderate   7 exp28 personal
# 8 keep-buffer 0.03 funded   9 D2 funded taker   10 Mubite default funded   11 HyroTrader 1-step calm   12 HyroTrader 1-step swing
cd /agent/workspace/proj
while pgrep -f "harness_v[4].py" > /dev/null; do sleep 20; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
D2="SLEEVES=listing,core,ml3,ml7,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d REBAL_BAND=0.3"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
BOLD="BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30"; MOD="BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; MOD125="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"; FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"
run() { # name firm phases envmode extra...
  local name=$1 firm=$2 phases=$3 scale=$4; shift 4
  echo "=== $(date -u +%H:%M:%S) $name"; env $F2 FIRM=$firm $scale $HP --firm $firm --starts $STARTS --maker-fill 0.33 --mode challenge --phases $phases --max-days 700 "$@" 2>&1 | grep -v "^  20" > logs/$name.log
}
run rollF2_hyro2_bold20 hyrotrader_2step 2 "$BOLD"
run rollF2_hyro2swing_bold20 hyrotrader_2step_swing 2 "$BOLD"
run rollF2_mubite2step_mod15 mubite_2step 2 "$MOD"
run rollF2_ptb_bold20 ptb_bybit 1 "$BOLD"
run rollF2_ptb_mod125 ptb_bybit 1 "$MOD125"
run rollF2_hyro1_mod125 hyrotrader_1step 1 "$MOD125"
echo "=== $(date -u +%H:%M:%S) exp28 personal"; python3 -u exp28_personal.py > logs/exp28.log 2>&1
echo "=== $(date -u +%H:%M:%S) funded keep-buffer 0.03"; env $F2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.03 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_kb03.log
echo "=== $(date -u +%H:%M:%S) D2 funded TAKER"; env $D2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.0 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollD2_funded125_taker.log
echo "=== $(date -u +%H:%M:%S) Mubite default funded"; env $F2 FIRM=mubite_2step $FUND $HP --firm mubite_2step --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_mubite2step_funded125.log
run rollF2_hyro1_calm10 hyrotrader_1step 1 "$CALM"
run rollF2_hyro1swing_mod125 hyrotrader_1step_swing 1 "$MOD125"
echo "=== $(date -u +%H:%M:%S) ALL DONE"
