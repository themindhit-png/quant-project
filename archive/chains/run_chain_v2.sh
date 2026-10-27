#!/bin/bash
# Round-4 final chain: everything re-measured with the kill rule v2 (net-of-cost shadow, 0.5/0 thresholds) and the low-cap proxy.
cd /agent/workspace/proj
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
BOLD="BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30"; MOD="BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; MOD125="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"; FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"
run() { local name=$1 firm=$2 phases=$3 scale=$4; shift 4
  echo "=== $(date -u +%H:%M:%S) $name"; env $F2 FIRM=$firm $scale $HP --firm $firm --starts $STARTS --maker-fill 0.33 --mode challenge --phases $phases --max-days 700 "$@" 2>&1 | grep -v "^  20" > logs/$name.log; }
run rollF2v2_bold20 mubite_8_dd10 2 "$BOLD"
echo "=== $(date -u +%H:%M:%S) funded v2"; env $F2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2v2_funded125.log
run rollF2v2_hyro2swing_mod15 hyrotrader_2step_swing 2 "$MOD"
run rollF2v2_hyro2swing_bold20 hyrotrader_2step_swing 2 "$BOLD"
run rollF2v2_hyro1swing_mod125 hyrotrader_1step_swing 1 "$MOD125"
run rollF2v2_ptb_calm10 ptb_bybit 1 "$CALM"
run rollF2v2_ptb_mod125 ptb_bybit 1 "$MOD125"
run rollF2v2_hyro2_mod15 hyrotrader_2step 2 "$MOD"
run rollF2v2_flat10 mubite_8_dd10 2 "$CALM"
echo "=== $(date -u +%H:%M:%S) ALL DONE"
