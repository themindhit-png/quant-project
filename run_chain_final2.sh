#!/bin/bash
# FINAL chain, part 2 (after run_chain_final.sh): HyroTrader 2-step at moderate/calm scale (bold halts on the trailing daily
# DD), swing at moderate, PTB calm, Mubite add-on preset at moderate.
cd /agent/workspace/proj
while pgrep -f "run_chain_fina[l].sh|harness_v[4].py" > /dev/null; do sleep 20; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
MOD="BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"
run() { local name=$1 firm=$2 phases=$3 scale=$4; shift 4
  echo "=== $(date -u +%H:%M:%S) $name"; env $F2 FIRM=$firm $scale $HP --firm $firm --starts $STARTS --maker-fill 0.33 --mode challenge --phases $phases --max-days 700 "$@" 2>&1 | grep -v "^  20" > logs/$name.log; }
run rollF2_hyro2_mod15 hyrotrader_2step 2 "$MOD"
run rollF2_hyro2_calm10 hyrotrader_2step 2 "$CALM"
run rollF2_hyro2swing_mod15 hyrotrader_2step_swing 2 "$MOD"
run rollF2_ptb_calm10 ptb_bybit 1 "$CALM"
run rollF2_dd10_mod15 mubite_8_dd10 2 "$MOD"
echo "=== $(date -u +%H:%M:%S) ALL DONE"
