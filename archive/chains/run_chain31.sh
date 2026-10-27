#!/bin/bash
# chain 31 (after chain 30): funded keep-buffer variants (payout-day cushion), Mubite default preset funded, D2 funded taker,
# D2 calm challenge — Opus P2.28 / P2.34.
cd /agent/workspace/proj
while pgrep -f "run_chain3[0].sh|harness_v[4].py" > /dev/null; do sleep 20; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
D2="SLEEVES=listing,core,ml3,ml7,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d REBAL_BAND=0.3"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"
echo "=== $(date -u +%H:%M:%S) 1 funded keep-buffer 0.03"; env $F2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.03 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_kb03.log
echo "=== $(date -u +%H:%M:%S) 2 funded keep-buffer 0.05"; env $F2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.05 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_kb05.log
echo "=== $(date -u +%H:%M:%S) 3 funded Mubite DEFAULT preset (DD8)"; env $F2 FIRM=mubite_2step $FUND $HP --firm mubite_2step --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_mubite2step_funded125.log
echo "=== $(date -u +%H:%M:%S) 4 funded HyroTrader 1-step preset (6/4 trailing) calm"; env $F2 FIRM=hyrotrader_1step $CALM $HP --firm hyrotrader_1step --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_hyro1_funded10.log
echo "=== $(date -u +%H:%M:%S) 5 D2 funded TAKER"; env $D2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.0 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollD2_funded125_taker.log
echo "=== $(date -u +%H:%M:%S) 6 D2 calm challenge"; env $D2 FIRM=mubite_8_dd10 $CALM $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollD2_flat10.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
