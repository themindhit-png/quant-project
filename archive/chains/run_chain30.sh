#!/bin/bash
# chain 30: final measurements after the Opus round-3 fixes. (1) pure F2 regression (eq-base, kill rule), (2) rolling
# challenge on the DEFAULT Mubite 2-step (8% static DD, 10+5%), (3) HyroTrader 1-step (6% / 4% trailing), (4) rolling funded
# F2 with the payout-as-cash-flow fix, (5) HyroTrader 2-step (10/5 trailing), (6) Mubite moderate, (7) Hyro 1-step calm.
cd /agent/workspace/proj
while pgrep -f "exp27_firm_rule[s]|exp26_opus[3]" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
BOLD="BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30"; MOD="BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; MOD125="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"; FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"
echo "=== $(date -u +%H:%M:%S) 1 F2 pure eq-base (post-fix regression)"; env $F2 FIRM=mubite_8_dd10 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --firm mubite_8_dd10 --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_v3.log
echo "=== $(date -u +%H:%M:%S) 2 rolling Mubite 2-step DEFAULT (DD8, 10+5) bold"; env $F2 FIRM=mubite_2step $BOLD $HP --firm mubite_2step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_mubite2step_bold20.log
echo "=== $(date -u +%H:%M:%S) 3 rolling HyroTrader 1-step (6/4 trailing) moderate 1.25"; env $F2 FIRM=hyrotrader_1step $MOD125 $HP --firm hyrotrader_1step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 1 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_hyro1_mod125.log
echo "=== $(date -u +%H:%M:%S) 4 rolling funded F2 (payout = cash flow fix)"; env $F2 FIRM=mubite_8_dd10 $FUND $HP --firm mubite_8_dd10 --starts $STARTS --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_v3.log
echo "=== $(date -u +%H:%M:%S) 5 rolling HyroTrader 2-step (10/5 trailing) bold"; env $F2 FIRM=hyrotrader_2step $BOLD $HP --firm hyrotrader_2step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_hyro2_bold20.log
echo "=== $(date -u +%H:%M:%S) 6 rolling Mubite 2-step DEFAULT moderate 1.5"; env $F2 FIRM=mubite_2step $MOD $HP --firm mubite_2step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_mubite2step_mod15.log
echo "=== $(date -u +%H:%M:%S) 7 rolling HyroTrader 1-step calm 1.0"; env $F2 FIRM=hyrotrader_1step $CALM $HP --firm hyrotrader_1step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 1 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_hyro1_calm10.log
echo "=== $(date -u +%H:%M:%S) 8 rolling HyroTrader 1-step SWING-like static (6/4 static) moderate 1.25"; env $F2 FIRM=hyrotrader_1step_swing $MOD125 $HP --firm hyrotrader_1step_swing --starts $STARTS --maker-fill 0.33 --mode challenge --phases 1 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_hyro1swing_mod125.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
