#!/bin/bash
# chain 25: validation of the final book F2 = listing + core + age-free ML3/ML7 + ML8 (q.2) + fchg + fcarry (band .5) in the bot:
# pure (equity-based cap, 2 seeds, kill rule on/off, 1h lag), rolling real paths (bold/calm/funded), pessimistic ladder
# (taker-only, 1h lag, both), and the ml8-off book (kill-rule scenario).
cd /agent/workspace/proj
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) F2 pure eq-base seed 1 (kill rule on)"
env $F2 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq.log
echo "=== $(date -u +%H:%M:%S) F2 pure eq-base seed 2"
env $F2 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded --seed 2 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_s2.log
echo "=== $(date -u +%H:%M:%S) F2 pure eq-base, kill rule OFF (research comparability)"
env $F2 NOTIONAL_BASE=equity ML8_KILL_RULE=0 MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_nokill.log
echo "=== $(date -u +%H:%M:%S) F2 pure eq-base, exec lag 1h"
env $F2 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.33 --pure --mode funded --exec-lag-h 1 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_lag1.log
echo "=== $(date -u +%H:%M:%S) F2 pure eq-base, TAKER"
env $F2 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0 $HP --maker-fill 0.0 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_taker.log
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
R="$HP --firm mubite_8_dd10 --starts $STARTS"
BOLD="BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30"; CALM="BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0"; FUND="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0"
echo "=== $(date -u +%H:%M:%S) F2 rolling bold"
env $F2 $BOLD $R --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_bold20.log
echo "=== $(date -u +%H:%M:%S) F2 rolling calm"
env $F2 $CALM $R --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_flat10.log
echo "=== $(date -u +%H:%M:%S) F2 rolling funded 1.25"
env $F2 $FUND $R --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125.log
echo "=== $(date -u +%H:%M:%S) F2 rolling bold TAKER"
env $F2 $BOLD $R --maker-fill 0.0 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_bold20_taker.log
echo "=== $(date -u +%H:%M:%S) F2 rolling funded TAKER"
env $F2 $FUND $R --maker-fill 0.0 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_taker.log
echo "=== $(date -u +%H:%M:%S) F2 rolling bold lag 1h"
env $F2 $BOLD $R --maker-fill 0.33 --exec-lag-h 1 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_bold20_lag1.log
echo "=== $(date -u +%H:%M:%S) F2 rolling funded lag 1h"
env $F2 $FUND $R --maker-fill 0.33 --exec-lag-h 1 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF2_funded125_lag1.log
echo "=== $(date -u +%H:%M:%S) F2 rolling bold TAKER + lag 1h (worst case)"
env $F2 $BOLD $R --maker-fill 0.0 --exec-lag-h 1 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_bold20_taker_lag1.log
D2="SLEEVES=listing,core,ml3,ml7,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d REBAL_BAND=0.3 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) D2 (ml8 off) rolling bold"
env $D2 $BOLD $R --maker-fill 0.33 --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollD2_bold20.log
echo "=== $(date -u +%H:%M:%S) D2 (ml8 off) rolling funded"
env $D2 $FUND $R --maker-fill 0.33 --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollD2_funded125.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
