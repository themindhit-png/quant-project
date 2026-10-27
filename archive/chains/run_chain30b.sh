#!/bin/bash
# chain 30b (after chains 30/31): redo the HyroTrader 1-step rolling that crashed on the harness FakeMD (mcap), then the
# personal-account research runs (exp28: F2 at higher vol targets, no firm rules).
cd /agent/workspace/proj
while pgrep -f "run_chain3[01].sh|harness_v[4].py" > /dev/null; do sleep 20; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5"
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
MOD125="BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0.30"
echo "=== $(date -u +%H:%M:%S) 3b rolling HyroTrader 1-step (6/4 trailing) moderate 1.25"; env $F2 FIRM=hyrotrader_1step $MOD125 $HP --firm hyrotrader_1step --starts $STARTS --maker-fill 0.33 --mode challenge --phases 1 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF2_hyro1_mod125.log
echo "=== $(date -u +%H:%M:%S) exp28 personal account"; python3 -u exp28_personal.py > logs/exp28.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
