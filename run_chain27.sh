#!/bin/bash
# chain 27: re-run the kill-rule pure tests with the corrected (marginal, signed) attribution. Waits for chain 26.
cd /agent/workspace/proj
while pgrep -f "run_chain2[6].sh" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
HP="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01"
F2="SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5 FIRM=mubite_8_dd10 NOTIONAL_BASE=equity MODE=funded BASE_SCALE=1.0"
echo "=== $(date -u +%H:%M:%S) F2 pure, kill 90d (signed attribution)"
env $F2 $HP --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_kill90_v2.log
echo "=== $(date -u +%H:%M:%S) F2 pure, kill 180d hard -1 restore 60 (signed attribution)"
env $F2 ML8_KILL_WIN_D=180 ML8_KILL_HARD=-1.0 ML8_KILL_RESTORE_D=60 $HP --maker-fill 0.33 --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF2_pure_eq_kill180soft_v2.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
