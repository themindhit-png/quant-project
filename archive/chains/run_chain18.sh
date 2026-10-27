#!/bin/bash
# chain 18: final candidate F = listing + core + age-free ML3/ML7 (uni >= 90d) + ML8 (q .2, age >= 180d, inv-vol, w2),
# band .5, smooth .5, direct VT, no beta cap. (1) final age-free models for live; (2) harness pure (2 seeds);
# (3) rolling Mubite: bold / calm / funded; (4) fallback K3-age funded rolling for the survival comparison.
cd /agent/workspace/proj
while pgrep -f "exp22e_noage_combo[s]|harness_v[4].py" > /dev/null; do sleep 15; done
export ACCOUNT_SIZE=10000
[ -f models/ml_h7_noage.txt ] || { echo "=== $(date -u +%H:%M:%S) exp6b_noage_final (deployable age-free models)"; python3 -u exp6b_noage_final.py > logs/exp6b_noage.log 2>&1; }
H="python3 -u v4/harness_v4.py --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage --start 2022-01-01 --maker-fill 0.33"
F="SLEEVES=listing,core,ml3,ml7,ml8 ML_MODEL_SUFFIX=_noage ML_MIN_AGE_D=90 ML_DROP_FEATS=age_d ML8_TOP_FRAC=0.2 REBAL_BAND=0.5 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) F harness pure"
env $F MODE=funded BASE_SCALE=1.0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/hF_pure.log
echo "=== $(date -u +%H:%M:%S) F harness pure seed 2"
env $F MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --seed 2 2>&1 | grep -v "^  20" > logs/hF_pure_s2.log
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
R="$H --firm mubite_8_dd10 --starts $STARTS"
echo "=== $(date -u +%H:%M:%S) F rolling challenge bold (base 2.0 c.25 abandon .3)"
env $F BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF_bold20.log
echo "=== $(date -u +%H:%M:%S) F rolling challenge calm (base 1.0)"
env $F BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/rollF_flat10.log
echo "=== $(date -u +%H:%M:%S) F rolling funded base 1.25 payouts 30d"
env $F BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollF_funded125.log
H2="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --maker-fill 0.33"
K3="SLEEVES=listing,core,ml3,ml7,ml8 ML8_TOP_FRAC=0.2 REBAL_BAND=0.4 FIRM=mubite_8_dd10"
echo "=== $(date -u +%H:%M:%S) K3 (age ML fallback) rolling funded base 1.25"
env $K3 BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0 $H2 --firm mubite_8_dd10 --starts $STARTS --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/rollK3_funded125.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
