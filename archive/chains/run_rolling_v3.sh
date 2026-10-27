#!/bin/sh
# 1) Bybit-feasible regression ML (v3r_bybit) + portfolio impact; 2) bold-policy rolling sims on v2 panels;
# 3) rolling sims on v3r_bybit panels. Writes ALL DONE to logs/rolling_v3.log (exp17 waits for it).
cd /agent/workspace/proj
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
echo "=== $(date -u +%H:%M:%S) exp16b-r (bybit features, regression)" >> logs/rolling_v3.log
FEATSET=bybit VARIANTS=r SUFFIX=_bybit python3 -u exp16_ml_v3.py > logs/exp16b_r_bybit.log 2>&1
python3 -u exp16c_portfolio.py > logs/exp16c2.log 2>&1
P2="--ml-panels"
echo "=== $(date -u +%H:%M:%S) v2 bold base2.0 c0.25 abandon30" >> logs/rolling_v3.log
BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py $P2 --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_v2_bold20.log 2>&1
echo "=== $(date -u +%H:%M:%S) v2 base1.5 c0.5 abandon30" >> logs/rolling_v3.log
BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py $P2 --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_v2_flat15.log 2>&1
P3="--ml-panels --panels-prefix out_exp16_pred --panels-suffix _r_bybit"
if [ -f out_exp16_pred_h3_r_bybit.npy ]; then
  echo "=== $(date -u +%H:%M:%S) v3r_bybit flat base1.0" >> logs/rolling_v3.log
  python3 -u v4/harness_v4.py $P3 --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_v3r_flat10.log 2>&1
  echo "=== $(date -u +%H:%M:%S) v3r_bybit bold base2.0 c0.25 abandon30" >> logs/rolling_v3.log
  BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py $P3 --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_v3r_bold20.log 2>&1
fi
echo "=== $(date -u +%H:%M:%S) ALL DONE" >> logs/rolling_v3.log
