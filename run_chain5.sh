#!/bin/bash
# chain 5 (strictly sequential, one harness process at a time):
#  A) 5-sleeve execution sensitivity: taker-only, maker 0.5, no short stop
#  B) rolling real-path simulations of the 5-sleeve book (Mubite 8+5 / DD10): challenge calm/flat15/bold20, funded 1.25/1.5
cd /agent/workspace/proj
export ACCOUNT_SIZE=10000
H="python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01"
S5="SLEEVES=listing,core,ml3,ml7,ml8"
echo "=== $(date -u +%H:%M:%S) A1 5-sleeve taker-only"
env $S5 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --maker-fill 0.0 2>&1 | grep -v "^  20" > logs/h5_taker.log
echo "=== $(date -u +%H:%M:%S) A2 5-sleeve maker 0.5"
env $S5 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 $H --pure --mode funded --maker-fill 0.5 2>&1 | grep -v "^  20" > logs/h5_maker50.log
echo "=== $(date -u +%H:%M:%S) A3 5-sleeve no short stop"
env $S5 FIRM=mubite_8_dd10 MODE=funded BASE_SCALE=1.0 STOP_SHORT_PCT=0 $H --pure --mode funded 2>&1 | grep -v "^  20" > logs/h5_nostop.log
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
R="$H --firm mubite_8_dd10 --starts $STARTS"
echo "=== $(date -u +%H:%M:%S) B1 challenge flat base1.0 c0.5 (calm)"
env $S5 BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_flat10.log
echo "=== $(date -u +%H:%M:%S) B2 challenge base1.5 c0.5 abandon30"
env $S5 BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_flat15.log
echo "=== $(date -u +%H:%M:%S) B3 challenge base2.0 c0.25 abandon30"
env $S5 BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 $R --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_bold20.log
echo "=== $(date -u +%H:%M:%S) B4 funded base1.25 c0.5 payouts 30d"
env $S5 BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/roll5_funded125.log
echo "=== $(date -u +%H:%M:%S) B5 funded base1.5 c0.5 payouts 30d"
env $S5 BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0 $R --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/roll5_funded15.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
