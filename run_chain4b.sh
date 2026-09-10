#!/bin/bash
# chain 4b: rolling real-path simulations of the 5-sleeve book (ml8 age180+IV, fixed VT) — Mubite 8+5 / DD10.
# waits for chain4a to finish (memory: one harness process at a time)
cd /agent/workspace/proj
while pgrep -f "run_chain4[a].sh" > /dev/null; do sleep 20; done
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
export SLEEVES=listing,core,ml3,ml7,ml8 ACCOUNT_SIZE=10000
P="--ml-panels --start 2022-01-01 --firm mubite_8_dd10 --starts $STARTS"
echo "=== $(date -u +%H:%M:%S) challenge flat base1.0 c0.5 (calm)"
BASE_SCALE=1.0 CPPI_FRAC=0.5 ABANDON_FRAC=0 python3 -u v4/harness_v4.py $P --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_flat10.log
echo "=== $(date -u +%H:%M:%S) challenge base1.5 c0.5 abandon30"
BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py $P --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_flat15.log
echo "=== $(date -u +%H:%M:%S) challenge base2.0 c0.25 abandon30"
BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py $P --mode challenge --phases 2 --max-days 700 2>&1 | grep -v "^  20" > logs/roll5_bold20.log
echo "=== $(date -u +%H:%M:%S) funded base1.25 c0.5, payouts every 30d"
BASE_SCALE=1.25 CPPI_FRAC=0.5 ABANDON_FRAC=0 python3 -u v4/harness_v4.py $P --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/roll5_funded125.log
echo "=== $(date -u +%H:%M:%S) funded base1.5 c0.5, payouts every 30d"
BASE_SCALE=1.5 CPPI_FRAC=0.5 ABANDON_FRAC=0 python3 -u v4/harness_v4.py $P --mode funded --payout-days 30 --keep-buffer 0.0 --max-days 365 2>&1 | grep -v "^  20" > logs/roll5_funded15.log
echo "=== $(date -u +%H:%M:%S) ALL DONE"
