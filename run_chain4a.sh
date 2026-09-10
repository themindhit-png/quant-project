#!/bin/bash
# chain 4a (sequential, memory-safe): daily-book harness (fixed VT) -> 5-sleeve harness taker-only -> 5-sleeve maker 0.5
cd /agent/workspace/proj
echo "=== harness daily (direct VT) ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded 2>&1 | grep -v "^  20"
echo "=== harness 5-sleeve ml8, maker-fill 0.0 (taker only) ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 SLEEVES=listing,core,ml3,ml7,ml8 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded --maker-fill 0.0 2>&1 | grep -v "^  20"
echo "=== harness 5-sleeve ml8, maker-fill 0.5 ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 SLEEVES=listing,core,ml3,ml7,ml8 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded --maker-fill 0.5 2>&1 | grep -v "^  20"
echo "=== harness 5-sleeve ml8, no short stop (STOP_SHORT_PCT=0) maker 0.7 ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 SLEEVES=listing,core,ml3,ml7,ml8 STOP_SHORT_PCT=0 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded 2>&1 | grep -v "^  20"
echo "DONE"
