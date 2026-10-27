#!/bin/bash
# chain: unit tests -> harness daily book (fixed VT) -> harness 5-sleeve book (ml8 age180+IV)
cd /agent/workspace/proj
echo "=== tests ==="
python3 v4/tests_v4.py 2>&1 | tail -25 || exit 1
echo "=== harness daily (direct VT) ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded 2>&1 | tail -40
echo "=== harness 5-sleeve ml8 (direct VT) ==="
FIRM=mubite_8_dd10 MODE=funded ACCOUNT_SIZE=10000 BASE_SCALE=1.0 SLEEVES=listing,core,ml3,ml7,ml8 python3 -u v4/harness_v4.py --pure --ml-panels --start 2022-01-01 --mode funded 2>&1 | tail -40
echo "DONE"
