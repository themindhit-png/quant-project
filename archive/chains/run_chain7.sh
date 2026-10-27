#!/bin/bash
# chain 7: Opus battery (exp21a) then the 2021 regime test (exp21b) — sequential (memory)
cd /agent/workspace/proj
echo "=== $(date -u +%H:%M:%S) exp21a"
python3 -u exp21a_opus_tests.py > logs/exp21a.log 2>&1
echo "=== $(date -u +%H:%M:%S) exp21b"
python3 -u exp21b_2021.py > logs/exp21b.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE"
