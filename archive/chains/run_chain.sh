#!/bin/sh
# sequential experiment chain (one heavy process at a time; 4 GB RAM box)
cd /agent/workspace/proj
for e in "$@"; do
  echo "=== $(date -u +%H:%M:%S) start $e" >> logs/chain.log
  python3 -u "$e.py" > "logs/$e.log" 2>&1
  echo "=== $(date -u +%H:%M:%S) end $e rc=$?" >> logs/chain.log
done
