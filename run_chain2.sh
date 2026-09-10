#!/bin/sh
# robust chain: ensure final 8h model, stability grid, then 5-sleeve real-path sims (smooth 0.5)
cd /agent/workspace/proj
log() { echo "=== $(date -u +%H:%M:%S) $1" >> logs/chain2.log; }
if ! grep -q DONE logs/exp17b_final_8h.log 2>/dev/null; then
  if pgrep -f "[e]xp17b_final_8h.py" > /dev/null; then
    log "waiting for running exp17b"; until grep -q DONE logs/exp17b_final_8h.log 2>/dev/null; do sleep 20; done
  else
    log "starting exp17b"; python3 -u exp17b_final_8h.py > logs/exp17b_final_8h.log 2>&1
  fi
fi
log "exp19c"; python3 -u exp19c_stability.py > logs/exp19c.log 2>&1
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
log "roll5 bold20"; SLEEVES=listing,core,ml3,ml7,ml8 SMOOTH=0.5 BASE_SCALE=2.0 CPPI_FRAC=0.25 ABANDON_FRAC=0.30 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll5_bold20.log 2>&1
log "roll5 funded b1.5 keep10"; SLEEVES=listing,core,ml3,ml7,ml8 SMOOTH=0.5 BASE_SCALE=1.5 CPPI_FRAC=0.5 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode funded --firm mubite_8_dd10 --starts "2022-01-01,2022-07-01,2023-01-01,2023-07-01,2024-01-01,2024-07-01,2025-01-01,2025-07-01" --payout-days 30 --keep-buffer 0.10 --max-days 365 > logs/roll5_funded_b15_keep10.log 2>&1
log "roll5 flat1.0 challenge"; SLEEVES=listing,core,ml3,ml7,ml8 SMOOTH=0.5 BASE_SCALE=1.0 CPPI_FRAC=0.5 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll5_flat10.log 2>&1
log "ALL DONE"
