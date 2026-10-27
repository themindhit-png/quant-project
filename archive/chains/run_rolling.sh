#!/bin/sh
# rolling-start challenge/funded simulations of the REAL bot (fair OOS ML panels); one Data load per config
cd /agent/workspace/proj
STARTS="2022-01-01,2022-03-01,2022-05-01,2022-07-01,2022-09-01,2022-11-01,2023-01-01,2023-03-01,2023-05-01,2023-07-01,2023-09-01,2023-11-01,2024-01-01,2024-03-01,2024-05-01,2024-07-01,2024-09-01,2024-11-01,2025-01-01,2025-03-01,2025-05-01,2025-07-01,2025-09-01,2025-11-01,2026-01-01"
run() { # $1 label, then env assignments..., args via ENV var ARGS
  echo "=== $(date -u +%H:%M:%S) $1" >> logs/rolling.log
}
run "mubite_8_dd10 flat base1.0";        FIRM=mubite_8_dd10 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_mubite_flat.log 2>&1
run "mubite_8_dd10 convex kmax2.5 c1.0"; K_MAX=2.5 CPPI_FRAC=1.0 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm mubite_8_dd10 --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_mubite_convex.log 2>&1
run "hyro swing flat base1.0";           python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm hyrotrader_2step_swing --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_hyro_swing_flat.log 2>&1
run "hyro swing convex kmax2.5 c1.0";    K_MAX=2.5 CPPI_FRAC=1.0 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode challenge --firm hyrotrader_2step_swing --starts "$STARTS" --phases 2 --max-days 700 > logs/roll_hyro_swing_convex.log 2>&1
run "mubite funded payouts 30d flat";    python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode funded --firm mubite_8_dd10 --starts "2022-01-01,2022-07-01,2023-01-01,2023-07-01,2024-01-01,2024-07-01,2025-01-01,2025-07-01" --payout-days 30 --max-days 365 > logs/roll_mubite_funded.log 2>&1
run "mubite funded payouts 30d convex";  K_MAX=2.5 CPPI_FRAC=1.0 python3 -u v4/harness_v4.py --ml-panels --start 2022-01-01 --mode funded --firm mubite_8_dd10 --starts "2022-01-01,2022-07-01,2023-01-01,2023-07-01,2024-01-01,2024-07-01,2025-01-01,2025-07-01" --payout-days 30 --max-days 365 > logs/roll_mubite_funded_convex.log 2>&1
echo "=== $(date -u +%H:%M:%S) ALL DONE" >> logs/rolling.log
