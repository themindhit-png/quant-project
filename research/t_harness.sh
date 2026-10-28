#!/bin/bash
# Phase T: full-harness check of the fast-simulator finalists (Mubite Two-Step Pro with all add-ons: target 8 %, DD 10 %).
# Consensus risk layer: V2 book, CPPI 1.0 (challenge) / 0.75 (funded), K_MAX 1.5x base, daily throttle 0.5 %, daily halt 0.5 of the firm
# limit, floor guard 3 %, drawdown mode below 30 % buffer (top-vol third off), no abandon.
# stress: SLIP=1.5 STRESS_TAG=_x15 (costs x1.5) or MAKER_FILL=0 STRESS_TAG=_taker (taker only)
# usage: research/t_harness.sh challenge <VT> <BASE>   -> logs/t_ch_<VT>_<BASE>.csv  (123 starts every 14 days, both phases)
#        research/t_harness.sh funded <VT> <BASE>      -> logs/t_fu_<VT>_<BASE>.csv  (98 starts every 14 days, 365 days, Mubite payouts / 28 d)
set -e
cd "$(dirname "$0")/.."
MODE=$1; VT=$2; B=$3; TAG=${VT}_${B}${STRESS_TAG:-}
FILT='^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH'
COMMON=(--firm mubite_8_dd10 --maker-fill ${MAKER_FILL:-0.33} --slip-scale ${SLIP:-1.0} --exec-lag-h 1 --end "2026-10-26 23:00" --ml-panels --panels-prefix out_exp21c_pred
        --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy)
KMAX=$(python3 -c "print($B * 1.5)")
( set -a; . <(grep -v "$FILT" evns/base.env); set +a
  export MODEL_DIR=$PWD/v4/models HIVOL_MULT=0.5 VT_TARGET_DVOL=$VT BASE_SCALE=$B BASE_SCALE_P2=$B K_MAX=$KMAX ABANDON_FRAC=0 \
         DD_HIVOL_BUF=0.3 DD_HIVOL_MULT=0 DAILY_THROTTLE_PCT=0.5 DAILY_HALT_FRAC=0.5 FLOOR_GUARD_FRAC=0.03
  if [ "$MODE" = challenge ]; then
    export CPPI_FRAC=1.0
    STARTS=$(python3 -c "import pandas as pd; print(','.join(str(d.date()) for d in pd.date_range('2022-01-03', '2026-09-07', freq='14D')))")
    python3 v4/harness_v4.py --mode challenge --phases 2 "${COMMON[@]}" --starts "$STARTS" --out-tag _t_$TAG
  else
    export CPPI_FRAC=0.75
    STARTS=${FU_STARTS:-$(python3 -c "import pandas as pd; print(','.join(str(d.date()) for d in pd.date_range('2022-01-03', '2025-10-20', freq='14D')))")}
    python3 v4/harness_v4.py --mode funded "${COMMON[@]}" --starts "$STARTS" --max-days 365 --payout-days 28 --mubite-payouts \
      --keep-buffer 0.02 --out-tag _t_$TAG
  fi ) > logs/t_${MODE:0:2}_$TAG.log 2>&1
cp out_harness_rolling_mubite_8_dd10_${MODE}_t_$TAG.csv logs/t_${MODE:0:2}_$TAG.csv
echo "DONE $MODE $TAG"
