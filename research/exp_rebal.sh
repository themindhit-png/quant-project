#!/bin/bash
# Experiment: rebalance cadence / no-trade band / weight smoothing on the REAL bot (harness, pure: fixed exposure 2, no firm floors),
# V2 book, 2022-01..2026-10, maker-fill 0.33, lag 1 h. usage: research/exp_rebal.sh <tag> "<ENV overrides>"  -> logs/rb_<tag>.log
set -e
cd "$(dirname "$0")/.."
TAG=$1; OVR=$2
FILT='^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH'
( set -a; . <(grep -v "$FILT" evns/base.env); set +a
  export MODEL_DIR=$PWD/v4/models HIVOL_MULT=0.5 VT_TARGET_DVOL=10 VT_MAX_LEV=2 MAX_GROSS_X_EQ=100 NOTIONAL_BASE=equity \
         NOTIONAL_X=100 POSITION_X=100 MARGIN_PCT=10000 TOTAL_DD_MODE=static $OVR
  python3 v4/harness_v4.py --firm personal --mode funded --pure --maker-fill 0.33 --exec-lag-h 1 --start 2022-01-01 --end "2026-10-26 23:00" \
    --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy \
    --out-tag _rb_$TAG ) > logs/rb_$TAG.log 2>&1
mv out_harness_v4_personal_funded_pure_rb_$TAG.csv logs/rb_$TAG.csv
echo "DONE $TAG"
