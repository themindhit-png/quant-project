#!/bin/bash
# Phase T: minimum capital for the own account — the real bot in the harness on accounts of different size (order minimums, dust rule
# MIN_TRADE_USDT, per-position caps). 2025-01..2026-10, own-account preset at VT 1.2 %/day. usage: research/personal_min.sh <ACCOUNT>
set -e
cd "$(dirname "$0")/.."
A=$1
FILT='^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH'
( set -a; . <(grep -v "$FILT" evns/base.env); set +a
  export MODEL_DIR=$PWD/v4/models HIVOL_MULT=0.5 VT_TARGET_DVOL=0.012 VT_MAX_LEV=4 BASE_SCALE=1 CPPI_FRAC=0.5 NOTIONAL_BASE=equity \
         MAX_GROSS_X_EQ=3.5 POS_CAP_PCT=3 SHORT_CAP_PCT=2 DD_HIVOL_BUF=0.3 DD_HIVOL_MULT=0 ACCOUNT_SIZE=$A
  nice python3 v4/harness_v4.py --firm personal --mode funded --account $A --maker-fill 0.33 --exec-lag-h 1 --start 2025-01-01 --end "2026-10-26 23:00" \
    --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy --out-tag _pm_$A ) > logs/pm_$A.log 2>&1
echo "DONE $A"
