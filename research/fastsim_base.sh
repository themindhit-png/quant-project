#!/bin/bash
# Phase T: base path for the fast account simulator (research/fastsim.py). One long harness run of the REAL bot (pure: K=1, no firm
# floors) with the leverage FIXED at E0 (VT_TARGET_DVOL huge, VT_MAX_LEV=E0) and no gross caps, so the hourly equity return / E0 is the
# book's return per unit of exposure, net of fees, slippage, funding, 1 h execution lag and short stops. Hourly gross/equity is saved too.
# usage: research/fastsim_base.sh <HIVOL_MULT> <E0>   -> logs/fs_base_h<HIVOL>_e<E0>.csv (+ _gx.csv, .log)
set -e
cd "$(dirname "$0")/.."
H=$1; E0=$2; TAG=_fs_h${H}_e${E0}
FILT='^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH'
( set -a; . <(grep -v "$FILT" evns/base.env); set +a
  export MODEL_DIR=$PWD/v4/models HIVOL_MULT=$H VT_TARGET_DVOL=10 VT_MAX_LEV=$E0 MAX_GROSS_X_EQ=100 NOTIONAL_BASE=equity \
         NOTIONAL_X=100 POSITION_X=100 MARGIN_PCT=10000 TOTAL_DD_MODE=static
  python3 v4/harness_v4.py --firm personal --mode funded --pure --maker-fill 0.33 --exec-lag-h 1 --start 2021-12-01 --end "2026-10-26 23:00" \
    --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy \
    --gx-every-h 1 --out-tag $TAG ) > logs/fs_base_h${H}_e${E0}.log 2>&1
mv out_harness_v4_personal_funded_pure$TAG.csv logs/fs_base_h${H}_e${E0}.csv
mv out_harness_gx_personal_funded_pure$TAG.csv logs/fs_base_h${H}_e${E0}_gx.csv
mv out_harness_rho_personal_funded_pure$TAG.csv logs/fs_base_h${H}_e${E0}_rho.csv
echo "DONE $TAG"
