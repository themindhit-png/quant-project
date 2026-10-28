#!/bin/bash
# R3-bis (docs/PHASE_R_PLAN.md): one drawdown policy x vol target on the V2 book (HIVOL_MULT=0.5).
# usage: research/r3bis_run.sh <P0|P1|P2|P3> <VT, e.g. 0.006>
# Challenge (mubite 2-step, 123 starts every 14 days 2022-01-03..2026-09-07), then funded 90 days from each pass date <= 2026-07-26.
# Output: logs/r3b_ch_<P>_<VT>.csv, logs/r3b_fu_<P>_<VT>.csv (+ .log). EV table: python3 research/r3bis_ev.py
set -e
cd "$(dirname "$0")/.."
P=$1; VT=$2; TAG=${P}_${VT}
STARTS=$(python3 -c "import pandas as pd; print(','.join(str(d.date()) for d in pd.date_range('2022-01-03', '2026-09-07', freq='14D')))")
COMMON=(--firm mubite_8_dd10 --maker-fill 0.33 --exec-lag-h 1 --end "2026-10-26 23:00" --ml-panels --panels-prefix out_exp21c_pred
        --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy)
FILT='^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH'

policy() {   # $1 = challenge|funded: CPPI / ABANDON / K_MIN_FRAC / drawdown mode
  case $P in
    P0) [ "$1" = challenge ] && export CPPI_FRAC=0.25 ABANDON_FRAC=0.30 || export CPPI_FRAC=0.5 ABANDON_FRAC=0 ;;
    P1) export CPPI_FRAC=0.25 ABANDON_FRAC=0 ;;
    P2) export CPPI_FRAC=0.5 ABANDON_FRAC=0 K_MIN_FRAC=0.3 ;;
    P3) export CPPI_FRAC=0.5 ABANDON_FRAC=0 K_MIN_FRAC=0.3 DD_HIVOL_BUF=0.5 DD_HIVOL_MULT=0 ;;
    *) echo "unknown policy $P"; exit 1 ;;
  esac
}

run() {   # $1 = challenge|funded, rest = extra harness args
  local mode=$1; shift
  ( set -a; . <(grep -v "$FILT" evns/base.env); . <(grep -v "$FILT" archive/evns_old/$mode.env); set +a
    export MODEL_DIR=$PWD/v4/models HIVOL_MULT=0.5 VT_TARGET_DVOL=$VT
    policy $mode
    python3 v4/harness_v4.py --mode $mode "${COMMON[@]}" --out-tag _r3b_$TAG "$@" )
}

run challenge --phases 2 --starts "$STARTS" > logs/r3b_ch_$TAG.log 2>&1
cp out_harness_rolling_mubite_8_dd10_challenge_r3b_$TAG.csv logs/r3b_ch_$TAG.csv
PASS=$(python3 -c "
import pandas as pd; df = pd.read_csv('logs/r3b_ch_$TAG.csv'); df = df[df.p2 == 'pass']
d = pd.to_datetime(df.start) + pd.to_timedelta(df.total_days.round(), unit='D'); d = d[d <= '2026-07-26']
print(','.join(str(x.date()) for x in d))")
run funded --payout-days 30 --max-days 90 --starts "$PASS" > logs/r3b_fu_$TAG.log 2>&1
cp out_harness_rolling_mubite_8_dd10_funded_r3b_$TAG.csv logs/r3b_fu_$TAG.csv
echo "DONE $TAG"
