# usage: r3_funded.sh <HIVOL_MULT> <tag> <starts-csv>
cd /home/user/quant-project; set -a
. <(grep -v "^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN" evns/base.env)
. <(grep -v "^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH" evns/funded.env)
set +a; export MODEL_DIR=$PWD/v4/models HIVOL_MULT=$1
python3 v4/harness_v4.py --mode funded --firm mubite_8_dd10 --maker-fill 0.33 --exec-lag-h 1 --end "2026-10-26 23:00" --ml-panels --panels-prefix out_exp21c_pred --panels-suffix _noage_oct --panels-8h out_exp17_pred_8h_oct_noconst.npy --payout-days 30 --max-days 90 --starts $3
cp out_harness_rolling_mubite_8_dd10_funded.csv logs/r3_funded_$2.csv
