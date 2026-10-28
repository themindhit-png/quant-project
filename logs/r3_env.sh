set -a
. <(grep -v "^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN" evns/base.env)
. <(grep -v "^#\|^$\|BYBIT_\|TG_\|STATUS_\|STATE_\|MODEL_DIR\|PORT\|DRY_RUN\|EXPECTED_CONFIG_HASH" evns/challenge.env)
set +a
export MODEL_DIR=$PWD/v4/models
