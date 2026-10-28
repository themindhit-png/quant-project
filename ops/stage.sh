#!/usr/bin/env bash
# Смена стадии бота одной командой (запускать от root из клона репо, обычно /root/quant-project). Подробно — docs/GUIDE.md.
#   bash ops/stage.sh phase1 10000     # новый челлендж Mubite $10k, фаза 1
#   bash ops/stage.sh phase2 10000     # фаза 2 того же челленджа (после письма фирмы о прохождении фазы 1)
#   bash ops/stage.sh funded 10000     # funded-счёт $10k (после прохождения обеих фаз и KYC)
#   bash ops/stage.sh personal 5000    # свой счёт Bybit (сумма = сколько на нём сейчас USDT)
#   bash ops/stage.sh keys             # вставить новые API-ключи (новый субаккаунт / продление), без смены стадии
# Что делает: берёт из текущего env-файла службы только «свои» строки (ключи Bybit, Telegram, порт, токен, MODEL_DIR; всё прочее
# отбрасывается — одна версия настроек), добавляет стратегию
# (evns/base.env) и профиль стадии (evns/mubite_challenge.env / mubite_funded.env / personal.env), ставит PHASE / MODE /
# ACCOUNT_SIZE и НОВЫЙ файл состояния (каждая стадия начинается с чистого счёта), считает хэш конфигурации, перезапускает бота
# и ждёт /status. Старый env сохраняется рядом (.bak.<время>). Ничего секретного не печатает.
set -euo pipefail
UNIT="${UNIT:-propsleeves}"; APP="${APP:-/opt/propsleeves}"; SRC="$APP/src/v4"; BOTPY="${BOTPY:-$APP/venv/bin/python}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; TS="$(date -u +%Y%m%d-%H%M%S)"
die() { printf '\n!! ОШИБКА: %s\n' "$*" >&2; exit 1; }
[[ $EUID -eq 0 || -n "${STAGE_DRY:-}" ]] || die "запусти от root"
STAGE="${1:-}"; SIZE="${2:-}"
ENVF="${ENVF:-$(systemctl show -p EnvironmentFiles --value "$UNIT" | awk '{print $1}' | sed 's/^-//')}"
[[ -n "$ENVF" && -f "$ENVF" ]] || die "у службы $UNIT нет env-файла"

if [[ "$STAGE" == keys ]]; then
  read -r -p "API key: " K; read -r -s -p "API secret (не отображается): " S; echo
  [[ -n "$K" && -n "$S" ]] || die "пустой ключ"
  cp -a "$ENVF" "$ENVF.bak.$TS"
  sed -i -E '/^[[:space:]]*(export[[:space:]]+)?BYBIT_API_(KEY|SECRET)=/d' "$ENVF"
  printf 'BYBIT_API_KEY=%s\nBYBIT_API_SECRET=%s\n' "$K" "$S" >> "$ENVF"
  systemctl restart "$UNIT"; echo "ключи заменены, бот перезапущен (бэкап env: $ENVF.bak.$TS)"; exit 0
fi

case "$STAGE" in
  phase1) PROFILE=mubite_challenge.env; MODE=challenge; PHASE=1 ;;
  phase2) PROFILE=mubite_challenge.env; MODE=challenge; PHASE=2 ;;
  funded) PROFILE=mubite_funded.env;    MODE=funded;    PHASE=1 ;;
  personal) PROFILE=personal.env;       MODE=funded;    PHASE=1 ;;
  *) die "стадия: phase1 | phase2 | funded | personal | keys" ;;
esac
[[ "$SIZE" =~ ^[0-9]+$ ]] || die "укажи размер счёта числом, например: bash ops/stage.sh $STAGE 10000"
[[ -f "$REPO/evns/$PROFILE" ]] || die "нет профиля evns/$PROFILE"

# ключи профиля и стратегии — их берём из репо, не из старого env
LOCAL_RE='^(BYBIT_|TG_|STATUS_|STATE_|PORT|DRY_RUN|MODEL_DIR|EXPECTED_CONFIG_HASH)'
NEW="$(mktemp)"
{
  echo "# собрано ops/stage.sh $STAGE $SIZE ($TS UTC); свои строки (ключи, Telegram, порт) сохранены из прежнего env"
  grep -E '^[[:space:]]*(export[[:space:]]+)?[A-Z0-9_]+=' "$ENVF" | sed -E 's/^[[:space:]]*(export[[:space:]]+)?//' \
    | grep -E '^(BYBIT_API_|TG_|STATUS_TOKEN|PORT|MODEL_DIR|DRY_RUN)[A-Z0-9_]*='  # only own lines; any other old setting is dropped
  echo; echo "# --- стратегия (evns/base.env)"; grep -E '^[A-Z0-9_]+=' "$REPO/evns/base.env" | grep -vE "$LOCAL_RE" | grep -vE '^(ACCOUNT_SIZE|MODE|PHASE|FIRM)='
  echo; echo "# --- профиль (evns/$PROFILE)"; grep -E '^[A-Z0-9_]+=' "$REPO/evns/$PROFILE" | grep -vE '^(MODE|PHASE)='
  echo; echo "# --- стадия"
  [[ "$STAGE" == personal ]] && echo "BYBIT_BASE_URL=https://api.bybit.com" || echo "BYBIT_BASE_URL=https://api-demo.bybit.com"
  echo "MODE=$MODE"; echo "PHASE=$PHASE"; echo "ACCOUNT_SIZE=$SIZE"
  echo "STATE_FILE=$APP/state/${STAGE}_${SIZE}_$TS.json"; echo "STATE_BACKUP_FILE=$APP/state/${STAGE}_${SIZE}_$TS.backup.json"
} > "$NEW"
grep -q '^BYBIT_API_KEY=' "$NEW" || die "в прежнем env нет BYBIT_API_KEY — сначала: bash ops/stage.sh keys"
[[ -n "${STAGE_DRY:-}" ]] || mkdir -p "$APP/state"

# хэш конфигурации новым env (тем же способом, что ops/deploy.sh)
HASH="$("$BOTPY" - "$NEW" "$SRC" <<'PY'
import os, sys, subprocess
env = {}
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1); env[k.strip()] = v.strip().strip('"\'')
env['PATH'] = os.environ.get('PATH', '/usr/bin:/bin')
r = subprocess.run([sys.executable, '-c', 'import config as C; print(C.effective_config()[1])'], cwd=sys.argv[2], env=env, capture_output=True, text=True)
if r.returncode:
    sys.stderr.write(r.stderr[-1500:]); sys.exit(1)
print(r.stdout.strip().splitlines()[-1])
PY
)" || die "не смог посчитать хэш (код бота в $SRC устарел? сначала bash ops/deploy.sh)"
echo "EXPECTED_CONFIG_HASH=$HASH" >> "$NEW"

if [[ -n "${STAGE_DRY:-}" ]]; then sed -E 's/^(BYBIT_API_SECRET|BYBIT_API_KEY|TG_BOT_TOKEN|STATUS_TOKEN)=.*/\1=***/' "$NEW"; rm -f "$NEW"; exit 0; fi
cp -a "$ENVF" "$ENVF.bak.$TS"; install -m 600 "$NEW" "$ENVF"; rm -f "$NEW"
echo "стадия: $STAGE, счёт $SIZE, хэш $HASH (бэкап прежнего env: $ENVF.bak.$TS)"
systemctl restart "$UNIT"
PORT="$(grep -E '^PORT=' "$ENVF" | tail -1 | cut -d= -f2)"; PORT="${PORT:-8080}"
TOKEN="$(grep -E '^STATUS_TOKEN=' "$ENVF" | tail -1 | cut -d= -f2)"
for i in $(seq 1 36); do
  sleep 5
  if curl -fsS -m 5 -H "X-Status-Token: $TOKEN" "http://127.0.0.1:$PORT/status" 2>/dev/null | grep -q "$HASH"; then
    echo "бот запущен с новой стадией (/status отвечает, хэш $HASH)."; exit 0
  fi
done
echo "!! /status не ответил с новым хэшем за 3 мин. Логи: bash ops/bot.sh logs. Откат: cp -a $ENVF.bak.$TS $ENVF && systemctl restart $UNIT"
exit 1
