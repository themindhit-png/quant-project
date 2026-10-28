#!/usr/bin/env bash
# Второй бот на том же VPS — для СВОЕГО счёта Bybit (работает одновременно с ботом проп-счёта). Один раз (docs/GUIDE.md):
#   bash ops/add_personal.sh            # создаёт службу propsleeves-personal (копия основной: тот же код и модели, свой env,
#                                       # свой порт 8081, свой лог /opt/propsleeves/log_personal.txt), спрашивает ключи своего счёта
#   UNIT=propsleeves-personal bash ops/stage.sh personal <сколько USDT на счёте>    # собрать настройки и запустить
set -euo pipefail
MAIN="${MAIN:-propsleeves}"; NEWU="${NEWU:-propsleeves-personal}"; APP="${APP:-/opt/propsleeves}"; SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
die() { printf '\n!! ОШИБКА: %s\n' "$*" >&2; exit 1; }
[[ $EUID -eq 0 ]] || die "запусти от root"
SRCU="$(systemctl show -p FragmentPath --value "$MAIN")"; [[ -f "$SRCU" ]] || die "не нашёл файл службы $MAIN"
MAINENV="$(systemctl show -p EnvironmentFiles --value "$MAIN" | awk '{print $1}' | sed 's/^-//')"
[[ -f "$SYSTEMD_DIR/$NEWU.service" ]] && die "служба $NEWU уже есть (systemctl status $NEWU)"
ENVP="$APP/personal.env"
read -r -p "API key своего счёта Bybit (права: Read-Write, Unified Trading; без вывода средств!): " K
read -r -s -p "API secret (не отображается): " S; echo
[[ -n "$K" && -n "$S" ]] || die "пустой ключ"
{ echo "BYBIT_API_KEY=$K"; echo "BYBIT_API_SECRET=$S"
  grep -E '^(TG_|MODEL_DIR)[A-Z0-9_]*=' "$MAINENV" || true
  echo "STATUS_TOKEN=$(head -c 24 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)"; echo "PORT=8081"; } > "$ENVP"
chmod 600 "$ENVP"
sed -e "s#EnvironmentFile=.*#EnvironmentFile=$ENVP#" -e "s#log_bot.txt#log_personal.txt#g" "$SRCU" > "$SYSTEMD_DIR/$NEWU.service"
systemctl daemon-reload; systemctl enable -q "$NEWU"
echo "служба $NEWU создана (env $ENVP, порт 8081, лог $APP/log_personal.txt)."
echo "дальше: UNIT=$NEWU bash ops/stage.sh personal <сколько USDT на счёте>"
