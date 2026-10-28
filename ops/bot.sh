#!/bin/bash
# Управление ботом на VPS одной командой (docs/GUIDE.md).
#   bash ops/bot.sh status    # состояние: equity, K, фаза, прибыльные дни, пауза, срок ключа
#   bash ops/bot.sh pause     # перед выплатой / выводом: бот закрывает ВСЕ позиции и ждёт (в Telegram придёт «Пауза: всё закрыто»)
#   bash ops/bot.sh resume    # после того, как выплата прошла (баланс уменьшился): бот учтёт разницу как вывод и продолжит
#   bash ops/bot.sh restart   # перезапуск службы (после смены настроек в env-файле)
#   bash ops/bot.sh logs      # последние 60 строк лога
set -euo pipefail
UNIT="${UNIT:-propsleeves}"
ENVF="$(systemctl show -p EnvironmentFiles --value "$UNIT" | awk '{print $1}' | sed 's/^-//')"
[[ -n "$ENVF" && -f "$ENVF" ]] || { echo "у службы $UNIT нет env-файла (systemctl cat $UNIT)"; exit 1; }
envget() { { grep -E "^[[:space:]]*(export[[:space:]]+)?$1=" "$ENVF" || true; } | tail -1 | cut -d= -f2- | tr -d "\"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'; }
STATE="$(envget STATE_FILE)"; PAUSE="$(envget PAUSE_FILE)"
[[ -n "$PAUSE" ]] || PAUSE="${STATE}.pause"
[[ -n "$STATE" || -n "$(envget PAUSE_FILE)" ]] || { echo "в $ENVF нет STATE_FILE — пауза не найдёт файл. Пришли вывод в чат."; exit 1; }
PORT="$(envget PORT)"; PORT="${PORT:-8080}"; TOKEN="$(envget STATUS_TOKEN)"
status() { curl -fsS -m 5 -H "X-Status-Token: $TOKEN" "http://127.0.0.1:$PORT/status" | python3 -m json.tool 2>/dev/null || echo "бот не отвечает на /status"; }
case "${1:-status}" in
  status)  status; [[ -f "$PAUSE" ]] && echo "ПАУЗА ВКЛЮЧЕНА ($PAUSE)" || true ;;
  pause)   touch "$PAUSE"; echo "Пауза включена. Через 1-2 минуты в Telegram придёт «все позиции закрыты». Потом запрашивай выплату / выводи деньги." ;;
  resume)  rm -f "$PAUSE"; echo "Пауза снята. Бот учтёт изменение баланса как вывод (не убыток) и продолжит торговать." ;;
  restart) systemctl restart "$UNIT"; sleep 5; systemctl --no-pager -l status "$UNIT" | head -5 ;;
  logs)    [[ "$UNIT" == *personal* ]] && DEF=/opt/propsleeves/log_personal.txt || DEF=/opt/propsleeves/log_bot.txt; LOG="${LOG:-$DEF}"; [[ -f "$LOG" ]] && tail -60 "$LOG" || journalctl -u "$UNIT" -n 60 --no-pager ;;
  *) echo "команды: status | pause | resume | restart | logs"; exit 1 ;;
esac
