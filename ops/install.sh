#!/usr/bin/env bash
# Установка на ЧИСТЫЙ VPS одной командой (Ubuntu 22.04 / 24.04 или Debian 12, от root). Подробно — docs/GUIDE.md, раздел «Чистый сервер».
#   git clone <репо> /root/quant-project && cd /root/quant-project && git checkout claude/capabilities-environment-limits-3w4yfe
#   bash ops/install.sh
# Ставит: пакеты, Python venv, код бота (/opt/propsleeves/src/v4), модели, службы:
#   propsleeves          — бот проп-счёта (запускается командой ops/stage.sh, не здесь);
#   propsleeves-tg       — управление из Telegram (/status /pause /resume);
#   propsleeves-daily    — 00:30 UTC: бэкап состояния и сводка в Telegram;
#   propsleeves-collect  — каждый час :05: данные Bybit на диск (/opt/propsleeves/data/bybit).
# Повторный запуск ничего не ломает (обновляет код и службы, env-файл не трогает).
set -euo pipefail
APP="${APP:-/opt/propsleeves}"; UNIT=propsleeves
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
say() { printf '\n== %s\n' "$*"; }
die() { printf '\n!! ОШИБКА: %s\n' "$*" >&2; exit 1; }
[[ $EUID -eq 0 || -n "${SKIP_SYSTEMD:-}" ]] || die "запусти от root"

say "1/7 Пакеты"
export DEBIAN_FRONTEND=noninteractive
if [[ -z "${SKIP_APT:-}" ]]; then
  apt-get update -qq
  apt-get install -y -qq python3 python3-venv python3-pip git curl libgomp1 logrotate ca-certificates >/dev/null
fi
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || die "нужен Python 3.10+"

say "2/7 Каталоги и venv"
mkdir -p "$APP"/{src/v4,models,state,backup,data,ops}
[[ -x "$APP/venv/bin/python" ]] || python3 -m venv "$APP/venv"
"$APP/venv/bin/pip" install -q --upgrade pip
"$APP/venv/bin/pip" install -q -r "$REPO/v4/requirements.txt"

say "3/7 Код, модели, сервисные скрипты"
cp -f "$REPO"/v4/*.py "$REPO"/v4/requirements.txt "$APP/src/v4/"; rm -rf "$APP/src/v4/__pycache__"
cp -f "$REPO"/v4/models/* "$APP/models/"
cp -f "$REPO"/ops/opslib.py "$REPO"/ops/tg_control.py "$REPO"/ops/daily.py "$REPO"/ops/collect_bybit.py "$APP/ops/"
echo "код: $(cd "$REPO" && git log --oneline -1 2>/dev/null || echo '?')"

say "4/7 Тесты бота (синтетические данные)"
TV="$REPO/.venv-tests"                                  # отдельный venv для тестов (как в ops/deploy.sh), бот остаётся лёгким
[[ -x "$TV/bin/python" ]] || python3 -m venv "$TV"
"$TV/bin/pip" install -q -r "$REPO/requirements.txt"
if ! V4_TESTS_SYNTH=1 "$TV/bin/python" "$REPO/v4/tests_v4.py" > /tmp/install_tests.log 2>&1; then
  tail -20 /tmp/install_tests.log; die "тесты бота не прошли (/tmp/install_tests.log) — пришли вывод в чат"
fi
tail -1 /tmp/install_tests.log
"$TV/bin/python" "$REPO/ops/tests_ops.py" > /tmp/install_tests_ops.log 2>&1 || { tail -20 /tmp/install_tests_ops.log; die "тесты сервисов не прошли"; }
tail -1 /tmp/install_tests_ops.log

say "5/7 Настройки ($APP/bot.env)"
ENVF="$APP/bot.env"
if [[ ! -f "$ENVF" ]]; then
  read -r -p "Telegram: токен бота (от @BotFather): " TGT
  read -r -p "Telegram: ваш chat id (число): " TGC
  { echo "TG_BOT_TOKEN=$TGT"; echo "TG_CHAT_ID=$TGC"
    echo "STATUS_TOKEN=$(head -c 32 /dev/urandom | base64 | tr -dc 'A-Za-z0-9' | head -c 32)"; echo "PORT=8080"
    echo "MODEL_DIR=$APP/models"; } > "$ENVF"
  chmod 600 "$ENVF"
  curl -fsS -m 10 "https://api.telegram.org/bot$TGT/sendMessage" -d chat_id="$TGC" -d text="✅ Сервер подключён к Telegram (установка propsleeves)" >/dev/null \
    && echo "в Telegram отправлено тестовое сообщение — проверь" || echo "!! Telegram не ответил — проверь токен / chat id (их можно поправить в $ENVF)"
else
  echo "$ENVF уже есть — не трогаю"
fi

say "6/7 Службы systemd"
[[ -n "${SKIP_SYSTEMD:-}" ]] && { echo "пропущено (SKIP_SYSTEMD)"; exit 0; }
cat > /etc/systemd/system/$UNIT.service <<EOF
[Unit]
Description=PROP-SLEEVES bot (prop account)
After=network-online.target
Wants=network-online.target
[Service]
WorkingDirectory=$APP/src/v4
EnvironmentFile=$ENVF
ExecStart=$APP/venv/bin/python main.py
Restart=always
RestartSec=30
StandardOutput=append:$APP/log_bot.txt
StandardError=append:$APP/log_bot.txt
[Install]
WantedBy=multi-user.target
EOF
cat > /etc/systemd/system/propsleeves-tg.service <<EOF
[Unit]
Description=PROP-SLEEVES Telegram control (/status /pause /resume)
After=network-online.target
Wants=network-online.target
[Service]
Environment=APP=$APP
ExecStart=$APP/venv/bin/python $APP/ops/tg_control.py
Restart=always
RestartSec=30
StandardOutput=append:$APP/log_tg.txt
StandardError=append:$APP/log_tg.txt
[Install]
WantedBy=multi-user.target
EOF
for job in daily collect; do
  cat > /etc/systemd/system/propsleeves-$job.service <<EOF
[Unit]
Description=PROP-SLEEVES $job
[Service]
Type=oneshot
Environment=APP=$APP
ExecStart=$APP/venv/bin/python $APP/ops/$( [[ $job == daily ]] && echo daily.py || echo collect_bybit.py )
StandardOutput=append:$APP/log_$job.txt
StandardError=append:$APP/log_$job.txt
EOF
done
cat > /etc/systemd/system/propsleeves-daily.timer <<'EOF'
[Unit]
Description=PROP-SLEEVES daily backup and summary
[Timer]
OnCalendar=*-*-* 00:30:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
EOF
cat > /etc/systemd/system/propsleeves-collect.timer <<'EOF'
[Unit]
Description=PROP-SLEEVES hourly Bybit data collector
[Timer]
OnCalendar=*-*-* *:05:00 UTC
Persistent=true
[Install]
WantedBy=timers.target
EOF
cat > /etc/logrotate.d/propsleeves <<EOF
$APP/log_*.txt {
  weekly
  rotate 8
  compress
  missingok
  notifempty
  copytruncate
}
EOF
systemctl daemon-reload
systemctl enable -q propsleeves-tg propsleeves-daily.timer propsleeves-collect.timer     # сам бот включает ops/stage.sh
systemctl restart propsleeves-tg; systemctl start propsleeves-daily.timer propsleeves-collect.timer
echo "службы установлены; бот propsleeves запустится после выбора стадии (ops/stage.sh)"

say "7/7 Готово. Дальше (docs/GUIDE.md):"
echo "  bash ops/stage.sh keys             # API-ключ демо-субаккаунта (Mubite) или своего демо"
echo "  bash ops/stage.sh trial 10000      # проба Mubite  (или phase1 10000 — челлендж)"
echo "  в Telegram: /status"
