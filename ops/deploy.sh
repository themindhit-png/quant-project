#!/usr/bin/env bash
# Деплой бота v4 на VPS одной командой (запускать от root из клона репо, обычно /root/quant-project):
#   bash ops/deploy.sh --check     # только проверить: обновить репо, тесты, посчитать хэши, показать, что изменится
#   bash ops/deploy.sh             # то же + бэкап, копирование кода, новый EXPECTED_CONFIG_HASH, перезапуск, проверка /status
#   bash ops/deploy.sh --models    # то же + замена моделей из репо (только когда это решение записано в docs/STATUS.md)
# Ничего секретного не печатает. Откат — команды в конце вывода (бэкапы с меткой времени рядом с оригиналами).
set -euo pipefail

BRANCH="${BRANCH:-claude/capabilities-environment-limits-3w4yfe}"
UNIT="${UNIT:-propsleeves}"
APP="${APP:-/opt/propsleeves}"               # там src/v4 (код бота), venv, models, state
SRC="$APP/src/v4"
BOTPY="${BOTPY:-$APP/venv/bin/python}"
SYSTEMD_DIR="${SYSTEMD_DIR:-/etc/systemd/system}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TS="$(date -u +%Y%m%d-%H%M%S)"
CHECK=0; MODELS=0
for a in "$@"; do
  case "$a" in --check) CHECK=1 ;; --models) MODELS=1 ;; *) echo "неизвестный ключ: $a" >&2; exit 1 ;; esac
done

say()  { printf '\n== %s\n' "$*"; }
die()  { printf '\n!! ОШИБКА: %s\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "запусти от root (sudo bash ops/deploy.sh)"
[[ -x "$BOTPY" ]] || die "нет python бота: $BOTPY"
[[ -d "$SRC" ]] || die "нет каталога кода бота: $SRC"

# ---------------------------------------------------------------- 1. репо
say "1/7 Обновляю репо $REPO (ветка $BRANCH)"
cd "$REPO"
git diff --quiet && git diff --cached --quiet || die "в $REPO есть незакоммиченные правки — разберись (git status)"
git fetch -q origin "$BRANCH"
git checkout -q "$BRANCH"
git pull -q --ff-only origin "$BRANCH"
echo "коммит: $(git log --oneline -1)"

# ---------------------------------------------------------------- 2. тесты (отдельный venv, бот не трогаем)
say "2/7 Тесты v4/tests_v4.py"
TV="$REPO/.venv-tests"
if [[ ! -x "$TV/bin/python" ]]; then
  echo "создаю venv для тестов: $TV (один раз, ~2 мин)"
  "$BOTPY" -m venv "$TV" 2>/dev/null || python3 -m venv "$TV"
fi
"$TV/bin/pip" install -q -r "$REPO/requirements.txt"
if ! "$TV/bin/python" "$REPO/v4/tests_v4.py" > "/tmp/deploy_tests_$TS.log" 2>&1; then
  tail -30 "/tmp/deploy_tests_$TS.log"; die "тесты не прошли (полный лог /tmp/deploy_tests_$TS.log). Деплой отменён."
fi
tail -1 "/tmp/deploy_tests_$TS.log"

# ---------------------------------------------------------------- 3. env-файл сервиса
say "3/7 Ищу env-файл сервиса $UNIT"
ENVF="$(systemctl show -p EnvironmentFiles --value "$UNIT" | awk '{print $1}' | sed 's/^-//')"
[[ -n "$ENVF" && -f "$ENVF" ]] || die "у $UNIT нет EnvironmentFile (systemctl cat $UNIT). Пришли вывод в чат."
echo "env-файл: $ENVF"

# значение переменной из env-файла (последнее вхождение, без кавычек и пробелов по краям)
envget() {
  { grep -E "^[[:space:]]*(export[[:space:]]+)?$1=" "$ENVF" || true; } | tail -1 | cut -d= -f2- | tr -d "\"'" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

# хэш конфига = config.py из каталога $1 с переменными из env-файла (+ Environment= юнита); печатает 16 символов
cfg_hash() {
  "$BOTPY" - "$ENVF" "$1" "$(systemctl show -p Environment --value "$UNIT")" <<'PY'
import os, sys, shlex, subprocess
env = {}
for line in open(sys.argv[1], encoding='utf-8'):
    line = line.strip()
    if not line or line.startswith('#') or '=' not in line:
        continue
    k, v = line.split('=', 1); k = k.strip()
    if k.startswith('export '):
        k = k[7:].strip()
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in '"\'':
        v = v[1:-1]
    env[k] = v
for kv in shlex.split(sys.argv[3] or ''):
    if '=' in kv:
        k, v = kv.split('=', 1); env.setdefault(k, v)
env.pop('EXPECTED_CONFIG_HASH', None); env['PATH'] = os.environ.get('PATH', '/usr/bin:/bin')
r = subprocess.run([sys.executable, '-c', 'import config as C; print(C.effective_config()[1])'], cwd=sys.argv[2], env=env,
                   capture_output=True, text=True)
if r.returncode != 0:
    sys.stderr.write(r.stderr[-1500:]); sys.exit(1)
print(r.stdout.strip().splitlines()[-1])
PY
}

# ---------------------------------------------------------------- 4. хэши: старый код (сверка) и новый
say "4/7 Хэши конфигурации"
CUR="$(envget EXPECTED_CONFIG_HASH)"
OLD="$(cfg_hash "$SRC")" || die "не смог посчитать хэш старым кодом"
NEW="$(cfg_hash "$REPO/v4")" || die "не смог посчитать хэш новым кодом"
echo "в env сейчас:        ${CUR:-<нет>}"
echo "старый код с env:    $OLD"
echo "новый код с env:     $NEW"
if [[ -n "$CUR" && "$CUR" != "$OLD" ]]; then
  die "старый код с этим env даёт $OLD, а в env записан $CUR: env прочитан не так, как его видит бот. Деплой отменён, пришли вывод в чат."
fi
RESETS="$(grep -nE '^[[:space:]]*(export[[:space:]]+)?RESET_[A-Z_]+=' "$ENVF" | grep -vE '=[[:space:]]*"?0?"?[[:space:]]*$' | cut -d= -f1 || true)"
[[ -n "$RESETS" ]] && echo "в env есть RESET_* (будут закомментированы, иначе сработают ещё раз): $RESETS"
MD="$(envget MODEL_DIR)"; MD="${MD:-$SRC/models}"
MODELS_DIFF=0
for f in "$REPO"/v4/models/*; do
  b="$(basename "$f")"
  if [[ ! -f "$MD/$b" ]] || ! cmp -s "$f" "$MD/$b"; then echo "модель отличается от репо: $MD/$b"; MODELS_DIFF=1; fi
done
[[ $MODELS_DIFF -eq 0 ]] && echo "модели в $MD = репо"
RESTART="$(systemctl show -p Restart --value "$UNIT")"
echo "автоперезапуск сервиса: Restart=$RESTART"
echo "изменённые файлы кода:"; diff -rq --exclude=__pycache__ --exclude=models "$SRC" "$REPO/v4" || true

if [[ $CHECK -eq 1 ]]; then
  say "Режим --check: ничего не изменено. Для деплоя: bash ops/deploy.sh"; exit 0
fi
[[ $MODELS_DIFF -eq 0 || $MODELS -eq 1 ]] || die "модели бота не совпадают с репо — их не трогаю (это пред-регистрация). Пришли вывод в чат."

# ---------------------------------------------------------------- 5. бэкап + код + env
say "5/7 Бэкап и установка"
cp -a "$SRC" "$SRC.bak.$TS"; cp -a "$ENVF" "$ENVF.bak.$TS"
echo "бэкап кода: $SRC.bak.$TS ; бэкап env: $ENVF.bak.$TS"
for f in "$REPO"/v4/*.py "$REPO"/v4/requirements.txt; do cp -f "$f" "$SRC/"; done
rm -rf "$SRC/__pycache__"
if [[ -d "$APP/ops" ]]; then                     # сервисные скрипты (Telegram, бэкап, сборщик) — ставит ops/install.sh
  cp -f "$REPO"/ops/opslib.py "$REPO"/ops/tg_control.py "$REPO"/ops/daily.py "$REPO"/ops/collect_bybit.py "$APP/ops/"
  systemctl is-enabled -q propsleeves-tg 2>/dev/null && systemctl restart propsleeves-tg || true
fi
ROLLBACK="rm -rf $SRC && cp -a $SRC.bak.$TS $SRC && cp -a $ENVF.bak.$TS $ENVF"
if [[ $MODELS -eq 1 && $MODELS_DIFF -eq 1 ]]; then
  case "$(realpath -m "$MD")/" in
    "$(realpath -m "$SRC")"/*) echo "модели внутри $SRC — их бэкап уже в $SRC.bak.$TS" ;;
    *) cp -a "$MD" "$MD.bak.$TS"; echo "бэкап моделей: $MD.bak.$TS"; ROLLBACK="$ROLLBACK && rm -rf $MD && cp -a $MD.bak.$TS $MD" ;;
  esac
  cp -f "$REPO"/v4/models/* "$MD/"; echo "модели заменены из репо"
fi
ROLLBACK="$ROLLBACK && systemctl restart $UNIT"
if grep -qE '^[[:space:]]*(export[[:space:]]+)?EXPECTED_CONFIG_HASH=' "$ENVF"; then
  sed -i -E "s/^([[:space:]]*(export[[:space:]]+)?EXPECTED_CONFIG_HASH=).*/\1$NEW/" "$ENVF"
else
  printf '\nEXPECTED_CONFIG_HASH=%s\n' "$NEW" >> "$ENVF"
fi
if [[ -n "$RESETS" ]]; then
  sed -i -E 's/^([[:space:]]*(export[[:space:]]+)?RESET_[A-Z_]+=)/# (deploy '"$TS"') \1/' "$ENVF"
fi
echo "EXPECTED_CONFIG_HASH: ${CUR:-<нет>} -> $NEW"
"$BOTPY" -m pip install -q -r "$SRC/requirements.txt"

# сбой торгового цикла теперь завершает процесс (код 1) — systemd обязан его поднять
if [[ "$RESTART" == "no" || -z "$RESTART" ]]; then
  mkdir -p "$SYSTEMD_DIR/$UNIT.service.d"
  printf '[Service]\nRestart=always\nRestartSec=30\n' > "$SYSTEMD_DIR/$UNIT.service.d/restart.conf"
  echo "добавлен автоперезапуск: $SYSTEMD_DIR/$UNIT.service.d/restart.conf"
fi
systemctl daemon-reload

# ---------------------------------------------------------------- 6. перезапуск
say "6/7 Перезапуск $UNIT"
systemctl restart "$UNIT"

# ---------------------------------------------------------------- 7. проверка /status
say "7/7 Жду /status (до 3 мин)"
PORT="$(envget PORT)"; PORT="${PORT:-8080}"
TOKEN="$(envget STATUS_TOKEN)"
GOT=""
for _ in $(seq 18); do
  sleep 10
  systemctl is-active -q "$UNIT" || continue
  GOT="$(curl -fsS -m 5 -H "X-Status-Token: $TOKEN" "http://127.0.0.1:$PORT/status" 2>/dev/null \
        | "$BOTPY" -c 'import json,sys; s=json.load(sys.stdin); print(s.get("config_hash") or "", s.get("version"), "cycles", s.get("cycles"), "blacklist", s.get("blacklist_n"))' 2>/dev/null || true)"
  [[ "$GOT" == "$NEW "* ]] && break
done
echo "status: ${GOT:-<нет ответа>}"
if [[ "$GOT" != "$NEW "* ]]; then
  journalctl -u "$UNIT" -n 40 --no-pager || true
  cat <<EOF

!! Бот не подтвердил новый хэш. ОТКАТ (скопируй и выполни):
   $ROLLBACK
Пришли вывод выше в чат.
EOF
  exit 1
fi
cat <<EOF

ГОТОВО: бот работает на новом коде, config_hash $NEW.
Откат при необходимости:
   $ROLLBACK
EOF
