#!/bin/bash
# Reg. 5 step R-1 (docs/REG5_PLAN.md): full Bybit funding history (public API, no keys) for the research symbols + everything listed now,
# from 2020-01-01. Runs ~1-2 h in the background at low priority (does not touch the bot), then pushes the file to branch `vps-funding`.
# Owner, on the VPS, from the repo folder:   bash ops/funding_history.sh        (progress: tail -f logs/funding_history.log)
set -e
cd "$(dirname "$0")/.."
mkdir -p logs research/vps_dump
PY=${PY:-$( [ -x /opt/propsleeves/venv/bin/python ] && echo /opt/propsleeves/venv/bin/python || echo python3 )}
nohup nice -n 15 bash -c "
  $PY fetch_bybit_funding.py --since 2020-01-01 --symbols-file research/vps_symbols.txt --add-listed --out research/vps_dump/funding_bybit_full.csv.gz &&
  git checkout -B vps-funding && git add -f research/vps_dump/funding_bybit_full.csv.gz &&
  git -c user.name=vps -c user.email=vps@example.com commit -m 'Bybit funding history 2020-01..now' &&
  git push -u origin vps-funding --force && git checkout - && echo 'ГОТОВО: файл в ветке vps-funding'
" > logs/funding_history.log 2>&1 &
echo "Запущено в фоне. Ход: tail -f logs/funding_history.log  (в конце строка ГОТОВО)"
