# CLAUDE.md — карта проекта (читать первым)

Проект: рыночно-нейтральный бот PROP-SLEEVES v4 (книга F2, 7 рукавов, перпы Bybit) для проп-челленджей Mubite (из Беларуси доступна только она — `docs/FIRMS.md`) и своего счёта Bybit. Владелец — не программист: объяснять просто, действия давать пошагово. Язык общения — русский.

## Личные разговоры (не про бота)
- Память — в ПРИВАТНОМ репо `themindhit-png/merlin`, ветка `notes`, файл `LIFE.md` (клон `/home/user/merlin`; нет клона — `add_repo` merlin с push, `git clone`, `git checkout notes`). Читать в начале личного разговора, обновлять самому без напоминаний (факты, решения, журнал), коммитить и пушить в `notes`.
- Личное НИКОГДА не писать в этот репозиторий: он публичный.

## Правила работы (экономия лимитов)
- Начинай с `docs/STATUS.md`: там текущая фаза и следующий шаг. Руководство владельца (одна версия): `docs/GUIDE.md`. План: `docs/PHASE_T_PLAN.md`.
- Одна фаза = одна сессия. В конце фазы обнови `docs/STATUS.md`, закоммить, запушь, скажи владельцу: что ввести (`/clear`, нужная `/model`).
- Модели: Opus — решения, план, ревью диффов; Sonnet — исполнение. Чтение больших объёмов — субагентам Sonnet (Explore). Долгие расчёты — в фоне, вывод в `logs/`, в контекст не тянуть.
- Ветка разработки: `claude/capabilities-environment-limits-3w4yfe`. PR не создавать без просьбы.

## Что читать, а что нет
- НЕ читать целиком: `docs/REPORT.md` (92 КБ), `docs/OPUS_REPLY.md` (62 КБ), `docs/PLAN.md` (28 КБ), `archive/**`, любые `*.csv`, `*.npy`, `*.parquet`, `v4/models/*.txt`, `data/raw`, `logs/*`.
- Вместо них: `docs/RESEARCH_SUMMARY.md` (выжимка). В большие документы лезть только точечно (`grep -n`, `sed -n`) и по номеру раздела.
- Бот: `v4/` (`main.py`, `risk.py`, `execution.py`, `signals.py`, `data.py`, `bybit.py`, `config.py`, `state.py`). Тесты: `v4/tests_v4.py`.

## Структура
- `v4/` — живой бот. `v4/models/` — обученные LightGBM + spec_*.json.
- Корень — ядро исследования (пока не двигать, пути заданы через `__file__`): `bt.py` (движок), `build_panels.py`, `dl_binance.py`, `exp_common.py`, `ml_features*.py`, обучение (`exp6_ml`, `exp21c_ml_noage`, `exp17_intraday_ml`, `exp17b_final_8h`, `exp24b_slow_ml`, `exp6b_noage_final`), финал (`exp24f`, `exp26`), проверки (`exp15_leak`, `exp21e_lag_fix`, `exp30`).
- `docs/` — документация и аудит. `archive/` — старые эксперименты, пресеты (`archive/evns_old`), результаты (`archive/out`).
- `evns/` — действующие профили: `base.env` (стратегия) + `mubite_challenge.env` / `mubite_funded.env` / `personal.env`; собирает `ops/stage.sh`.
- `ops/` — `deploy.sh` (код на VPS), `stage.sh` (смена стадии), `bot.sh` (status/pause/resume/logs), `add_personal.sh` (второй бот).
- `research/fastsim*.py` — быстрый симулятор риск-слоя (Фаза T), `research/ladder.py` — экономика лестницы счетов.
- Не в git (см. `.gitignore`): `data/raw`, `data/panels`, `logs`, `*.npy`.

## Команды (из корня репо)
```
pip install -r requirements.txt
python3 dl_binance.py klines,fundingRate,premiumIndexKlines     # ~1.3 ГБ, ~15 мин
python3 build_panels.py "2026-09-30 23:00" core                  # затем ohlc, premium
python3 check_panels.py
python3 -m pytest v4/tests_v4.py                                 # нужны панели
```

## Железные правила
- Параметры из `docs/PREREG.md` не менять «по результатам демо». Любое изменение = новая регистрация с новым хэшем.
- Не коммитить секреты (`.env`, ключи). Перед пушем с VPS-данными — проверка на ключи.
- Из облака недоступны `api.bybit.com`, `fapi.binance.com`; доступны `data.binance.vision`, `public.bybit.com`, PyPI. Всё, что требует живого API Bybit, делается на VPS (netcup) скриптами; владелец только запускает их через Termius.
- Правила фирм меняются: перед покупкой челленджа сверять с официальным сайтом (Mubite: https://mubite.com/en/challengeRules).
