#!/usr/bin/env python3
"""Write PREREG.md: the pre-registered production configuration of book F2, hashed the way the bot hashes it at start-up
(config.effective_config -> EXPECTED_CONFIG_HASH), model hashes separately, the expected live ranges per mode and the
decision rules — fixed BEFORE the demo starts so that nothing decided after seeing demo data counts as a test."""
import hashlib, json, os, sys, subprocess, datetime
ROOT = os.path.dirname(os.path.abspath(__file__))
ENV = dict(SLEEVES='listing,core,ml3,ml7,ml8,fchg,fcarry', FUND_TOP_FRAC='0.2', SLEEVE_SCALES='listing:1,core:1,ml3:1,ml7:1,ml8:2,fchg:1,fcarry:1', ML_MODEL_SUFFIX='_noage', ML_DROP_FEATS='age_d',
           ML_MIN_AGE_D='90', ML8_TOP_FRAC='0.2', ML8_MIN_AGE_D='180', ML8_INV_VOL='1', REBAL_BAND='0.5', SMOOTH='0.5', REBAL_EVERY_H='8', REBAL_HOUR_UTC='1', VT_TARGET_DVOL='0.006',
           VT_MAX_LEV='3.0', VT_WIN_D='30', BETA_CAP='-1', NOTIONAL_BASE='initial', POS_CAP_PCT='2', SHORT_CAP_PCT='1.5', MAJOR_CAP_PCT='30', MAX_GROSS_X_EQ='2', STOP_SHORT_PCT='60',
           ML8_KILL_RULE='1', ML8_KILL_WIN_D='180', ML8_KILL_SOFT='0.5', ML8_KILL_HARD='0.0', ML8_KILL_RESTORE_SHARPE='1.0', ML8_COST_BPS_EST='5.0', ML8_KILL_MIN_GAP_D='30', ML8_KILL_RESTORE_D='60',
           MAKER_ATTEMPTS='3', MAKER_WAIT_S='90', CAND_POOL='350', LOWCAP_YOUNG_D='90', FUNDING_CHECK='1', FUNDING_ALERT_D='14', REQUIRE_SL='0',
           FIRM='mubite_8_dd10', ACCOUNT_SIZE='10000')
MODES = {'challenge_bold': dict(MODE='challenge', PHASE='1', BASE_SCALE='2.0', CPPI_FRAC='0.25', ABANDON_FRAC='0.30'),
         'challenge_moderate': dict(MODE='challenge', PHASE='1', BASE_SCALE='1.5', CPPI_FRAC='0.5', ABANDON_FRAC='0.30'),
         'funded': dict(MODE='funded', PHASE='1', BASE_SCALE='1.25', CPPI_FRAC='0.5', ABANDON_FRAC='0.0'),
         'funded_calm': dict(MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0'),   # first 60 funded days
         # HyroTrader 2-step with the Swing add-on (static daily DD), moderate scale — the second-firm candidate (PLAN §3.3)
         'challenge_hyro_swing': dict(FIRM='hyrotrader_2step_swing', MODE='challenge', PHASE='1', BASE_SCALE='1.5', CPPI_FRAC='0.5', ABANDON_FRAC='0.30'),
         # demo / canary on the own account at the research vol target
         'personal_canary': dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0',
                                 VT_TARGET_DVOL='0.006', VT_MAX_LEV='3.0', NOTIONAL_BASE='equity'),
         # own account (PLAN §5): no firm floors except the self-imposed 25% trailing stop; vol target 1.0%/day, equity base
         'personal_vt10': dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0',
                               VT_TARGET_DVOL='0.010', VT_MAX_LEV='4.0', NOTIONAL_BASE='equity', MAX_GROSS_X_EQ='3.5', POS_CAP_PCT='3', SHORT_CAP_PCT='2'),
         # own account, aggressive: 1.5%/day, leverage <= 5, gross <= 4x (PLAN §5 'максимальная' настройка)
         'personal_vt15': dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0',
                               VT_TARGET_DVOL='0.015', VT_MAX_LEV='5.0', NOTIONAL_BASE='equity', MAX_GROSS_X_EQ='4.0', POS_CAP_PCT='3', SHORT_CAP_PCT='2',
                               TOTAL_DD_PCT='35')}


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()[:16]


def cfg_hash(mode_env):
    env = dict(os.environ); env.update(ENV); env.update(mode_env); env.update(BYBIT_API_KEY='x', BYBIT_API_SECRET='x')
    out = subprocess.run([sys.executable, '-c', 'import sys; sys.path.insert(0, "v4"); import config as C; cfg, h = C.effective_config(); print(h, len(cfg))'],
                         cwd=ROOT, env=env, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(out.stderr)
    h, n = out.stdout.split()[-2:]
    return h, int(n)


hashes = {k: cfg_hash(v) for k, v in MODES.items()}
models = {f: sha(os.path.join(ROOT, 'models', f)) for f in ('ml_h3_noage.txt', 'ml_h7_noage.txt', 'ml_8h.txt', 'spec_noage.json', 'spec_8h.json') if os.path.exists(os.path.join(ROOT, 'models', f))}
model_hash = hashlib.sha256(json.dumps(models, sort_keys=True).encode()).hexdigest()[:16]
code = {f: sha(os.path.join(ROOT, 'v4', f)) for f in ('main.py', 'risk.py', 'signals.py', 'config.py', 'execution.py', 'ml_features.py', 'ml_features_8h.py', 'data.py') if os.path.exists(os.path.join(ROOT, 'v4', f))}
md = f"""# PREREG — пред-регистрация продакшн-конфигурации книги F2 (F + фандинг-рукава fchg, fcarry) ({datetime.date.today().isoformat()})

Всё ниже зафиксировано ДО демо-этапа. Любое изменение конфига после просмотра демо-данных — новая версия с новым хэшем и новым отсчётом теста.

## Три хэша (стратегия, модели, код) — проверяются раздельно

**1. Хэш стратегии+риска** — считается самим ботом при старте из всех непаролных параметров (`config.effective_config()`: рукава, веса, полосы, VT, капы,
правила фирмы, режим, BASE_SCALE/CPPI/ABANDON, исполнение). Бот пишет его в лог и в `/status` (`config_hash`) и **отказывается стартовать**, если задан
`EXPECTED_CONFIG_HASH` и он не совпадает. Хэш зависит от режима (challenge/funded) и от ACCOUNT_SIZE/FIRM — поэтому три значения:

| режим | env поверх базового | EXPECTED_CONFIG_HASH | ключей |
|---|---|---|---|
""" + '\n'.join(f'| {k} | `{json.dumps(v)}` | `{h}` | {n} |' for k, (v, (h, n)) in zip(MODES.keys(), zip(MODES.values(), hashes.values()))) + f"""

Базовый env (одинаков для всех режимов):
```
{json.dumps(ENV, indent=1, ensure_ascii=False)}
```

**2. Хэш моделей** `{model_hash}` (sha256/16 файлов): {json.dumps(models)}
Модели меняются ТОЛЬКО по процедуре переобучения (ниже); новая модель = новый хэш моделей, хэш стратегии не меняется.

**3. Код бота** (sha256/16, для аудита; правки кода, не меняющие стратегию — багфиксы исполнения/логов — допустимы и фиксируются в CHANGELOG):
{json.dumps(code)}

## Процедура переобучения (записана заранее)

* Периодичность: раз в 6 месяцев (1 января / 1 июля) ИЛИ по правилу 1 (падение вклада ml8) — не чаще.
* Данные: только бары до даты переобучения минус 8 дней (гэп на горизонт таргета), тот же набор признаков (spec_noage.json / spec_8h.json), те же гиперпараметры (exp6b_noage_final.py, exp17), без подбора по демо-результатам.
* Приёмка новой модели: walk-forward Sharpe рукава на последнем году ≥ 0.8 × старой модели на том же окне; иначе остаётся старая.
* Новая модель включается только на границе месяца, с записью хэша моделей в этот файл.

## Ожидаемые диапазоны на демо / первом счёте (окно оценки — 60 календарных дней, затем ежемесячно)

Нормированные метрики (не зависят от BASE_SCALE): все проверяются по `/status`.

| метрика | ожидание (любой режим) | источник |
|---|---|---|
| реализованная dvol equity / BASE_SCALE (`dvol_norm`) | 0.50–0.70 % | VT 0.6 %, exp23a |
| `mult_real` / BASE_SCALE (реализованный множитель книги к целевому) | 0.7–1.3 | harness |
| доля ребалансов с клиппингом гросса (`clip_share`) | < 5 % | exp23a clip 0 % |
| плечо VT (`lev`) | 0.3–2.0, без выходов на клипы 0.05 / 3.0 | exp23a lev p5 0.34 / p95 1.66 |
| время от закрытия бара до последнего филла (`last_fill_seconds`) | ≤ 900 с (3 попытки × 90 с + маркет) | exp21e: −0.3 Sharpe/час |
| доля мейкер-филлов по нотионалу (`maker_share_real`, реальные филлы) | ≥ 0.5 (на демо завышено — см. PLAN §4) | exp22d/23a |
| неблагоприятный отбор (`adverse_bps_real`, взвешенно по нотионалу) | ≤ 3 бп | exp26 S3: 2 бп = −0.1 Sharpe |
| вклад ml8 за скользящие 180 дней (Sharpe теневого вклада `ml8_shadow`) | > 0 | exp26 S1; harness kill-rule |
| число позиций / целей | 100–160, `n_positions ≈ n_targets` | harness |
| доля позитивных месяцев | ≥ 60 % (харнесс 68–70 %) | hF_pure* |

Абсолютные (зависят от режима): gross/equity в среднем по дню — challenge bold 1.0–2.6×, challenge moderate 0.8–2.0×, funded 0.6–1.6× (harness gross/equity by year).

## Правила решения (записаны заранее)

1. **Отключение 8h-рукава, версия 2 (ML8_KILL_WIN_D=180, ML8_KILL_SOFT=0.5, ML8_KILL_HARD=0.0, ML8_KILL_RESTORE_SHARPE=1.0, ML8_COST_BPS_EST=5, ML8_KILL_RESTORE_D=60, ML8_KILL_MIN_GAP_D=30):** мера — Sharpe *теневого* вклада ml8 (рукав на полном весе 2, независимо от действующего множителя) **за вычетом его оценённых издержек** (оборот теневого рукава × 5 бп) за 180 дней. Net Sharpe ≤ 0.5 → вес 2 → 1; ≤ 0 (или второй раз ≤ 0.5) → 0. Возврат веса на шаг — после 60 дней net Sharpe > 1.0; не чаще одного изменения в 30 дней. При множителе 0 бот переходит на дневной ребаланс (fcarry становится дневным). Зачем v2: версия 1 (порог 0 по валовому вкладу) на 2022–26 не сработала ни разу, т.е. ловила только гибель рукава, а не его увядание до «насоса комиссий» (Opus, раунд 4). Цена v2 в бэктесте бота — см. REPORT §5l.
2. **Исполнение:** `last_fill_seconds` медиана > 900 с два цикла подряд → MAKER_ATTEMPTS 3 → 2; > 1800 с → дневная книга (SLEEVES без ml8, REBAL_EVERY_H=24).
3. **Волтаргет:** `dvol_norm` вне 0.45–0.80 % за 30 дней или clip_share > 10 % → остановка и разбор (не подстройка параметров на лету).
4. **Отрицательные месяцы:** 3 отрицательных месяца подряд ИЛИ 6-месячный Sharpe < 0 → остановка книги, разбор атрибуции по рукавам; возобновление — только по заранее записанному критерию (вклад каждого рукава ≥ 0 за 90 дней). Частота срабатывания на истории 2022–26 — exp26 S6 (см. REPORT §5j).
5. **Правила фирмы:** любой сигнал от фирмы о «нерыночных филлах» / ограничении числа позиций — стоп до письменного разъяснения.
6. **Фандинг на счёте (автоматизировано, FUNDING_CHECK=1):** бот раз в сутки читает лог транзакций (`/v5/account/transaction-log`, расчёты SETTLEMENT / поле `funding`) и пишет в Telegram и `/status` сумму и число расчётов; если за FUNDING_ALERT_D=14 дней расчётов нет — алерт «счёт не начисляет фандинг → книга F» (SLEEVES без fchg/fcarry, отдельный хэш). Дополнительно раз в сутки считается базис фандинга Bybit↔Binance по 20 крупнейшим именам (`funding_basis`: корреляция и средняя разница в бп).
6a. **Тест книги на фандинге Bybit (до любой покупки дороже первого Mubite):** `fetch_bybit_funding.py` на VPS (публичный `/v5/market/funding/history`, 24 месяца, все имена юниверса) → `exp30_bybit_funding.py` (пересчёт F2 и F с сигналами и P&L на ставках Bybit на пересечении окон). Критерий приёмки записан заранее: Sharpe F2 на ставках Bybit ≥ 0.8 × Sharpe F2 на ставках Binance за то же окно и корреляция дневных сумм фандинга ≥ 0.7; иначе — книга F. Из среды разработки Bybit API недоступен (гео-блок CloudFront), поэтому тест выполняется на VPS.
7. **Что НЕ делается на ходу:** подбор весов рукавов, полос, квантилей, окна VT по демо-результатам. Такие изменения возможны только как новая пред-регистрация с новым хэшем.

## Ожидания по доходности (p25 / медиана; исследование ≠ обещание)

Медиана: Sharpe ~2.0–2.3 (исследование F2 2.49 при 70 % мейкер-филлов и исполнении в минуты; спад 8h-рукава учтён снижением), выплаты на funded ~22 %/г (харнесс с kill-rule). **p25-сценарий (по которому принимается решение о покупке):** Sharpe 1.0–1.2, выплаты 10–13 %/г, ступень лестницы 6–8 месяцев вместо 3. Решение покупать челлендж принимается не из-за ожидаемых 22 %/г, а из-за асимметрии ставки: риск ограничен сбором, апсайд — капитал под управлением, а информационная ценность живого исполнения выше любого дальнейшего бэктеста.
"""
open(os.path.join(ROOT, 'PREREG.md'), 'w', encoding='utf-8').write(md); print(md[:900]); print('hashes', hashes, 'models', model_hash)
