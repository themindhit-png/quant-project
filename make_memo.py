#!/usr/bin/env python3
"""Build the operator's memo (memo.html, Russian, plain language) and the per-mode env files (envs/*.env).
Hashes are read from PREREG.md (single source) so the memo can never disagree with it. Run: python3 prereg.py && python3 make_memo.py"""
import re, os, html, datetime
ROOT = os.path.dirname(os.path.abspath(__file__))
pre = open(os.path.join(ROOT, 'PREREG.md'), encoding='utf-8').read()
HASH = {m.group(1): m.group(2) for m in re.finditer(r'^\| (\w+) \| `.*?` \| `([0-9a-f]{16})` \| \d+ \|', pre, flags=re.M)}
MODELS = re.search(r'\*\*2\. Хэш моделей\*\* `([0-9a-f]{16})`', pre).group(1)
BASE = """# --- книга F2 (стратегия; НЕ менять — иначе новый хэш и новая пред-регистрация) ---
SLEEVES=listing,core,ml3,ml7,ml8,fchg,fcarry
SLEEVE_SCALES=listing:1,core:1,ml3:1,ml7:1,ml8:2,fchg:1,fcarry:1
FUND_TOP_FRAC=0.2
ML_MODEL_SUFFIX=_noage
ML_DROP_FEATS=age_d
ML_MIN_AGE_D=90
ML8_TOP_FRAC=0.2
ML8_MIN_AGE_D=180
ML8_INV_VOL=1
REBAL_BAND=0.5
SMOOTH=0.5
REBAL_EVERY_H=8
REBAL_HOUR_UTC=1
REBAL_MINUTE=10
VT_TARGET_DVOL=0.006
VT_MAX_LEV=3.0
VT_WIN_D=30
BETA_CAP=-1
NOTIONAL_BASE=initial
POS_CAP_PCT=2
SHORT_CAP_PCT=1.5
MAJOR_CAP_PCT=30
MAX_GROSS_X_EQ=2
STOP_SHORT_PCT=60
ML8_KILL_RULE=1
ML8_KILL_WIN_D=180
ML8_KILL_SOFT=0.5
ML8_KILL_HARD=0.0
ML8_KILL_RESTORE_SHARPE=1.0
ML8_COST_BPS_EST=5.0
ML8_KILL_MIN_GAP_D=30
ML8_KILL_RESTORE_D=60
MAKER_ATTEMPTS=3
MAKER_WAIT_S=90
CAND_POOL=350
LOWCAP_YOUNG_D=90
FUNDING_CHECK=1
FUNDING_ALERT_D=14
REQUIRE_SL=0
ACCOUNT_SIZE=10000
# --- секреты и подключение (свои значения) ---
BYBIT_API_KEY=ВСТАВЬТЕ
BYBIT_API_SECRET=ВСТАВЬТЕ
BYBIT_BASE_URL=https://api-demo.bybit.com
TG_BOT_TOKEN=ВСТАВЬТЕ
TG_CHAT_ID=ВСТАВЬТЕ
STATUS_TOKEN=придумайте-длинный-пароль
STATE_FILE=/opt/propsleeves/state/state.json
STATE_BACKUP_FILE=/opt/propsleeves/state/state_backup.json
MODEL_DIR=/opt/propsleeves/models
PORT=8080
DRY_RUN=0
"""
MODES = {
    'demo':       ('Демо у фирмы (Bybit-демо, 3–4 недели)', dict(FIRM='mubite_8_dd10', MODE='challenge', PHASE='1', BASE_SCALE='2.0', CPPI_FRAC='0.25', ABANDON_FRAC='0.30'), 'challenge_bold',
                   'Тот же режим, что и на будущем челлендже — чтобы демо проверяло именно то, что будет торговаться. URL демо: https://api-demo.bybit.com.'),
    'canary':     ('Канарейка на своём счёте ($300–500, обязательно)', dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0', VT_TARGET_DVOL='0.006', VT_MAX_LEV='3.0', NOTIONAL_BASE='equity', BYBIT_BASE_URL='https://api.bybit.com'), 'personal_canary',
                   'Реальный счёт Bybit, реальные деньги, маленькая сумма. Смысл — измерить настоящую долю мейкер-филлов (maker_share_real) и неблагоприятный отбор (adverse_bps_real). Демо этого не покажет.'),
    'challenge':  ('Челлендж Mubite 2-step $10k с надбавками — bold', dict(FIRM='mubite_8_dd10', MODE='challenge', PHASE='1', BASE_SCALE='2.0', CPPI_FRAC='0.25', ABANDON_FRAC='0.30'), 'challenge_bold',
                   'Надбавки при покупке: Easier Target (8 %), Extra 2 % Drawdown Room, 90 % Profit Split. Итого ≈ $176. После прохождения фазы 1 бот остановится и напишет в Telegram — тогда PHASE=2 и перезапуск.'),
    'challenge_hyro': ('Челлендж HyroTrader 2-step + Swing — moderate', dict(FIRM='hyrotrader_2step_swing', MODE='challenge', PHASE='1', BASE_SCALE='1.5', CPPI_FRAC='0.5', ABANDON_FRAC='0.30'), 'challenge_hyro_swing',
                   'Только с надбавкой Swing (статическая дневная просадка). Без неё книга упирается в скользящую дневную просадку. Покупать только после прогонов и канарейки (см. план).'),
    'funded_calm': ('Funded — первые 60 дней (спокойно)', dict(FIRM='mubite_8_dd10', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0'), 'funded_calm',
                   'Первые два месяца на funded — множитель 1.0: запас до пола фирмы ещё маленький.'),
    'funded':     ('Funded — рабочий режим', dict(FIRM='mubite_8_dd10', MODE='funded', PHASE='1', BASE_SCALE='1.25', CPPI_FRAC='0.5', ABANDON_FRAC='0.0'), 'funded',
                   'Выплаты: медиана 28.7 %/год, p25 21.7 %/год (бэктест финальной версии бота). Запрашивайте выплату каждые 2–4 недели (у Mubite кап 5 % размера счёта за запрос).'),
    'personal_vt10': ('Личный счёт — рекомендуемый (1.0 %/день)', dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0', VT_TARGET_DVOL='0.010', VT_MAX_LEV='4.0', NOTIONAL_BASE='equity', MAX_GROSS_X_EQ='3.5', POS_CAP_PCT='3', SHORT_CAP_PCT='2', BYBIT_BASE_URL='https://api.bybit.com'), 'personal_vt10',
                   'Ожидание (2022–26): CAGR ≈ 55 %, просадка ≈ 14 %, худший день ≈ −4.4 %. Собственный стоп — 25 % от пика equity.'),
    'personal_vt15': ('Личный счёт — максимальная (1.5 %/день)', dict(FIRM='personal', MODE='funded', PHASE='1', BASE_SCALE='1.0', CPPI_FRAC='0.5', ABANDON_FRAC='0.0', VT_TARGET_DVOL='0.015', VT_MAX_LEV='5.0', NOTIONAL_BASE='equity', MAX_GROSS_X_EQ='4.0', POS_CAP_PCT='3', SHORT_CAP_PCT='2', TOTAL_DD_PCT='35', BYBIT_BASE_URL='https://api.bybit.com'), 'personal_vt15',
                   'Ожидание: CAGR ≈ 71 %, просадка ≈ 22 %, худший день ≈ −6 %. Дальше поднимать нет смысла — Sharpe уже падает из-за издержек. Собственный стоп — 35 %.'),
}
os.makedirs(os.path.join(ROOT, 'envs'), exist_ok=True)
open(os.path.join(ROOT, 'envs', 'base.env'), 'w', encoding='utf-8').write(BASE)
def env_text(mode):
    title, over, hkey, note = MODES[mode]
    lines = [f'# {title}', f'# скопируйте base.env и добавьте/замените строки ниже; хэш из PREREG.md ({hkey})']
    for k, v in over.items():
        lines.append(f'{k}={v}')
    lines.append(f'EXPECTED_CONFIG_HASH={HASH.get(hkey, "СМ. PREREG.md")}')
    txt = '\n'.join(lines) + '\n'
    open(os.path.join(ROOT, 'envs', f'{mode}.env'), 'w', encoding='utf-8').write(txt)
    return txt
ENVS = {m: env_text(m) for m in MODES}
def esc(t): return html.escape(t)
today = datetime.date.today().isoformat()
css = """
:root{--ink:#14213d;--muted:#5b6478;--line:#e2e6ee;--bg:#ffffff;--soft:#f4f6fa;--acc:#0b5fff;--ok:#0f8b4c;--warn:#b7791f;--bad:#c0392b}
*{box-sizing:border-box}body{margin:0;font:16px/1.55 -apple-system,Segoe UI,Roboto,Inter,Arial,sans-serif;color:var(--ink);background:var(--bg)}
.wrap{max-width:1040px;margin:0 auto;padding:32px 28px 80px}h1{font-size:30px;margin:0 0 6px}h2{font-size:22px;margin:44px 0 12px;padding-top:12px;border-top:2px solid var(--ink)}
h3{font-size:17px;margin:26px 0 8px}p{margin:8px 0}.lead{color:var(--muted);font-size:15px}
table{border-collapse:collapse;width:100%;margin:10px 0 16px;font-size:14px}th,td{border:1px solid var(--line);padding:7px 9px;vertical-align:top;text-align:left}th{background:var(--soft)}
code,pre{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:13px}pre{background:#0f172a;color:#e5e7eb;padding:14px 16px;border-radius:8px;overflow:auto;line-height:1.45}
code{background:var(--soft);padding:1px 5px;border-radius:4px}.box{border:1px solid var(--line);border-left:5px solid var(--acc);background:var(--soft);padding:12px 16px;border-radius:6px;margin:14px 0}
.box.ok{border-left-color:var(--ok)}.box.warn{border-left-color:var(--warn)}.box.bad{border-left-color:var(--bad)}
.toc{columns:2;column-gap:28px;font-size:14px;margin:14px 0 8px}.toc a{color:var(--acc);text-decoration:none}.toc a:hover{text-decoration:underline}
.step{display:grid;grid-template-columns:44px 1fr;gap:12px;margin:12px 0}.num{width:36px;height:36px;border-radius:50%;background:var(--ink);color:#fff;display:flex;align-items:center;justify-content:center;font-weight:700}
.small{font-size:13px;color:var(--muted)}ul{margin:6px 0 6px 20px}li{margin:3px 0}.tag{display:inline-block;padding:1px 8px;border-radius:12px;font-size:12px;background:var(--soft);border:1px solid var(--line);margin-right:4px}
@media print{pre{white-space:pre-wrap}}
"""
def pre_block(t): return f'<pre>{esc(t)}</pre>'
modes_html = ''
for m, (title, over, hkey, note) in MODES.items():
    modes_html += f'<h3 id="mode-{m}">{esc(title)}</h3><p>{esc(note)}</p>{pre_block(ENVS[m])}'
H = f"""<!DOCTYPE html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>PROP-SLEEVES v4 — памятка оператора</title><style>{css}</style></head><body><div class="wrap">
<h1>PROP-SLEEVES v4 — памятка оператора</h1>
<p class="lead">Что делать, в каком порядке, какие настройки для какого режима, как запускать и на что смотреть. Собрано {today}. Хэши конфигурации взяты автоматически из PREREG.md (хэш моделей <code>{MODELS}</code>). Подробности и обоснования — в PLAN.md, HOWITWORKS.md, REPORT.md, PREREG.md.</p>
<div class="toc">
<a href="#road">0. Дорожная карта</a><br><a href="#setup">1. Подготовка VPS и ключей</a><br><a href="#modes">2. Режимы и настройки (env)</a><br><a href="#run">3. Запуск и первые дни</a><br><a href="#change">4. Что менять и где</a><br><a href="#firms">5. Правила фирм: бот vs вы</a><br><a href="#funding">6. Фандинг: как проверить</a><br><a href="#ladder">7. План масштабирования</a><br><a href="#personal">8. Личный счёт</a><br><a href="#faq">9. Что делать, если…</a><br><a href="#gloss">10. Словарь</a>
</div>

<h2 id="road">0. Дорожная карта</h2>
<div class="step"><div class="num">0</div><div><b>Неделя 0–4: демо + канарейка (параллельно).</b> Демо у фирмы (бесплатно) в режиме <code>demo</code>; одновременно свой счёт Bybit $300–500 в режиме <code>canary</code>. Смотрим Telegram каждый день. Зелёный свет — чек-лист в разделе 3.</div></div>
<div class="step"><div class="num">1</div><div><b>Неделя 4–5: первая покупка.</b> Mubite 2-step $10 000 с тремя надбавками (Easier Target 8 %, Extra 2 % Drawdown, 90 % split) ≈ $176. Режим <code>challenge</code> (bold). Ожидание: funded через ~84 дня (медиана), 88 % шанс за полгода; резерв на вторую попытку $176.</div></div>
<div class="step"><div class="num">2</div><div><b>Параллельно: тест фандинга Bybit на VPS</b> (<code>fetch_bybit_funding.py</code> → <code>exp30_bybit_funding.py</code>, раздел 6). До его результата ничего дороже $176 не покупаем.</div></div>
<div class="step"><div class="num">3</div><div><b>Funded.</b> Первые 60 дней режим <code>funded_calm</code>, потом <code>funded</code>. Выплаты каждые 2–4 недели (кап Mubite 5 % за запрос). Сбор возвращается с первой выплатой.</div></div>
<div class="step"><div class="num">4</div><div><b>Вторая фирма — только после демо, канарейки и теста фандинга.</b> HyroTrader 2-step <b>+ Swing</b> (режим <code>challenge_hyro</code>; измерено: funded 23/25, медиана 176 дней, 95 % за год, 1.1 попытки — вдвое дольше Mubite), размер $25–50k. PTB — только тест на $10k (быстро проходится, но funded-правила и юрлицо не проверены). Помните: два счёта одного бота — одна ставка, не диверсификация.</div></div>
<div class="step"><div class="num">5</div><div><b>Рост.</b> Mubite: программа масштабирования (кнопка scaling в дашборде: +10 % за 120 дней → счёт ×2, до $1M) или покупка большего челленджа ($100k–200k) после закрытия малого. Потолок: $200k на фирму. Один счёт на фирму — всегда.</div></div>
<div class="step"><div class="num">6</div><div><b>Личный счёт.</b> Когда канарейка отработала месяц без сюрпризов — основной капитал в режиме <code>personal_vt10</code>; через 3 месяца метрик в норме — по желанию <code>personal_vt15</code>.</div></div>
<div class="box warn"><b>Три правила, которые нельзя нарушать:</b> (1) один счёт на фирму, никогда два счёта одной фирмы одновременно; (2) параметры стратегии (рукава, веса, полосы, волтаргет) не трогать — бот сам не стартует при несовпадении хэша; (3) первые $176 — единственный «слепой» риск; всё дороже — после измерений.</div>

<h2 id="setup">1. Подготовка VPS и ключей</h2>
<h3>1.1 Сервер</h3>
<p>Любой VPS вне США (Bybit блокирует американские IP): 2 vCPU, 4 GB RAM, Ubuntu 22.04/24.04, Франкфурт/Хельсинки/Амстердам. Оплата картой или криптой — на ваш выбор. Нужен только ради аптайма: бот должен работать 24/7.</p>
{pre_block('''sudo apt update && sudo apt install -y python3 python3-venv python3-pip unzip
sudo mkdir -p /opt/propsleeves/state && sudo chown -R $USER /opt/propsleeves && cd /opt/propsleeves
# скопируйте сюда prop_sleeves_v4_bundle.tar.gz (scp с компьютера) и распакуйте:
tar xzf prop_sleeves_v4_bundle.tar.gz
python3 -m venv venv && source venv/bin/activate
pip install -r v4/requirements.txt
python3 v4/tests_v4.py          # должно быть 44/44 passed (нужны данные исследования; если их нет на VPS — пропустите)''')}
<h3>1.2 Ключи Bybit</h3>
<ul>
<li><b>Демо фирмы:</b> фирма даёт Bybit-демо-счёт (или подключает ваш). В аккаунте Bybit → API → создать ключ <b>для демо-торговли</b> (Demo Trading API), права: <b>Contract Trade (Read + Write)</b>, без вывода средств. URL: <code>https://api-demo.bybit.com</code>.</li>
<li><b>Свой реальный счёт (канарейка, личный):</b> API-ключ основного аккаунта, права <b>Contract Trade + Read</b>, <b>без Withdraw</b>, привязка к IP вашего VPS. URL: <code>https://api.bybit.com</code>. Единый торговый аккаунт (UTA), режим маржи cross, позиции one-way — бот сам переключит one-way.</li>
<li>Ключи и секреты хранятся только в env-файле на VPS (права 600). В коде и логах их нет.</li>
</ul>
<h3>1.3 Telegram</h3>
<p>Создайте бота у @BotFather → токен → <code>TG_BOT_TOKEN</code>. Напишите боту любое сообщение, затем узнайте свой chat id (например, через @userinfobot) → <code>TG_CHAT_ID</code>. Бот присылает: запуск, каждый ребаланс, стопы, фандинг за сутки, правило отключения ml8, аварии, раз в 6 часов — копию состояния файлом.</p>

<h2 id="modes">2. Режимы и настройки</h2>
<p>Базовый файл <code>envs/base.env</code> одинаковый для всех режимов (это и есть стратегия). Режим = базовый файл + несколько строк поверх. Ниже — точные строки для каждого режима и хэш, который бот сверит при старте. Файлы уже лежат в <code>envs/</code> бандла.</p>
<h3>Базовый env (одинаков везде)</h3>{pre_block(BASE)}
{modes_html}
<div class="box"><b>Как понять, что режим собран правильно:</b> в логе первой строкой идёт <code>effective config hash XXXXXXXX (96 keys)</code>. Он должен совпасть с <code>EXPECTED_CONFIG_HASH</code> — иначе бот сам остановится и напишет в Telegram. Это защита от опечаток и от «подкрутить на ходу».</div>

<h2 id="run">3. Запуск и первые дни</h2>
<h3>3.1 Запуск</h3>
{pre_block('''cd /opt/propsleeves && source venv/bin/activate
cat envs/base.env envs/demo.env > .env          # соберите нужный режим: base + режим
nano .env                                        # вставьте ключи, токен, chat id, пароль статуса
set -a; source .env; set +a
python3 v4/main.py                               # первый запуск — руками, смотрим лог 2–3 минуты''')}
<p>Постоянный запуск через systemd (перезапуск при падении, старт после ребута):</p>
{pre_block('''sudo tee /etc/systemd/system/propsleeves.service >/dev/null <<'EOF'
[Unit]
Description=PROP-SLEEVES v4
After=network-online.target
[Service]
WorkingDirectory=/opt/propsleeves
EnvironmentFile=/opt/propsleeves/.env
ExecStart=/opt/propsleeves/venv/bin/python3 v4/main.py
Restart=always
RestartSec=20
[Install]
WantedBy=multi-user.target
EOF
sudo systemctl daemon-reload && sudo systemctl enable --now propsleeves
journalctl -u propsleeves -f                     # живой лог''')}
<h3>3.2 Что видно в первые часы</h3>
<ul>
<li>Telegram: «✅ PROP-SLEEVES v4 запущен | фирма | режим | правила…». Если вместо этого «🛑 Конфигурация не совпадает» — проверьте env.</li>
<li>Первый ребаланс — в ближайший слот 01:10 / 09:10 / 17:10 UTC: «📊 Ребаланс: equity …, целей 100–160, gross …, ордера maker/market …».</li>
<li>Страница статуса: <code>http://IP:8080/status?token=ВАШ_STATUS_TOKEN</code> — JSON со всеми метриками ниже.</li>
</ul>
<h3>3.3 Чек-лист «зелёного света» (демо ≥ 14 дней + канарейка)</h3>
<table><tr><th>Что</th><th>Где смотреть</th><th>Норма</th><th>Если не в норме</th></tr>
<tr><td>Фандинг начисляется</td><td>Telegram «💱 Фандинг за сутки: … (N расчётов)»; <code>/status.funding_settlements_24h</code></td><td>N &gt; 0 почти каждый день</td><td>14 дней N=0 → алерт; торгуем книгу F (раздел 4.3)</td></tr>
<tr><td>Скорость исполнения</td><td><code>orders.last_fill_seconds</code></td><td>≤ 900 с</td><td>&gt;900 два цикла подряд → MAKER_ATTEMPTS=2; &gt;1800 → дневная книга (правило 2 PREREG)</td></tr>
<tr><td>Доля мейкера (канарейка!)</td><td><code>orders.maker_share_real</code></td><td>≥ 0.5</td><td>ниже → режим moderate/calm вместо bold</td></tr>
<tr><td>Неблагоприятный отбор</td><td><code>orders.adverse_bps_real</code></td><td>≤ 3 бп</td><td>больше → как выше</td></tr>
<tr><td>Число позиций</td><td><code>n_positions</code>, <code>n_targets</code></td><td>100–160, примерно равны</td><td>сильно меньше → смотреть <code>data_bad</code>, <code>errors</code></td></tr>
<tr><td>Волатильность книги</td><td><code>vt</code>, <code>lev</code>, <code>mult_real</code>, <code>clip_share</code></td><td>dvol/BASE_SCALE 0.50–0.70 %; lev 0.3–2.0; clip_share &lt; 5 %</td><td>вне — стоп и разбор (правило 3 PREREG)</td></tr>
<tr><td>Базис фандинга Bybit↔Binance</td><td><code>funding_basis.corr</code> (понедельник в Telegram)</td><td>≥ 0.7</td><td>ниже → приоритет тесту exp30, возможна книга F</td></tr>
<tr><td>Ошибки данных</td><td><code>ok</code>, <code>error</code>, «REBALANCE ABORTED» в логе</td><td>нет</td><td>смотреть лог; чаще всего — API/интернет</td></tr>
</table>

<h2 id="change">4. Что менять и где</h2>
<h3>4.1 Можно менять (операционное, хэш не меняется)</h3>
<table><tr><th>Переменная</th><th>Что делает</th></tr>
<tr><td><code>BYBIT_BASE_URL</code>, ключи, <code>TG_*</code>, <code>STATUS_TOKEN</code>, <code>STATE_FILE</code>, <code>STATE_BACKUP_FILE</code>, <code>MODEL_DIR</code>, <code>PORT</code></td><td>подключение и хранение</td></tr>
<tr><td><code>DRY_RUN=1</code></td><td>всё считает, ордера не шлёт (проверка настроек)</td></tr>
<tr><td><code>KILL_SWITCH=1</code></td><td>аварийно закрыть всё и остановиться</td></tr>
<tr><td><code>RESET_TOTAL_STOP=1</code> / <code>RESET_TARGET_HIT=1</code> / <code>RESET_ABANDONED=1</code></td><td>снять соответствующий латч после разбора (одноразово, потом убрать из env)</td></tr>
<tr><td><code>STATE_JSON_B64</code></td><td>восстановить состояние из последней копии, присланной в Telegram (base64 файла)</td></tr>
</table>
<h3>4.2 Меняется по плану (каждое — новый режим и его хэш из PREREG.md)</h3>
<table><tr><th>Ситуация</th><th>Что менять</th></tr>
<tr><td>Прошли фазу 1 (бот остановился, Telegram «🎉 ФАЗА 1 ПРОЙДЕНА»)</td><td>дождаться подтверждения фирмы → <code>PHASE=2</code>, перезапуск (хэш тот же режима challenge)</td></tr>
<tr><td>Прошли фазу 2, счёт funded</td><td>файл режима <code>funded_calm</code> (60 дней) → затем <code>funded</code>; удалить старый state.json (новый счёт = новая память)</td></tr>
<tr><td>Другая фирма</td><td>файл режима <code>challenge_hyro</code> и т.п.; <code>ACCOUNT_SIZE</code> = размер счёта</td></tr>
<tr><td>Личный счёт</td><td>файлы <code>canary</code> → <code>personal_vt10</code> → <code>personal_vt15</code></td></tr>
</table>
<h3>4.3 Книга F вместо F2 (если фандинг не начисляется или тест exp30 провален)</h3>
{pre_block('''SLEEVES=listing,core,ml3,ml7,ml8
SLEEVE_SCALES=listing:1,core:1,ml3:1,ml7:1,ml8:2
# хэш для книги F: запустить  python3 prereg.py  с этими двумя строками в ENV (prereg.py) — или один раз запустить бота без EXPECTED_CONFIG_HASH,
# взять из лога строку "effective config hash …" и вписать её в env. Это единственный случай, когда хэш берётся из лога.''')}
<div class="box bad"><b>Нельзя менять без новой пред-регистрации:</b> рукава и веса, полосу ребаланса, сглаживание, волтаргет и его окно, капы, правило отключения ml8, время ребаланса, число попыток мейкера. Бот заметит (хэш), но смысл не в защите от бота, а в защите от самого себя: всё, что подкручено по промежуточным результатам, перестаёт быть проверенным.</div>

<h2 id="firms">5. Правила фирм: что делает бот, что делаете вы</h2>
<table><tr><th>Правило</th><th>Mubite (2-step)</th><th>HyroTrader (2-step + Swing)</th><th>Кто следит</th></tr>
<tr><td>Цели фаз</td><td>8 % (надбавка) + 5 %</td><td>10 % + 5 %</td><td>бот (фиксирует цель после минимума дней, закрывает всё и останавливается)</td></tr>
<tr><td>Макс. просадка</td><td>10 % статическая (надбавка)</td><td>10 % (режим не опубликован)</td><td>бот: пол + стоп-латч при 10–15 % запаса</td></tr>
<tr><td>Дневная просадка</td><td>5 %</td><td>5 % статическая только со Swing</td><td>бот: стоп при половине лимита, ежечасный скейл-пасс</td></tr>
<tr><td>Мин. торговых дней</td><td>13</td><td>5 + 5</td><td>бот считает; вы не выключаете бота на выходные</td></tr>
<tr><td>Убыток на позицию 3 %</td><td>да</td><td>да</td><td>бот (watchdog); позиции ≤ 2 %, почти не срабатывает</td></tr>
<tr><td>Экспозиция</td><td>funded: позиция ≤ 2×, всего ≤ 3× initial</td><td>маржа ≤ 25 %, нотионал ≤ 2× initial, low-cap ≤ 5 %</td><td>бот</td></tr>
<tr><td>Консистентность</td><td>—</td><td>40 % на один день в оценке</td><td>бот (дневной P&amp;L ровный по построению)</td></tr>
<tr><td>Стоп-лосс</td><td>не требуется</td><td>не обязателен</td><td>бот (свои стопы); <code>REQUIRE_SL=1</code> если потребуют</td></tr>
<tr><td>Несколько счетов / копирование</td><td><b>запрещено</b></td><td><b>запрещено</b></td><td><b>вы</b>: один счёт на фирму, никаких копий</td></tr>
<tr><td>KYC, выплаты, надбавки</td><td>KYC до первой выплаты; выплата ≤ 5 % за запрос</td><td>KYC после прохождения; мин. $100</td><td><b>вы</b>: KYC заранее, запросы выплат каждые 2–4 недели</td></tr>
<tr><td>Ручные сделки</td><td colspan="2">не делайте ничего руками на торговом счёте бота — любая ручная позиция ломает учёт нейтральности и атрибуцию</td><td><b>вы</b></td></tr>
</table>

<h2 id="funding">6. Фандинг: как проверить и почему это важно</h2>
<p><b>Что это.</b> Раз в 8 часов (у некоторых монет — 4 или 1 час) держатели лонгов и шортов обмениваются платой. Наша книга около половины результата получает именно фандингом (рукава fchg и fcarry). Если счёт фандинг не начисляет — книга F2 там не работает, работает книга F.</p>
<p><b>Где смотреть руками.</b> Bybit → Assets → Derivatives (USDT Perpetual) → Transaction History / История транзакций → фильтр типа <b>Funding Fee</b> (в API это тип <code>SETTLEMENT</code>). Каждая строка — одна выплата/списание по одной монете в момент расчёта 00:00 / 08:00 / 16:00 UTC. На реальном счёте они есть всегда, пока открыта позиция в момент расчёта. На демо — зависит от площадки.</p>
<p><b>Что делает бот.</b> Раз в сутки читает лог транзакций и пишет в Telegram «💱 Фандинг за сутки: +X USDT (N расчётов); с начала: …». Если за 14 дней N=0 — алерт «счёт не начисляет фандинг → книга F». Раз в сутки сравнивает фандинг Bybit и Binance по 20 крупнейшим монетам (<code>funding_basis</code>), по понедельникам — строка в Telegram.</p>
<p><b>Тест на VPS (обязателен до второй покупки).</b> Из среды разработки Bybit недоступен, поэтому:</p>
{pre_block('''cd /opt/propsleeves && source venv/bin/activate
python3 fetch_bybit_funding.py --months 24 --out data/bybit_funding.csv     # ~1 час, публичный API без ключа
# затем файл data/bybit_funding.csv отправляется на анализ (exp30_bybit_funding.py требует данных исследования — они не входят в бандл;
# запуск делает тот, у кого есть панели исследования). Критерий приёмки записан в PREREG 6a.''')}

<h2 id="ladder">7. План масштабирования (сжато; полностью — PLAN.md)</h2>
<table><tr><th>Этап</th><th>Когда</th><th>Действие</th><th>Из кармана</th><th>Ожидание (медиана / p25)</th></tr>
<tr><td>0a</td><td>месяц 0</td><td>демо у фирмы 3–4 недели</td><td>$0</td><td>книга F2 или F; исполнение</td></tr>
<tr><td>0b</td><td>месяц 0</td><td>канарейка на своём счёте</td><td>$300–500 (остаются вашими)</td><td>доля мейкера → bold или calm</td></tr>
<tr><td>0c</td><td>месяц 0–1</td><td>тест фандинга Bybit (VPS)</td><td>$0</td><td>F2 или F</td></tr>
<tr><td>1</td><td>месяц 1</td><td>Mubite 2-step $10k + 3 надбавки, bold</td><td>$176 (+$176 резерв)</td><td>funded за 84 / 156 дней; 88 % за 180 дней</td></tr>
<tr><td>2</td><td>месяцы 4–7</td><td>funded_calm 60 дней → funded; выплаты каждые 2–4 недели</td><td>0</td><td>$215 / $163 в месяц с $10k</td></tr>
<tr><td>3</td><td>после 0a–0c и первой выплаты</td><td>HyroTrader 2-step + Swing, bold, $25–50k; PTB $10k — тест</td><td>$300–450</td><td>funded за 176 / 229 дней; 95 % за год</td></tr>
<tr><td>4</td><td>через 3 стабильных месяца</td><td>рост Mubite: программа scaling или покупка $100k–200k после закрытия малого</td><td>из выплат</td><td>funded за 84 / 156 дней</td></tr>
<tr><td>5</td><td>месяц 12–18 / 18–30</td><td>$200k Mubite + $200k HyroTrader</td><td>≈ $2.5–3k сборов за весь путь</td><td>$100k / $76k в год (p10 $36k; «все по рынку» $48k)</td></tr>
</table>
<div class="box warn">Два счёта одного бота — одна ставка: плохие полгода ударят по обоим одновременно. Польза второй фирмы — только «одна не заплатит, вторая заплатит» и лимит $200k на пользователя.</div>

<h2 id="personal">8. Личный счёт</h2>
<table><tr><th>Режим</th><th>Волтаргет</th><th>Sharpe</th><th>CAGR</th><th>Просадка</th><th>Худший день</th><th>Когда</th></tr>
<tr><td><code>canary</code></td><td>0.6 %/день</td><td>2.38</td><td>33 %</td><td>−10 %</td><td>−3.0 %</td><td>первые 3–4 недели, $300–500</td></tr>
<tr><td><code>personal_vt10</code></td><td>1.0 %/день</td><td>2.46</td><td>55 %</td><td>−14 %</td><td>−4.4 %</td><td>основной капитал</td></tr>
<tr><td><code>personal_vt15</code></td><td>1.5 %/день</td><td>2.32</td><td>71 %</td><td>−22 %</td><td>−6.1 %</td><td>после 3 месяцев метрик в норме, если готовы к −22 %</td></tr>
<tr><td>стресс: всё по рынку, 1.0 %</td><td>1.0 %/день</td><td>1.49</td><td>29 %</td><td>−20 %</td><td>−4.1 %</td><td>что будет при плохом исполнении</td></tr>
</table>
<p>Как читать: это исторические числа 2022–26 при Sharpe ≈ 2.4; впереди честнее ждать Sharpe 2.2–2.3, а планировать по p25 (доходности примерно вдвое ниже, просадки — те же). Держите на бирже только торговый капитал, прибыль выше плана выводите раз в месяц. Правила остановки те же, что и на проп-счёте (3 отрицательных месяца подряд → стоп и разбор).</p>

<h2 id="faq">9. Что делать, если…</h2>
<table>
<tr><td><b>Бот остановился с «🛑 ОБЩИЙ СТОП»</b></td><td>Сработал стоп-латч у пола фирмы. Ничего не перезапускать «чтобы отыграться». Прочитать лог, проверить equity в дашборде фирмы. Если счёт жив и правила не нарушены — разбор причин, затем <code>RESET_TOTAL_STOP=1</code> на один запуск.</td></tr>
<tr><td><b>«🏳️ Челлендж сдан по правилу перезапуска»</b></td><td>Запас до пола стал меньше 30 % — быстрее купить новый челлендж, чем выползать. Это рассчитанное решение, не сбой. Резерв $176 — для этого.</td></tr>
<tr><td><b>«⚠️ Рукав(а) без целей» / «REBALANCE ABORTED»</b></td><td>Сломались данные или модель. Бот не торгует этот цикл и держит позиции. Если повторяется 2–3 цикла — проверить интернет/API, перезапустить.</td></tr>
<tr><td><b>Фирма спрашивает о «нетипичной торговле»</b></td><td>Остановить бота (systemctl stop), ответить: рыночно-нейтральный алгоритмический портфель из 100–160 позиций, лонги ≈ шорты, ребаланс 3 раза в день, без копирования и без второго счёта. Возобновлять — только после письменного ответа.</td></tr>
<tr><td><b>Фирма изменила правила</b></td><td>Сравнить с таблицей раздела 5; если изменился лимит — сменить пресет/режим (новый хэш) до следующего ребаланса, или остановить бота.</td></tr>
<tr><td><b>Потерялся state.json</b></td><td>Бот при старте восстановит якоря из лога транзакций биржи; полную память — из копии, присланной в Telegram: <code>STATE_JSON_B64=…</code> (base64 файла).</td></tr>
<tr><td><b>3 отрицательных месяца подряд или 6-месячный Sharpe &lt; 0</b></td><td>Правило 4 PREREG: стоп книги и разбор атрибуции по рукавам (<code>sleeve_attrib</code> в статусе). На истории такое случилось один раз за 4.7 года (май–июнь 2026).</td></tr>
<tr><td><b>Правило ml8 выключило рукав («⚙️ Правило отключения ml8 … → 0»)</b></td><td>Штатно: рукав перестал окупать издержки; бот переходит на дневной ребаланс и вернёт рукав сам, когда его чистый вклад снова &gt; 1.0 Sharpe за 180 дней.</td></tr>
<tr><td><b>Bybit API недоступен / ошибки 10006</b></td><td>Бот сам ждёт и повторяет; если больше часа — проверить VPS и ключ (IP-привязка).</td></tr>
</table>

<h2 id="gloss">10. Словарь</h2>
<p><span class="tag">equity</span> баланс + нереализованная прибыль. <span class="tag">пол</span> equity, ниже которого фирма закрывает счёт. <span class="tag">BASE_SCALE</span> множитель риска (1.0 спокойно, 1.25 funded, 2.0 bold). <span class="tag">волтаргет</span> целевая дневная качка equity (0.6 % на проп, 1.0–1.5 % на своём). <span class="tag">bold/moderate/calm</span> режимы челленджа 2.0 / 1.5 / 1.0. <span class="tag">F2 / F</span> книга с фандинг-рукавами / без них. <span class="tag">p25</span> четверть худших исходов — по ней принимаем решения. <span class="tag">хэш</span> отпечаток всех настроек стратегии; несовпадение = бот не стартует.</p>
<p class="small">Документы: PLAN.md (лестница и экономика), HOWITWORKS.md (как устроен бот), PREREG.md (пред-регистрация, правила остановки, хэши), REPORT.md (исследование), OPUS_REPLY.md (ответы ревьюеру), v4/README.md (технический). Бандл: prop_sleeves_v4_bundle.tar.gz.</p>
</div></body></html>"""
open(os.path.join(ROOT, 'memo.html'), 'w', encoding='utf-8').write(H)
print('memo.html', len(H), 'bytes; hashes', HASH)
