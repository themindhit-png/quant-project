#!/usr/bin/env python3
"""Offline tests for the VPS services (no systemd / Telegram / bots needed): python3 ops/tests_ops.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import opslib as L, tg_control as T

tmp = tempfile.mkdtemp()
ENV = {'propsleeves': os.path.join(tmp, 'bot.env'), 'propsleeves-personal': os.path.join(tmp, 'personal.env')}
open(ENV['propsleeves'], 'w').write(f'FIRM=mubite_8_dd10\nMODE=challenge\nPHASE=1\nACCOUNT_SIZE=10000\nSTATE_FILE={tmp}/p1.json\n'
                                     'TG_BOT_TOKEN=x\nTG_CHAT_ID=42\nPORT=8080\nSTATUS_TOKEN=s\n')
open(ENV['propsleeves-personal'], 'w').write(f'FIRM=personal\nMODE=funded\nACCOUNT_SIZE=5000\nSTATE_FILE={tmp}/pers.json\nPORT=8081\n')
L.bot_units = lambda: ['propsleeves', 'propsleeves-personal']
L.env_file = lambda u: ENV[u]
L.is_active = lambda u: True
L.status = lambda env, timeout=5: ({'equity': 10230.5, 'total_floor': 9000.0, 'trading_days': 7, 'profit_days': 4, 'k': 2.0, 'n_positions': 150,
                                    'n_targets': 152, 'last_cycle_utc': '2026-11-01T09:10:00'} if env.get('PORT') == '8080' else None)
res = []
def check(name, ok, info=''):
    res.append(ok); print(('PASS ' if ok else 'FAIL ') + name, info)

s = T.handle('/status')
check('status lists both bots with equity, floor buffer, days', '[prop]' in s and '[personal]' in s and '10,230.50' in s and 'до пола фирмы 1,230' in s
      and 'торговых дней 7, прибыльных 4' in s, repr(s[:200]))
r = T.handle('/pause')
check('pause creates the prop pause file', os.path.exists(f'{tmp}/p1.json.pause') and 'пауза включена' in r)
check('status shows the pause', 'ПАУЗА' in T.handle('/status prop'))
r = T.handle('/resume')
check('resume removes it', not os.path.exists(f'{tmp}/p1.json.pause') and 'пауза снята' in r)
T.handle('/pause personal')
check('pause personal targets the own-account bot', os.path.exists(f'{tmp}/pers.json.pause') and not os.path.exists(f'{tmp}/p1.json.pause'))
check('unknown bot / command answered', 'нет бота' in T.handle('/pause foo') and 'Команды' in T.handle('/what'))
check('tg_conf from the prop env', L.tg_conf() == ('x', '42'))
# collector: fake exchange, two runs an hour apart -> rows appended once, no duplicates, back-fill limited to BACKFILL_D
import gzip, importlib
os.environ['COLLECT_DIR'] = os.path.join(tmp, 'bybit'); os.environ['COLLECT_TOP'] = '2'
import collect_bybit as CB; importlib.reload(CB)
H = 3600 * 1000; NOW = [1_790_000_000_000 // H * H + 5 * 60 * 1000]
def fake_get(path, params, tries=4):
    if path.endswith('tickers'):
        return {'list': [{'symbol': 'BTCUSDT', 'turnover24h': '9'}, {'symbol': 'AAAUSDT', 'turnover24h': '5'}, {'symbol': 'X-USDT', 'turnover24h': '1'}]}
    st = params.get('start', params.get('startTime')); end = NOW[0] // H * H
    ts = list(range(st // H * H + (H if st % H else 0), end + 1, H))[:200]
    if 'kline' in path:
        return {'list': [[str(t), '1', '2', '0.5', '1.5', '10', '15'] for t in reversed(ts)]}
    if 'funding' in path:
        return {'list': [{'fundingRateTimestamp': str(t), 'fundingRate': '0.0001'} for t in reversed(ts) if t % (8 * H) == 0]}
    return {'list': [{'timestamp': str(t), 'openInterest': '100'} for t in reversed(ts)]}
CB.get = fake_get; CB.time.time = lambda: NOW[0] / 1000
CB.main(); NOW[0] += H; CB.main()
def rows(kind):
    out = []
    for f in sorted(os.listdir(os.path.join(tmp, 'bybit', kind))):
        out += [l for l in gzip.open(os.path.join(tmp, 'bybit', kind, f), 'rt') if not l.startswith('symbol')]
    return out
k = rows('kline'); keys = [tuple(l.split(',')[:2]) for l in k]
syms = {x[0] for x in keys}
check('collector: 7 days back-filled + 1 new hour per symbol, no duplicates, only USDT perps without "-"',
      len(keys) == len(set(keys)) and syms == {'BTCUSDT', 'ETHUSDT', 'AAAUSDT'} and len(keys) == 3 * (7 * 24 + 1) and len(rows('oi')) == len(keys)
      and 0 < len(rows('funding')) < len(keys), f'{len(keys)} rows {sorted(syms)}')

# stage.sh (dry run, PREREG reg. 4): the new env points the bot at the previous stage's state file for the ml8 kill-rule memory
import subprocess
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = os.path.join(tmp, 'app'); os.makedirs(os.path.join(app, 'src'), exist_ok=True); os.symlink(os.path.join(REPO, 'v4'), os.path.join(app, 'src', 'v4'))
old_state = os.path.join(tmp, 'trial_100000_x.json'); open(old_state, 'w').write('{}')
envf = os.path.join(tmp, 'stage.env')
open(envf, 'w').write(f'BYBIT_API_KEY=abcd\nBYBIT_API_SECRET=efgh\nTG_BOT_TOKEN=t\nPORT=8080\nSTATE_FILE={old_state}\nMODEL_DIR={REPO}/v4/models\n')
def stage(*a):
    e = dict(os.environ, STAGE_DRY='1', ENVF=envf, APP=app, BOTPY=sys.executable)
    return subprocess.run(['bash', os.path.join(REPO, 'ops', 'stage.sh'), *a], env=e, capture_output=True, text=True, timeout=120)
r = stage('phase1', '100000')
check('stage.sh carries the ml8 memory: ML8_CARRY_FROM = previous STATE_FILE, new STATE_FILE, hash computed',
      r.returncode == 0 and f'ML8_CARRY_FROM={old_state}' in r.stdout and 'STATE_FILE=' + app + '/state/phase1_100000_' in r.stdout
      and 'EXPECTED_CONFIG_HASH=' in r.stdout and 'abcd' not in r.stdout, (r.stdout[-300:] + r.stderr[-300:]).replace('\n', ' | '))
os.remove(old_state); r2 = stage('phase1', '100000')
check('stage.sh without an old state file writes no ML8_CARRY_FROM', r2.returncode == 0 and 'ML8_CARRY_FROM' not in r2.stdout, r2.stderr[-300:])

print(f'\n{sum(res)}/{len(res)} passed'); sys.exit(0 if all(res) else 1)
