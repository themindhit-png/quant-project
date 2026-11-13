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

# ---------------------------------------------------------------- audit hardening 2026-11-13: B1 token scrub, F7 models swap / rollback / IC, F8 window + hash gate, F9 state_backup dir
import io, contextlib, requests, shutil, types
import numpy as np
import retrain as RT2, rebal_window as RW
tok = '123456789:AAHdqTcvCH1vGWJxfSeofSAs0K5PALDsaw0'
def _bad_post(url, **kw): raise requests.ConnectionError(f'HTTPSConnectionPool: Max retries exceeded with url: /bot{tok}/sendMessage')
_post0 = L.requests.post; L.requests.post = _bad_post; buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        L.tg_send('hello', tok, '42')
finally:
    L.requests.post = _post0
check('B1 tg_send: the bot token never reaches the log; scrub() covers the configured token and a token-shaped string',
      'tg fail' in buf.getvalue() and tok not in buf.getvalue() and tok not in L.scrub(f'x/bot{tok}/y', tok) and 'bot***' in L.scrub(f'/bot{tok}/y'), buf.getvalue()[:200])
check('B1 tg_control: errors are logged through scrub (getUpdates and the command reply)', open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tg_control.py')).read().count('L.scrub(e, token)') == 2)

# F7 models swap (rename), verify, rollback
def mk_dirs():
    base = tempfile.mkdtemp(); live, new = os.path.join(base, 'models'), os.path.join(base, 'models_new')
    os.makedirs(live); os.makedirs(new)
    for f in RT2.FILES:
        open(os.path.join(live, f), 'w').write('old-' + f); open(os.path.join(new, f), 'w').write('new-' + f)
    open(os.path.join(live, 'extra.txt'), 'w').write('keep')
    return base, live, new
rd = lambda d, f: open(os.path.join(d, f)).read()
H_new = lambda new: {f: RT2.sha12(os.path.join(new, f)) for f in RT2.FILES}
base, live, new = mk_dirs(); hn = H_new(new); rs = []
bak = RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T1', restart=lambda: rs.append(1), verify=lambda h: (True, ''))
check('F7 swap: new files in live (extra live files kept), the whole old directory is the backup, LIVE.new gone, one restart',
      rd(live, RT2.FILES[0]) == 'new-' + RT2.FILES[0] and rd(live, 'extra.txt') == 'keep' and rd(bak, RT2.FILES[0]) == 'old-' + RT2.FILES[0] and not os.path.exists(live + '.new') and len(rs) == 1, bak)
base, live, new = mk_dirs(); hn = H_new(new); rs = []; old_h = {f: RT2.sha12(os.path.join(live, f)) for f in RT2.FILES}
try:
    RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T2', restart=lambda: rs.append(1), verify=lambda h: (h != hn, '/status не отвечает'))
    msg = ''
except RT2.SwapFailed as e:
    msg = str(e)
check('F7 post-restart check fails -> automatic rollback: old models back (sha), both restarts done, message says the rollback was DONE and confirmed',
      rd(live, RT2.FILES[0]) == 'old-' + RT2.FILES[0] and {f: RT2.sha12(os.path.join(live, f)) for f in RT2.FILES} == old_h and len(rs) == 2 and 'откат ВЫПОЛНЕН' in msg and '/status не отвечает' in msg, msg)
base, live, new = mk_dirs(); hn = H_new(new); rs = []
try:
    RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T3', restart=lambda: rs.append(1), verify=lambda h: (False, 'нет ответа')); msg = ''
except RT2.SwapFailed as e:
    msg = str(e)
check('F7 rollback done but the old models are not confirmed after the 2nd restart: the message says so (no false "all fine")', rd(live, RT2.FILES[0]) == 'old-' + RT2.FILES[0] and 'не подтвердил' in msg and 'ВЫПОЛНЕН' not in msg, msg)
base, live, new = mk_dirs(); hn = H_new(new); _rm = RT2.restore_models; RT2.restore_models = lambda lv, bk: False
try:
    RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T4', restart=lambda: None, verify=lambda h: (False, 'x')); msg = ''
except RT2.SwapFailed as e:
    msg = str(e)
finally:
    RT2.restore_models = _rm
check('F7 rollback itself fails: the message says ОТКАТ НЕ УДАЛСЯ and gives the manual command with the backup path', 'ОТКАТ НЕ УДАЛСЯ' in msg and '.bak.T4' in msg and 'cp -a' in msg, msg)
base, live, new = mk_dirs(); os.remove(os.path.join(new, RT2.FILES[-1])); hn = {f: 'x' for f in RT2.FILES}
try:
    RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T5', restart=lambda: (_ for _ in ()).throw(AssertionError('no restart')), verify=None); msg = ''
except RT2.SwapFailed as e:
    msg = str(e)
check('F7 staging fails (a new file missing): live untouched, LIVE.new removed, message says the old models keep working, no restart',
      rd(live, RT2.FILES[0]) == 'old-' + RT2.FILES[0] and not os.path.exists(live + '.new') and 'не тронуты' in msg, msg)
base, live, new = mk_dirs(); hn = H_new(new); _ren = os.rename; calls_r = []
def _flaky(a, b):
    calls_r.append((a, b))
    if len(calls_r) == 2: raise OSError('busy')
    return _ren(a, b)
os.rename = _flaky
try:
    RT2.install_with_rollback(new, live, RT2.FILES, hn, 'T6', restart=lambda: (_ for _ in ()).throw(AssertionError('no restart')), verify=None); msg = ''
except RT2.SwapFailed as e:
    msg = str(e)
finally:
    os.rename = _ren
check('F7 second rename fails: the first is undone, old models in place, message truthful', rd(live, RT2.FILES[0]) == 'old-' + RT2.FILES[0] and os.path.isdir(live) and 'прежних моделях' in msg, msg)
# verify_running against fake /status
_st0 = L.status; _tt = [0.0]; _time0 = RT2.time.time; RT2.time.time = lambda: (_tt.__setitem__(0, _tt[0] + 0.1), _tt[0])[1]     # the collector test above froze time.time
exp = {'ml_8h.txt': 'aaaaaaaaaaaa'}
def st_ok(env, timeout=5): return {'ok': None, 'model_hashes': dict(exp)}
L.status = st_ok; r_ok = RT2.verify_running(exp, timeout=0.2, poll=0)
L.status = lambda env, timeout=5: (st_ok(env) if env.get('PORT') == '8080' else None); r_none = RT2.verify_running(exp, timeout=0.2, poll=0)
L.status = lambda env, timeout=5: {'ok': False, 'error': 'empty sleeves', 'model_hashes': dict(exp)}; r_bad = RT2.verify_running(exp, timeout=0.2, poll=0)
L.status = lambda env, timeout=5: {'ok': True, 'model_hashes': {'ml_8h.txt': 'bbbbbbbbbbbb'}}; r_hash = RT2.verify_running(exp, timeout=0.2, poll=0)
L.status = _st0; RT2.time.time = _time0
check('F7 verify_running: ok + matching model hashes -> True; no answer / ok=false / wrong hashes -> False with the reason',
      r_ok[0] and (not r_none[0]) and '/status не отвечает' in r_none[1] and (not r_bad[0]) and 'ok=false' in r_bad[1] and (not r_hash[0]) and 'хэши моделей' in r_hash[1], f'{r_none} {r_bad} {r_hash}')
# rank IC on held-out days
rng = np.random.default_rng(3); T_ = 1000; cff = 100 * np.cumprod(1 + 0.01 * rng.standard_normal((T_, 40)), axis=0); dfake = types.SimpleNamespace(cff=cff)
pn, po = {}, {}
for nm, step, Hh in (('h3', 24, 72), ('h7', 24, 168), ('8h', 8, 8)):
    pn[nm], po[nm] = {}, {}
    for i in range(T_ - 30 * 24, T_, step):
        if i + Hh <= T_ - 1:
            r = cff[i + Hh] / cff[i] - 1
            for j in range(40):
                pn[nm][(i, j)] = float(r[j]); po[nm][(i, j)] = -float(r[j])
ic_n, ic_o = RT2.heldout_ic(pn, dfake), RT2.heldout_ic(po, dfake)
check('F7 held-out rank IC: perfect predictions -> +1, inverted -> -1; horizon-aware (h3 last 14 d minus 3 d, 8h last 8 d); fmt_ic formats both',
      all(abs(ic_n[k][0] - 1) < 1e-9 and abs(ic_o[k][0] + 1) < 1e-9 and ic_n[k][1] >= 4 for k in ic_n) and ic_n['h7'][1] < ic_n['h3'][1] and '+1.000/-1.000' in RT2.fmt_ic({'new': ic_n, 'old': ic_o}), f'{ic_n} {ic_o}')
# F7 install.sh must not overwrite differing live models unless --models
REPO_ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
srcm, dstm = tempfile.mkdtemp(), tempfile.mkdtemp()
for f, a, b in (('a.txt', 'repo', 'live-retrained'), ('b.txt', 'same', 'same')):
    open(os.path.join(srcm, f), 'w').write(a); open(os.path.join(dstm, f), 'w').write(b)
open(os.path.join(srcm, 'c.txt'), 'w').write('new-file')
sh = lambda force: subprocess.run(['bash', os.path.join(REPO_, 'ops', 'sync_models.sh'), srcm, dstm, force], capture_output=True, text=True, timeout=60)
r0 = sh('0')
kept = rd(dstm, 'a.txt') == 'live-retrained' and rd(dstm, 'c.txt') == 'new-file' and 'НЕ перезаписана' in r0.stdout
r1 = sh('1'); baks = [d for d in os.listdir(os.path.dirname(dstm)) if d.startswith(os.path.basename(dstm) + '.bak.')]
check('F7 sync_models (install.sh): a differing live model is NOT overwritten without --models (a missing one is added); with --models it is replaced after a backup',
      kept and rd(dstm, 'a.txt') == 'repo' and len(baks) == 1 and rd(os.path.join(os.path.dirname(dstm), baks[0]), 'a.txt') == 'live-retrained', f'{r0.stdout} {r1.stdout}')
ins = open(os.path.join(REPO_, 'ops', 'install.sh')).read()
check('F7 install.sh: models go through sync_models.sh with the --models flag; no unconditional copy of repo models; state_backup directory created',
      'sync_models.sh' in ins and '--models) MODELS=1' in ins and 'cp -f "$REPO"/v4/models/*' not in ins and 'state_backup' in ins)
# F8 restart window and hash gate
env_w = {'REBAL_HOUR_UTC': '1', 'REBAL_MINUTE': '1', 'REBAL_EVERY_H': '12'}
clock = [datetime(2027, 1, 5, 13, 5, tzinfo=timezone.utc)] if False else None
from datetime import datetime, timezone, timedelta
clock = [datetime(2027, 1, 5, 13, 5, tzinfo=timezone.utc)]; naps = []
w_ok = RW.wait_outside(env_w, 45, now_fn=lambda: clock[0], sleep=lambda s: (naps.append(s), clock.__setitem__(0, clock[0] + timedelta(seconds=s))), say=lambda *a: None)
w_free = RW.wait_outside(env_w, 45, now_fn=lambda: datetime(2027, 1, 5, 3, 30, tzinfo=timezone.utc), sleep=lambda s: naps.append(('free', s)), say=lambda *a: None)
w_stuck = RW.wait_outside(env_w, 0, now_fn=lambda: datetime(2027, 1, 5, 13, 5, tzinfo=timezone.utc), sleep=lambda s: None, say=lambda *a: None)
check('F8 rebal_window: at 13:05 it waits (30 s naps) until 13:31 (+30 min window), a free time returns at once, a window that does not end in time returns False',
      w_ok and clock[0] >= datetime(2027, 1, 5, 13, 31, tzinfo=timezone.utc) and clock[0] < datetime(2027, 1, 5, 13, 33, tzinfo=timezone.utc) and w_free and ('free', 30) not in naps and not w_stuck, str(clock[0]))
dep = open(os.path.join(REPO_, 'ops', 'deploy.sh')).read(); stg_sh = open(os.path.join(REPO_, 'ops', 'stage.sh')).read()
check('F8 deploy.sh: a changed hash is accepted only with --accept-new-hash (the gate comes before any restart); deploy.sh and stage.sh restart via rebal_window.py',
      '--accept-new-hash) ACCEPT=1' in dep and dep.index('HASH_CHANGE -eq 0 || $ACCEPT -eq 1') < dep.index('systemctl restart "$UNIT"') and 'rebal_window.py' in dep and 'rebal_window.py' in stg_sh
      and stg_sh.index('rebal_window.py') < stg_sh.index('systemctl restart "$UNIT"'))
rr = stage('phase1', '100000')
check('F9 stage.sh: the state backup copy goes to APP/state_backup/ (another directory than APP/state/)', rr.returncode == 0 and f'STATE_BACKUP_FILE={app}/state_backup/phase1_100000_' in rr.stdout, rr.stdout[-300:])

# retrain.py (12.11): restart windows follow the bot env (reg. 5c 01:01/13:01), heavy services are paused and always resumed
import retrain as RT
from datetime import datetime as _dt, timezone as _tz
sl = RT.rebal_slots({'REBAL_HOUR_UTC': '1', 'REBAL_MINUTE': '1', 'REBAL_EVERY_H': '12'})
at = lambda h, m: _dt(2027, 1, 5, h, m, tzinfo=_tz.utc)
check('retrain: slots 01:01/13:01 from env; 13:05 and 00:59 blocked, 03:30 and 13:40 free; old default 01/09/17:10',
      sl == [61, 781] and RT.in_rebal_window(at(13, 5), sl) and RT.in_rebal_window(at(0, 59), sl) and not RT.in_rebal_window(at(3, 30), sl)
      and not RT.in_rebal_window(at(13, 40), sl) and RT.rebal_slots({}) == [70, 550, 1030], str(sl))
calls = []
class _DT:
    @staticmethod
    def now(tz=None): return at(3, 45)
RT.datetime = _DT; RT.time.sleep = lambda s: None
RT.subprocess.run = lambda cmd, **kw: calls.append(cmd) or type('R', (), {'returncode': 0, 'stdout': '', 'stderr': ''})()
_act = {'edge-bot': True}; RT.L.is_active = lambda u: _act.get(u, False)
os.environ['RETRAIN_PAUSE_UNITS'] = 'edge-bot,absent-unit'
pz = RT.pause_heavy(); RT.resume_heavy(pz)
flat = [' '.join(c) for c in calls]
check('retrain: edge-bot gets a 12 h dead-man timer, is stopped, then started and the timer removed; inactive units untouched',
      len(pz) == 1 and pz[0][0] == 'edge-bot' and flat[0].startswith('systemd-run --unit=edge-bot-resume-') and '--on-active=12h' in flat[0]
      and flat[1] == 'systemctl stop edge-bot' and flat[2] == 'systemctl start edge-bot' and flat[3].endswith('.timer')
      and not any('absent-unit' in f for f in flat), str(flat))
os.environ.pop('RETRAIN_PAUSE_UNITS'); calls.clear(); pz = RT.pause_heavy()
check('retrain: by default nothing is paused (one behaviour for demo, prop and own account)', pz == [] and not calls, str(calls))
L.is_active = lambda u: True

print(f'\n{sum(res)}/{len(res)} passed'); sys.exit(0 if all(res) else 1)
