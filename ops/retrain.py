#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic model retrain on the VPS (Phase 5.2; timer propsleeves-retrain: 5 January and 5 July, 03:30 UTC). Runs in the research venv
from the repo clone (/root/quant-project). Steps:
  1. resource check (RAM / disk);  2. Binance Vision: symbol list + incremental download (klines, funding, premium);
  3. panels up to the cutoff (1st of the month 00:00 UTC) + check;  4. final models into models_new/: h3/h7 (TRAIN_END = cutoff - 14 d,
     PHASE4_LOG «4.2 ревью»: gap h + 7 days, the h7 gap is used for both) and 8h (TRAIN_END = cutoff - 8 d);
  5. gates (no tuning — the models have no honest OOS): v4/tests_v4.py with the new models (feature parity T25/T25c, sleeves T4/T26);
     predictions not degenerate; Spearman new vs current models over the last 30 days >= 0.5;
  6. swap (F7, audit 2026-11): the new models are copied into LIVE.new (a copy of the live directory), every file is verified by sha, then the directories
     are swapped by rename (live -> LIVE.bak.<time>, LIVE.new -> live); any failure puts the backup back and the message says exactly what state the
     bot is in. Bot restart outside a rebalance window, then /status of each bot must answer (ok is not false, model hashes = the new files): otherwise
     the old models are restored and the bots restarted again (automatic rollback). Telegram report: hashes, Spearman, train ends and the rank IC of the
     new vs the old models on the held-out last 14 days (information only, not a gate).
Memory (VPS 2 GB): by default NOTHING else is stopped - every bot keeps trading (demo, prop and own account alike); the edge-lab bot fits next to
the retrain with MEM_LEAN=1 (~0.7 GB), the rest goes to swap (slower retrain, Nice=10). Optional emergency switch: RETRAIN_PAUSE_UNITS=unit[,unit]
stops those units for the retrain and always starts them again (end, failure, SIGTERM; a 12 h systemd timer if this process is killed).
Any failure -> the live models stay, the reason goes to Telegram. Manual: python3 ops/retrain.py [--cutoff 2027-01-01] [--dry] [--skip-data]"""
import os, sys, json, time, shutil, hashlib, argparse, subprocess, glob, signal
from datetime import datetime, timezone, timedelta
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ops'))
import opslib as L

GAP_HL_D, GAP_8H_D = 14, 8
LIVE = os.environ.get('LIVE_MODELS', os.path.join(L.APP, 'models'))
NEW = os.path.join(REPO, 'models_new')
FILES = ('ml_h3_noage.txt', 'ml_h7_noage.txt', 'spec_noage.json', 'ml_8h.txt', 'spec_8h.json')
LOG = []
HELD_DAYS = {'h3': 14, 'h7': 14, '8h': 8}              # models are trained to cutoff-14 d (h3/h7) / cutoff-8 d (8h): the days after that are held out
HORIZON_H = {'h3': 72, 'h7': 168, '8h': 8}              # forward-return horizon of each model, hours


def say(msg):
    line = f'[{datetime.now(timezone.utc):%Y-%m-%d %H:%M}Z] {msg}'; print(line, flush=True); LOG.append(line)


def run(cmd, env=None, timeout=4 * 3600):
    say('$ ' + ' '.join(cmd))
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, cwd=REPO, env=e, capture_output=True, text=True, timeout=timeout)
    tail = (r.stdout + r.stderr)[-1500:]
    if r.returncode != 0:
        raise RuntimeError(f'{cmd[1] if len(cmd) > 1 else cmd[0]} failed:\n{tail}')
    return r.stdout


def sha12(p):
    return hashlib.sha256(open(p, 'rb').read()).hexdigest()[:12]


def resources():
    mem = {k: int(v.split()[0]) for k, v in (l.split(':', 1) for l in open('/proc/meminfo'))}
    avail_gb = (mem.get('MemAvailable', 0) + mem.get('SwapFree', 0)) / 1e6; free_gb = shutil.disk_usage(REPO).free / 1e9
    say(f'resources: RAM+swap available {avail_gb:.1f} GB (swap {mem.get("SwapFree", 0) / 1e6:.1f}), disk free {free_gb:.1f} GB')
    if avail_gb < float(os.environ.get('RETRAIN_MIN_RAM_GB', '3.0')) or free_gb < 5:
        raise RuntimeError(f'не хватает ресурсов (RAM+подкачка {avail_gb:.1f} ГБ, диск {free_gb:.1f} ГБ; нужно ~3 ГБ и 5 ГБ диска)')


def vision_symbols():
    """All USDT-M perpetual symbols with monthly 1h klines on Binance Vision (incl. delisted) -> data/symbols_vision.json."""
    import re, requests
    base = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
    out, marker = [], ''
    for _ in range(200):
        r = requests.get(base, params={'prefix': 'data/futures/um/monthly/klines/', 'delimiter': '/', 'marker': marker}, timeout=60).text
        pre = re.findall(r'<Prefix>data/futures/um/monthly/klines/([^/<]+)/</Prefix>', r)
        out += pre
        if '<IsTruncated>true' not in r or not pre:
            break
        marker = f'data/futures/um/monthly/klines/{pre[-1]}/'
    syms = sorted({s for s in out if s.endswith('USDT') and s.isascii()})
    if len(syms) < 300:
        raise RuntimeError(f'Binance Vision: только {len(syms)} символов в списке — что-то не так')
    p = os.path.join(REPO, 'data', 'symbols_vision.json'); os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump({'klines': syms}, open(p, 'w')); say(f'symbols: {len(syms)}')
    return p


def predictions(model_dir, days=30):
    """Daily (h3/h7) and 8h predictions of the models in model_dir over the last `days` days of the panels, {name: {(i, sym): pred}}."""
    import numpy as np, lightgbm as lgb
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, 'v4'))
    from bt import Data
    from ml_features import compute_features, rank_norm, NEED_H
    from ml_features_8h import compute_features_8h, FEATURES_8H, NEED_H as NEED8
    spec = json.load(open(os.path.join(model_dir, 'spec_noage.json')))
    d = predictions.d = getattr(predictions, 'd', None) or Data(start='2021-01-01', end=None, min_turn_ever=5e6, load_hl=True, load_premium=True)
    T = d.ret.shape[0]; jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); out = {}; d.prev_mask = None
    for name, hz, grid in (('h3', 3, 24), ('h7', 7, 24), ('8h', 8, 8)):
        f = os.path.join(model_dir, f'ml_h{hz}_noage.txt' if hz != 8 else 'ml_8h.txt'); mdl = lgb.Booster(model_file=f); res = {}
        for i in [i for i in range(T - days * 24, T) if d.hh[i] % grid == 0]:
            m = d.universe(i, min_age_h=(90 if hz != 8 else 180) * 24, min_turn=1e7, top_n=150); js = np.flatnonzero(m)
            if hz == 8:
                sl = slice(i - NEED8 + 1, i + 1)
                F = compute_features_8h(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jb, (d.hh[i] + 1) % 24)
                X = np.column_stack([F[c][js] for c in FEATURES_8H])
            else:
                sl = slice(i - NEED_H + 1, i + 1)
                F = compute_features(d.cff[sl], d.fund[sl], d.t24[sl], d.prem[sl], d.high[sl], d.low[sl], d.age[i].astype(float), jb)
                X = np.column_stack([F[c][js] for c in spec['features']])
            p = mdl.predict(rank_norm(X.astype(np.float32)).astype(np.float64))
            res.update({(i, int(j)): float(v) for j, v in zip(js, p)})
        out[name] = res
    return out


def gates(cutoff):
    import numpy as np, pandas as pd
    for f in FILES:
        if not os.path.exists(os.path.join(NEW, f)):
            raise RuntimeError(f'нет файла новой модели {f}')
    run([sys.executable, 'v4/tests_v4.py'], env={'MODEL_DIR': NEW}, timeout=3600); say('gate 1: tests_v4 with the new models — PASS')
    new, old = predictions(NEW), predictions(LIVE); rep = {}
    for k in new:
        keys = [x for x in new[k] if x in old[k]]
        a = pd.DataFrame({'i': [x[0] for x in keys], 'n': [new[k][x] for x in keys], 'o': [old[k][x] for x in keys]})
        if len(a) < 500 or not np.isfinite(a.n).all() or a.groupby('i').n.std().min() <= 0:
            raise RuntimeError(f'gate 2: прогнозы {k} вырождены или их мало ({len(a)})')
        rho = a.groupby('i').apply(lambda g: g[['n', 'o']].corr(method='spearman').iloc[0, 1]).mean(); rep[k] = round(float(rho), 3)
        if not rho >= 0.5:
            raise RuntimeError(f'gate 3: Spearman новой и текущей модели {k} = {rho:.2f} < 0.5 — что-то сломано в данных')
    say(f'gates 2-3: predictions OK, Spearman new vs live {rep}')
    ic = {'new': heldout_ic(new, predictions.d), 'old': heldout_ic(old, predictions.d)}
    say(f'held-out rank IC (information, not a gate): {ic}')
    return rep, ic


def heldout_ic(preds, d, held=None):
    """Mean daily rank IC (Spearman of the prediction vs the realised forward return over the model's horizon, across names) on the held-out days:
    the last HELD_DAYS[model] days of the data, decisions whose forward window ends inside the data. -> {model: (ic|None, n_decisions)}."""
    import numpy as np, pandas as pd
    T = d.cff.shape[0]; out = {}
    for name, res in preds.items():
        H = HORIZON_H[name]; lo = T - (held or HELD_DAYS)[name] * 24; by = {}
        for (i, j), p in res.items():
            if i >= lo and i + H <= T - 1:
                by.setdefault(i, []).append((j, p))
        ics = []
        for i, lst in by.items():
            js = np.array([j for j, _ in lst]); p = np.array([v for _, v in lst], float)
            r = d.cff[i + H, js] / d.cff[i, js] - 1.0
            ok = np.isfinite(r) & np.isfinite(p)
            if ok.sum() >= 20:
                ic = pd.Series(p[ok]).rank().corr(pd.Series(r[ok]).rank())
                if np.isfinite(ic):
                    ics.append(float(ic))
        out[name] = (round(float(np.mean(ics)), 4) if ics else None, len(ics))
    return out


def fmt_ic(ic):
    f = lambda t: 'н/д' if t[0] is None else f'{t[0]:+.3f}'
    return ', '.join(f'{k} {f(ic["new"][k])}/{f(ic["old"][k])} (n={ic["new"][k][1]})' for k in ic['new'])


def rebal_slots(env):
    """Rebalance start times of one bot as minutes of the day, from its env (same defaults as v4/config.py)."""
    h0 = int(env.get('REBAL_HOUR_UTC') or 1); m0 = int(env.get('REBAL_MINUTE') or 10)
    every = int(env.get('REBAL_EVERY_H') or (8 if 'ml8' in (env.get('SLEEVES') or 'ml8') else 24))
    return sorted({((h0 + i * every) % 24) * 60 + m0 for i in range(max(1, 24 // every))})


def in_rebal_window(now, slots, before=3, after=30):
    """True from `before` min before a rebalance slot to `after` min after it (orders + fills take ~5 min)."""
    x = now.hour * 60 + now.minute
    return any(-before <= ((x - s + 720) % 1440) - 720 < after for s in slots)


class SwapFailed(RuntimeError):
    """The swap / post-restart check failed; the message is the complete, truthful Telegram text (state of the live models and of the bots)."""


def stage_models(new_dir, live, files):
    """LIVE.new = copy of the live directory + the new files, every new file verified by sha against new_dir. Live is untouched. -> path."""
    stg = live.rstrip('/') + '.new'
    shutil.rmtree(stg, ignore_errors=True)
    shutil.copytree(live, stg)
    for f in files:
        shutil.copy2(os.path.join(new_dir, f), os.path.join(stg, f))
    for f in files:
        if sha12(os.path.join(stg, f)) != sha12(os.path.join(new_dir, f)):
            raise RuntimeError(f'контрольная сумма {f} после копирования не совпала')
    return stg


def swap_models(live, stg, stamp):
    """Swap by rename: live -> live.bak.<stamp>, stg -> live. If the second rename fails the first is undone. -> backup path."""
    live = live.rstrip('/'); bak = f'{live}.bak.{stamp}'
    os.rename(live, bak)
    try:
        os.rename(stg, live)
    except Exception as e:
        try:
            os.rename(bak, live)
        except Exception as e2:
            raise SwapFailed(f'⚠️ Замена моделей оборвалась ({e}); прежние модели лежат в {bak}, а {live} НЕ на месте и не восстановлен ({e2}). '
                             f'Выполни вручную: mv {bak} {live} && systemctl restart propsleeves') from e2
        raise RuntimeError(f'замена каталога моделей не удалась ({e}); прежние модели возвращены на место')
    return bak


def restore_models(live, bak):
    """Put the backup back as the live directory (the failed one is kept as live.failed.<time>). True on success."""
    live = live.rstrip('/')
    try:
        if os.path.exists(live):
            os.rename(live, f'{live}.failed.{int(time.time())}')
        os.rename(bak, live)
        return True
    except Exception as e:
        say(f'restore failed: {e}')
        return False


def verify_running(expected, timeout=300, poll=10):
    """After a restart: every active bot answers /status with ok not false and the expected model hashes ({file: sha12}). -> (ok, reason)."""
    units = [u for u in L.bot_units() if L.is_active(u)]
    if not units:
        return True, 'активных ботов нет, проверять нечего'
    end = time.time() + timeout; why = {}
    pending = set(units)
    while pending:
        for u in list(pending):
            st = L.status(L.read_env(L.env_file(u) or ''))
            if st is None:
                why[u] = '/status не отвечает'
            elif st.get('ok') is False:
                why[u] = f'/status ok=false ({st.get("error") or st.get("guard_reason") or "?"})'
            else:
                got = st.get('model_hashes') or {}
                bad = [f for f, h in expected.items() if got.get(f) != h]
                if bad:
                    why[u] = f'хэши моделей в /status не те: {bad}'
                else:
                    pending.discard(u); why.pop(u, None)
        if not pending or time.time() >= end:
            break
        time.sleep(poll)
    return (not pending), '; '.join(f'{L.short(u)}: {r}' for u, r in why.items())


def install_with_rollback(new_dir, live, files, hashes, stamp, restart=None, verify=None):
    """Stage -> swap -> restart -> verify; on any failure restore the backup, restart and verify the old models. Returns the backup path.
    Raises SwapFailed with a truthful complete message when the new models are NOT in use."""
    restart = restart or safe_restart; verify = verify or verify_running
    old_hashes = {f: sha12(os.path.join(live, f)) for f in files if os.path.exists(os.path.join(live, f))}
    try:
        stg = stage_models(new_dir, live, files)
    except Exception as e:
        shutil.rmtree(live.rstrip('/') + '.new', ignore_errors=True)
        raise SwapFailed(f'⚠️ Переобучение: не удалось подготовить новые модели ({e}). Живые модели не тронуты, бот работает на прежних.') from e
    try:
        bak = swap_models(live, stg, stamp)
    except SwapFailed:
        raise
    except Exception as e:
        raise SwapFailed(f'⚠️ Переобучение: {e}. Бот работает на прежних моделях (бот не перезапускался).') from e
    say(f'models replaced (backup {bak}); new hashes {hashes}')
    try:
        restart(); ok, why = verify(hashes)
    except Exception as e:
        ok, why = False, f'{type(e).__name__}: {e}'
    if ok:
        return bak
    say(f'post-restart check FAILED ({why}); rolling back')
    if not restore_models(live, bak):
        raise SwapFailed(f'🛑 Новые модели не прошли проверку после перезапуска ({why}), а ОТКАТ НЕ УДАЛСЯ. Прежние модели: {bak}. '
                         f'Выполни вручную: rm -rf {live} && cp -a {bak} {live} && systemctl restart propsleeves')
    try:
        restart(); ok2, why2 = verify(old_hashes)
    except Exception as e:
        ok2, why2 = False, f'{type(e).__name__}: {e}'
    if ok2:
        raise SwapFailed(f'⚠️ Новые модели не прошли проверку после перезапуска ({why}). Автоматический откат ВЫПОЛНЕН: боты снова на прежних моделях '
                         f'(/status подтверждён).')
    raise SwapFailed(f'🛑 Новые модели не прошли проверку ({why}). Прежние модели возвращены в {live}, но после повторного перезапуска /status '
                     f'не подтвердил их ({why2}). Проверь бота: bash ops/bot.sh status')


def safe_restart():
    """Restart the bots outside their rebalance windows (slots from each bot's env: reg. 5c = 01:01 / 13:01 UTC)."""
    units = [u for u in L.bot_units() if L.is_active(u)]
    slots = sorted({s for u in units for s in rebal_slots(L.read_env(L.env_file(u) or ''))}) or rebal_slots({})
    while in_rebal_window(datetime.now(timezone.utc), slots):
        time.sleep(60)
    for u in units:
        subprocess.run(['systemctl', 'restart', u]); say(f'restarted {u}')


def pause_heavy():
    """Stop other memory-heavy services for the retrain (between their hourly cycles); a 12 h dead-man timer starts them again."""
    paused = []
    for u in [x.strip() for x in os.environ.get('RETRAIN_PAUSE_UNITS', '').split(',') if x.strip()]:
        if not L.is_active(u):
            continue
        while 8 <= datetime.now(timezone.utc).minute < 40:          # the paper bot cycles at hh:10 - do not cut a cycle
            time.sleep(60)
        dm = f'{u}-resume-{int(time.time())}'
        subprocess.run(['systemd-run', f'--unit={dm}', '--on-active=12h', '/bin/systemctl', 'start', u], capture_output=True, text=True)
        subprocess.run(['systemctl', 'stop', u]); paused.append((u, dm)); say(f'paused {u} for the retrain (memory)')
    return paused


def resume_heavy(paused):
    for u, dm in paused:
        subprocess.run(['systemctl', 'start', u]); say(f'resumed {u}: {"active" if L.is_active(u) else "NOT active"}')
        subprocess.run(['systemctl', 'stop', dm + '.timer'], capture_output=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cutoff', default=None); ap.add_argument('--dry', action='store_true')
    ap.add_argument('--skip-data', action='store_true', help='use the existing data/panels (testing)')
    A = ap.parse_args()
    now = datetime.now(timezone.utc)
    cutoff = datetime.fromisoformat(A.cutoff).replace(tzinfo=timezone.utc) if A.cutoff else datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    end_hl = (cutoff - timedelta(days=GAP_HL_D)).strftime('%Y-%m-%d 23:00'); end_8h = (cutoff - timedelta(days=GAP_8H_D)).strftime('%Y-%m-%d 23:00')
    say(f'retrain: cutoff {cutoff:%Y-%m-%d}, h3/h7 end {end_hl}, 8h end {end_8h}, live models {LIVE}')
    signal.signal(signal.SIGTERM, lambda *a: sys.exit(143))          # systemd timeout / stop -> the finally block still resumes paused units
    paused = []
    try:
        paused = pause_heavy()
        resources()
        if not A.skip_data:
            sf = vision_symbols()
            run([sys.executable, 'dl_binance.py', 'klines,fundingRate,premiumIndexKlines'], env={'SYMBOLS_FILE': sf}, timeout=6 * 3600)
            run([sys.executable, 'dl_binance.py', 'gaps'], timeout=2 * 3600)          # reg. 4: truncated monthly archives -> daily files
            pend = (cutoff - timedelta(hours=1)).strftime('%Y-%m-%d %H:00')
            for mode in ('core', 'ohlc', 'premium', 'freeze'):                         # freeze (reg. 4): frozen post-delisting prices -> NaN
                run([sys.executable, 'build_panels.py', pend, mode], timeout=3 * 3600)
            run([sys.executable, 'check_panels.py'])
        shutil.rmtree(NEW, ignore_errors=True); os.makedirs(NEW)
        run([sys.executable, 'exp6b_noage_final.py'], env={'TRAIN_END': end_hl, 'OUT_DIR': NEW})
        run([sys.executable, 'exp17b_final_8h.py'], env={'TRAIN_END': end_8h, 'OUT_DIR': NEW})
        rho, ic = gates(cutoff)
        hashes = {f: sha12(os.path.join(NEW, f)) for f in FILES}
        if A.dry:
            say(f'DRY RUN: gates passed, live models not touched. New hashes {hashes}')
            L.tg_send(f'🧪 Переобучение (проба, {cutoff:%Y-%m-%d}): проверки пройдены, модели НЕ заменены. Spearman с текущими {rho}. '
                      f'Ранговая IC на отложенных днях, новые/прежние: {fmt_ic(ic)}'); return 0
        bak = install_with_rollback(NEW, LIVE, FILES, hashes, f'{now:%Y%m%d-%H%M%S}')
        with open(os.path.join(L.APP, 'models_history.log'), 'a') as h:
            h.write(json.dumps(dict(utc=now.isoformat(), cutoff=str(cutoff.date()), end_hl=end_hl, end_8h=end_8h, spearman=rho, heldout_ic=ic, hashes=hashes, backup=bak)) + '\n')
        L.tg_send(f'🧠 Модели переобучены (данные до {cutoff:%Y-%m-%d}). Проверки пройдены, бот перезапущен и подтвердил новые модели в /status. Spearman с прежними {rho}. '
                  f'Ранговая IC на отложенных днях (3/7 дн: последние 14 дн, 8ч: последние 8 дн), новые/прежние: {fmt_ic(ic)} — для информации, не критерий. '
                  f'Откат вручную: rm -rf {LIVE} && cp -a {bak} {LIVE} && systemctl restart propsleeves')
        return 0
    except SwapFailed as e:
        say(f'FAILED (swap): {e}')
        L.tg_send(str(e)[:3500])
        return 1
    except Exception as e:
        say(f'FAILED: {e}')
        L.tg_send(f'⚠️ Переобучение моделей не выполнено, работают прежние модели. Причина: {str(e)[:700]}')
        return 1
    finally:
        try:
            resume_heavy(paused)
        except Exception as e:
            say(f'resume failed: {e}')
            L.tg_send(f'⚠️ После переобучения не запустился {[u for u, _ in paused]}: {e}. Запусти вручную: systemctl start {" ".join(u for u, _ in paused)}')
        try:
            with open(os.path.join(L.APP if os.path.isdir(L.APP) else REPO, 'retrain_runs.txt'), 'a') as f:
                f.write('\n'.join(LOG) + '\n')
        except OSError:
            pass


if __name__ == '__main__':
    sys.exit(main())
