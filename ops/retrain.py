#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Automatic model retrain on the VPS (Phase 5.2; timer propsleeves-retrain: 5 January and 5 July, 03:30 UTC). Runs in the research venv
from the repo clone (/root/quant-project). Steps:
  1. resource check (RAM / disk);  2. Binance Vision: symbol list + incremental download (klines, funding, premium);
  3. panels up to the cutoff (1st of the month 00:00 UTC) + check;  4. final models into models_new/: h3/h7 (TRAIN_END = cutoff - 14 d,
     PHASE4_LOG «4.2 ревью»: gap h + 7 days, the h7 gap is used for both) and 8h (TRAIN_END = cutoff - 8 d);
  5. gates (no tuning — the models have no honest OOS): v4/tests_v4.py with the new models (feature parity T25/T25c, sleeves T4/T26);
     predictions not degenerate; Spearman new vs current models over the last 30 days >= 0.5;
  6. swap: backup of the live models, copy, bot restart outside a rebalance window, Telegram report (hashes, Spearman, train ends).
Any failure -> the live models stay, the reason goes to Telegram. Manual: python3 ops/retrain.py [--cutoff 2027-01-01] [--dry] [--skip-data]"""
import os, sys, json, time, shutil, hashlib, argparse, subprocess, glob
from datetime import datetime, timezone, timedelta
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'ops'))
import opslib as L

GAP_HL_D, GAP_8H_D = 14, 8
LIVE = os.environ.get('LIVE_MODELS', os.path.join(L.APP, 'models'))
NEW = os.path.join(REPO, 'models_new')
FILES = ('ml_h3_noage.txt', 'ml_h7_noage.txt', 'spec_noage.json', 'ml_8h.txt', 'spec_8h.json')
LOG = []


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
    return rep


def safe_restart():
    """Restart the bots outside the rebalance windows (01/09/17 UTC hh:05-hh:25)."""
    while True:
        now = datetime.now(timezone.utc)
        if not (now.hour % 8 == 1 and 5 <= now.minute < 25):
            break
        time.sleep(60)
    for u in L.bot_units():
        if L.is_active(u):
            subprocess.run(['systemctl', 'restart', u]); say(f'restarted {u}')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--cutoff', default=None); ap.add_argument('--dry', action='store_true')
    ap.add_argument('--skip-data', action='store_true', help='use the existing data/panels (testing)')
    A = ap.parse_args()
    now = datetime.now(timezone.utc)
    cutoff = datetime.fromisoformat(A.cutoff).replace(tzinfo=timezone.utc) if A.cutoff else datetime(now.year, now.month, 1, tzinfo=timezone.utc)
    end_hl = (cutoff - timedelta(days=GAP_HL_D)).strftime('%Y-%m-%d 23:00'); end_8h = (cutoff - timedelta(days=GAP_8H_D)).strftime('%Y-%m-%d 23:00')
    say(f'retrain: cutoff {cutoff:%Y-%m-%d}, h3/h7 end {end_hl}, 8h end {end_8h}, live models {LIVE}')
    try:
        resources()
        if not A.skip_data:
            sf = vision_symbols()
            run([sys.executable, 'dl_binance.py', 'klines,fundingRate,premiumIndexKlines'], env={'SYMBOLS_FILE': sf}, timeout=6 * 3600)
            pend = (cutoff - timedelta(hours=1)).strftime('%Y-%m-%d %H:00')
            for mode in ('core', 'ohlc', 'premium'):
                run([sys.executable, 'build_panels.py', pend, mode], timeout=3 * 3600)
            run([sys.executable, 'check_panels.py'])
        shutil.rmtree(NEW, ignore_errors=True); os.makedirs(NEW)
        run([sys.executable, 'exp6b_noage_final.py'], env={'TRAIN_END': end_hl, 'OUT_DIR': NEW})
        run([sys.executable, 'exp17b_final_8h.py'], env={'TRAIN_END': end_8h, 'OUT_DIR': NEW})
        rho = gates(cutoff)
        hashes = {f: sha12(os.path.join(NEW, f)) for f in FILES}
        if A.dry:
            say(f'DRY RUN: gates passed, live models not touched. New hashes {hashes}')
            L.tg_send(f'🧪 Переобучение (проба, {cutoff:%Y-%m-%d}): проверки пройдены, модели НЕ заменены. Spearman с текущими {rho}'); return 0
        bak = f'{LIVE}.bak.{now:%Y%m%d-%H%M%S}'; shutil.copytree(LIVE, bak)
        for f in FILES:
            shutil.copy2(os.path.join(NEW, f), os.path.join(LIVE, f))
        say(f'models replaced (backup {bak}); new hashes {hashes}')
        safe_restart()
        with open(os.path.join(L.APP, 'models_history.log'), 'a') as h:
            h.write(json.dumps(dict(utc=now.isoformat(), cutoff=str(cutoff.date()), end_hl=end_hl, end_8h=end_8h, spearman=rho, hashes=hashes, backup=bak)) + '\n')
        L.tg_send(f'🧠 Модели переобучены (данные до {cutoff:%Y-%m-%d}). Проверки пройдены, бот перезапущен. Spearman с прежними {rho}. '
                  f'Откат: cp {bak}/* {LIVE}/ && systemctl restart propsleeves')
        return 0
    except Exception as e:
        say(f'FAILED: {e}')
        L.tg_send(f'⚠️ Переобучение моделей не выполнено, работают прежние модели. Причина: {str(e)[:700]}')
        return 1
    finally:
        try:
            with open(os.path.join(L.APP if os.path.isdir(L.APP) else REPO, 'retrain_runs.txt'), 'a') as f:
                f.write('\n'.join(LOG) + '\n')
        except OSError:
            pass


if __name__ == '__main__':
    sys.exit(main())
