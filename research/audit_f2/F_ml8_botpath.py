#!/usr/bin/env python3
"""Audit F2 (agent F): confirm that the LIVE code path (v4/signals.Sleeves.ml8 on a harness-style panel built from Binance data) equals
the 'rank over the age>=180d trade universe' variant of F_ml8_skew.py, and differs from the research path (rank over age>=30d, exp17).
Read-only. Run (repo root): python3 research/audit_f2/F_ml8_botpath.py"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
for line in open(os.path.join(ROOT, 'evns', 'base.env')):
    line = line.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1)
        if not k.startswith(('BYBIT_', 'TG_', 'STATUS_', 'STATE_', 'MODEL_DIR', 'PORT', 'DRY_RUN')):
            os.environ[k] = v
os.environ.update(FIRM='mubite_8_dd10', MODEL_DIR=os.path.join(ROOT, 'v4', 'models'), HIVOL_MULT='0.5')
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'v4')); os.chdir(ROOT)
from bt import Data, quantile_ls
from betautil import rolling_std
import config as C
from signals import Sleeves
from ml_features_8h import compute_features_8h, FEATURES_8H, rank_norm, NEED_H

d = Data(start='2021-01-01', end='2026-10-26 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); VOL = rolling_std(d.ret, 168, 72)
S = Sleeves(log=lambda *a: None); mdl = S.models[8]
syms = [str(s) for s in d.cols]
for ts in ('2026-09-10 08:00', '2026-10-01 16:00', '2026-10-20 00:00'):
    i = int(d.idx.searchsorted(pd.Timestamp(ts, tz='UTC'))); sl = slice(i - 2199, i + 1)
    panel = dict(cff=d.cff[sl], high=d.high[sl].astype(float), low=d.low[sl].astype(float), t24=d.t24[sl].astype(float),
                 fund=d.fund[sl].astype(float), prem=d.prem[sl].astype(float), valid=d.valid[sl], stale=~d.valid[i],
                 data_age=d.age[i].astype(float), symbols=syms)
    age = d.age[i].astype(float)
    S.uni.prev = {}
    mb = S.ml8_universe(panel, age); w_bot = S.ml8(panel, mb, age, 0)
    d.prev_mask = None; u30 = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150); d.prev_mask = None
    ub = d.universe(i, min_age_h=180 * 24, min_turn=1e7, top_n=150)
    F = compute_features_8h(d.cff[i - NEED_H + 1:i + 1], d.fund[i - NEED_H + 1:i + 1], d.t24[i - NEED_H + 1:i + 1], d.prem[i - NEED_H + 1:i + 1],
                            d.high[i - NEED_H + 1:i + 1], d.low[i - NEED_H + 1:i + 1], age, jb, 0)
    def pred(ix):
        p = np.full(N, np.nan); p[ix] = mdl.predict(rank_norm(np.column_stack([F[f][ix] for f in FEATURES_8H]).astype(np.float32)).astype(np.float64)); return p
    p30 = pred(np.flatnonzero(u30)); pb = pred(np.flatnonzero(mb))
    w_res = quantile_ls(np.where(ub, p30, np.nan), 0.2, inv_vol=VOL[i]); w_univ = quantile_ls(pb, 0.2, inv_vol=VOL[i])
    print(f'{ts}: bot universe == research trade universe: {bool((mb == ub).all())} (n {int(mb.sum())}) | '
          f'max|w_bot - w_rank180| {np.abs(w_bot - w_univ).max():.2e} | 0.5*L1(w_bot, w_research) {0.5 * np.abs(w_bot - w_res).sum():.3f}', flush=True)
print('DONE')
