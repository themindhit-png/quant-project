#!/usr/bin/env python3
"""Reg. 4 check (after audit F2 / agent F M1): the LIVE ml8 path (v4/signals.Sleeves.ml8 with the training-universe rank mask, as
Sleeves.combined calls it) vs the research path (exp17: features ranked over age >= 30 d top-150, traded on age >= 180 d), with the
deployed v4/models/ml_8h.txt on Binance panels. Prints prediction Spearman and 0.5*L1 of the weights, new path and old path.
Run (repo root): python3 research/audit_f2/R4_ml8_botpath.py [N_BARS]  -> logs/R4_ml8_botpath.log"""
import os, sys, numpy as np, pandas as pd
from scipy.stats import spearmanr
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
NB = int(sys.argv[1]) if len(sys.argv) > 1 else 40
d = Data(start='2025-06-01', end='2026-10-26 23:00', min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape; jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); VOL = rolling_std(d.ret, 168, 72)
S = Sleeves(log=lambda *a: None); mdl = S.models[8]; syms = [str(s) for s in d.cols]
bars = [i for i in range(T) if d.hh[i] % 8 == 0 and d.idx[i] >= pd.Timestamp('2025-09-01', tz='UTC')]
bars = bars[::max(1, len(bars) // NB)][:NB]
rows = []
for i in bars:
    sl = slice(i - 2199, i + 1); age = d.age[i].astype(float)
    panel = dict(cff=d.cff[sl], high=d.high[sl].astype(float), low=d.low[sl].astype(float), t24=d.t24[sl].astype(float), fund=d.fund[sl].astype(float),
                 prem=d.prem[sl].astype(float), valid=d.valid[sl], stale=~d.valid[i], data_age=age, symbols=syms)
    S.uni.prev = {}
    mb = S.ml8_universe(panel, age); mr = S.ml8_rank_universe(panel, age)
    w_new = S.ml8(panel, mb, age, 0, rank_mask=mr); w_old = S.ml8(panel, mb, age, 0)
    d.prev_mask = None; u30 = d.universe(i, min_age_h=720, min_turn=1e7, top_n=150); d.prev_mask = None
    ub = d.universe(i, min_age_h=180 * 24, min_turn=1e7, top_n=150)
    F = compute_features_8h(d.cff[i - NEED_H + 1:i + 1], d.fund[i - NEED_H + 1:i + 1], d.t24[i - NEED_H + 1:i + 1], d.prem[i - NEED_H + 1:i + 1],
                            d.high[i - NEED_H + 1:i + 1], d.low[i - NEED_H + 1:i + 1], age, jb, 0)
    def pred(ix):
        p = np.full(N, np.nan); p[ix] = mdl.predict(rank_norm(np.column_stack([F[f][ix] for f in FEATURES_8H]).astype(np.float32)).astype(np.float64)); return p
    p30, pb, pr = pred(np.flatnonzero(u30)), pred(np.flatnonzero(mb)), pred(np.flatnonzero(mr))
    w_res = quantile_ls(np.where(ub, p30, np.nan), 0.2, inv_vol=VOL[i])
    both = ub & np.isfinite(p30) & np.isfinite(pr) & np.isfinite(pb)
    rows.append(dict(t=str(d.idx[i]), uni_rank_eq=bool((mr == u30).all()), uni_trade_eq=bool((mb == ub).all()),
                     sp_new=spearmanr(pr[both], p30[both])[0], sp_old=spearmanr(pb[both], p30[both])[0],
                     l1_new=0.5 * np.abs(w_new - w_res).sum(), l1_old=0.5 * np.abs(w_old - w_res).sum()))
df = pd.DataFrame(rows); print(df.round(4).to_string())
print(f'\n{len(df)} bars 2025-09..2026-10 | rank universe == research: {df.uni_rank_eq.mean():.2f} | trade universe == research: {df.uni_trade_eq.mean():.2f}')
print(f'Spearman pred vs research: NEW {df.sp_new.mean():.4f} (min {df.sp_new.min():.4f}) | OLD {df.sp_old.mean():.4f} (min {df.sp_old.min():.4f})')
print(f'0.5*L1 weights vs research: NEW {df.l1_new.mean():.4f} (max {df.l1_new.max():.4f}) | OLD {df.l1_old.mean():.3f} (max {df.l1_old.max():.3f})')
