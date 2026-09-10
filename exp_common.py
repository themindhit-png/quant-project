#!/usr/bin/env python3
"""Shared research definitions (Data, sleeves, F/F2 books, BASE8/BASE24, go()) — sliced from exp26_opus3.py."""
import time, json, math, glob, itertools, numpy as np, pandas as pd
from scipy.stats import norm, skew, kurtosis
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r'), np.load('out_exp21c_pred_h7_noage.npy', mmap_mode='r'); Q14 = np.load('out_exp24b_pred_h14_noage.npy', mmap_mode='r')
VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5]); JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
FUND_REAL = d.fund.copy(); FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float64); CF = np.cumsum(FUND0, axis=0, dtype=np.float64)
def fsum(i, a, b): return CF[i - b] - CF[i - a]
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
def uni(dd, i): return dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
def q(sig, m, top=0.2): return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), top)
def fund_change(dd, i, mask): return q(-(fsum(i, 24, 0) - fsum(i, 48, 24)), uni(dd, i))
def fund_level(dd, i, mask): return q(-fsum(i, 72, 0), uni(dd, i))
ML3n, ML7n, ML14n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(Q14, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
class Held:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if dd.hh[i] % 24 == 0: self.w = self.fn(dd, i, mask)
        return self.w
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
def beta0(fn):
    from betautil import rolling_beta
    B = rolling_beta(d.ret, JB, 720)
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(B[i]), B[i], 1.0); w = w.copy(); w[JB] -= float((w * b).sum()); return w
    return g
def F(): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 1, 2])
def F2(carry_daily=False): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), Held(fund_level) if carry_daily else fund_level], [1, 1, 1, 1, 2, 1, 1])
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, band=0.3)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
SER = {}
def go(label, fn, base=BASE8, keep=None, attrib=False, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, attrib=attrib, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    if keep: SER[keep] = r
    print(f'{label:<64} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% fund {m["funding"]*100:+5.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
