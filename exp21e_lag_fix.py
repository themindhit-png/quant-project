#!/usr/bin/env python3
"""Experiment 21e — corrected execution-delay test. exp21a's 'lag 1h' silently DROPPED the ML sleeves (prediction panels
exist only on 00/08/16 bars). Here: decide on the prediction bar, execute `lag` hours later (engine reb_offset=lag,
signals computed at i-lag). Plus: book-beta neutralisation on selection/confirmation windows; drop top-20 non-major names."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_beta
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0]); BETA = rolling_beta(d.ret, JB, 720)
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
ML3, ML7, ML8 = make_ml(P3), make_ml(P7), make_ml(P8, 0.3, 180 * 24, True)
class HeldL:
    """daily sleeve decided on the 00 bar, executed lag hours later; held in between."""
    def __init__(self, fn, lag=0): self.fn = fn; self.lag = lag; self.w = np.zeros(N)
    def __call__(self, dd, i, mask):
        if (dd.hh[i] - self.lag) % 24 == 0: self.w = self.fn(dd, i - self.lag, mask)
        return self.w
def lagged(fn, lag): return lambda dd, i, mask: fn(dd, i - lag, mask)
def combo(parts, weights, black=None):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        w = w / sum(weights)
        if black is not None: w[black] = 0.0
        return w
    return fn
def book5(lag=0, black=None, listing=True):
    parts = ([HeldL(sl_listing, lag)] if listing else []) + [HeldL(sl_core, lag), HeldL(ML3, lag), HeldL(ML7, lag), lagged(ML8, lag)]
    return combo(parts, ([1] if listing else []) + [1, 1, 1, 2], black)
def book4(lag=0): return combo([lagged(sl_listing, lag), lagged(sl_core, lag), lagged(ML3, lag), lagged(ML7, lag)], [1, 1, 1, 1])
def beta_capped(fn, cap):
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0); bb = float((w * b).sum())
        if abs(bb) > cap: w = w.copy(); w[JB] -= (bb - math.copysign(cap, bb))
        return w
    return g
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== execution delay: decide on the prediction bar, execute `lag` hours later ===', flush=True)
go('R5 lag 0 (baseline)', book5(0))
for lag in (1, 2, 4): go(f'R5 lag {lag}h', book5(lag), reb_offset=lag)
go('R5 lag 8h (one full period)', book5(8), reb_offset=0)
go('DAILY lag 0 (baseline)', book4(0), BASE24)
for lag in (1, 3, 6): go(f'DAILY lag {lag}h', book4(lag), BASE24, reb_offset=lag)
go('DAILY lag 24h', book4(24), BASE24, reb_offset=0)
print('\n=== book-beta neutralisation (30d hourly betas) on selection/confirmation windows ===', flush=True)
go('R5 beta free', book5()); go('R5 beta 0', beta_capped(book5(), 0.0)); go('R5 beta ±0.05', beta_capped(book5(), 0.05))
go('R4 (no listing) beta free', book5(listing=False)); go('R4 beta 0', beta_capped(book5(listing=False), 0.0))
go('DAILY beta 0', beta_capped(book4(), 0.0), BASE24)
print('\n=== concentration: drop the 20 best NON-major names (hedge legs kept) ===', flush=True)
m = run(d, book5(), ret_diag=False, attrib=True, **{**BASE8, 'label': 'attrib'})[0]
a = m['attrib']; pnl = (a.price + a.funding - a.cost).drop(labels=[s for s in MAJ5 if s in a.index])
top20 = list(pnl.sort_values(ascending=False).index[:20]); print('  top-20 non-major names:', top20)
go('R5 without top-20 non-major names', book5(black=np.array([list(d.cols).index(s) for s in top20])))
top50 = list(pnl.sort_values(ascending=False).index[:50])
go('R5 without top-50 non-major names', book5(black=np.array([list(d.cols).index(s) for s in top50])))
print('DONE')
