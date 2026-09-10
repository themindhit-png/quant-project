#!/usr/bin/env python3
"""Experiment 25b — the unlock calendar as a CONSTRAINT/TILT on the book instead of a sleeve: (a) block longs in names
with a cliff unlock >= THR % of supply inside [-PRE, +POST] days (redistribute the long side), (b) additionally double the
short weight of such names (renormalised). Applied to F and to F+A1+A2 (the funding-sleeve book). SEL/CONF windows."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
cols = [str(c) for c in d.cols]; cidx = {c: j for j, c in enumerate(cols)}
U = pd.read_csv('data/unlocks/unlocks_events.csv', low_memory=False)
U = U[U.ticker.notna() & (U.unlock_type == 'cliff') & ~U.category.isin(['noncirculating', 'burned'])].copy()
U['sym'] = U.ticker.str.upper() + 'USDT'; U = U[U.sym.isin(cidx)]; U['date'] = pd.to_datetime(U.date, utc=True)
E = U.groupby(['date', 'sym'], as_index=False).pct_of_circ.sum()
# per-bar flag panel: 1 if a cliff unlock >= THR lies within [-POST, +PRE] days of the bar's day (built once per THR)
day_of_bar = d.idx.floor('D')
def flag_panel(thr=1.0, pre=7, post=2):
    Fp = np.zeros((T, N), dtype=bool); days = pd.DatetimeIndex(sorted(set(day_of_bar)))
    pos = {dt: k for k, dt in enumerate(days)}; M = np.zeros((len(days), N), dtype=bool)
    for r in E[E.pct_of_circ >= thr].itertuples():
        k = pos.get(r.date.floor('D'))
        if k is None: continue
        M[max(0, k - pre):min(len(days), k + post + 1), cidx[r.sym]] = True      # bars from pre days BEFORE the event to post days AFTER
    kb = np.searchsorted(days.values, day_of_bar.values); Fp[:] = M[kb]; return Fp
FLAG1 = flag_panel(1.0, 7, 2); FLAG05 = flag_panel(0.5, 7, 2)
print(f'flag coverage (share of (bar,name) cells flagged, thr 1%): {FLAG1.mean()*100:.3f}% ; avg flagged names per bar {FLAG1.sum(1).mean():.1f}', flush=True)
P8 = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r'), np.load('out_exp21c_pred_h7_noage.npy', mmap_mode='r'); VOL = rolling_std(d.ret, 168, 72)
FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float32); CF = np.cumsum(FUND0.astype(np.float64), axis=0, dtype=np.float64)
def fsum(i, a, b): return CF[i - b] - CF[i - a]
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(cols):
    mm = instr.get(s)
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in cidx]; jm = np.array([cidx[s] for s in MAJ5])
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
ML3n, ML7n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
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
def unlock_tilt(fn, FLAG, mode='block_longs', boost=2.0):
    def g(dd, i, mask):
        w = fn(dd, i, mask); f = FLAG[i]
        if not f.any(): return w
        w = w.copy(); L = w > 0; S = w < 0; l0, s0 = w[L].sum(), -w[S].sum()
        w[L & f] = 0.0                                   # never long into a large unlock
        if mode == 'boost_shorts': w[S & f] *= boost
        L2 = w > 0; S2 = w < 0
        if w[L2].sum() > 0: w[L2] *= l0 / w[L2].sum()   # keep the long side's gross
        if -w[S2].sum() > 0: w[S2] *= s0 / (-w[S2].sum())
        return w
    return g
def F(): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 1, 2])
def FA(): return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level], [1, 1, 1, 1, 2, 1, 1])
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
print('\n=== unlock calendar as a constraint ===', flush=True)
go('F', F()); go('F block longs into unlocks >=1%', unlock_tilt(F(), FLAG1)); go('F block longs >=0.5%', unlock_tilt(F(), FLAG05))
go('F block longs + boost shorts x2 (>=1%)', unlock_tilt(F(), FLAG1, 'boost_shorts')); go('F block longs + boost shorts x3 (>=1%)', unlock_tilt(F(), FLAG1, 'boost_shorts', 3.0))
go('F+A1+A2', FA()); go('F+A1+A2 block longs into unlocks >=1%', unlock_tilt(FA(), FLAG1)); go('F+A1+A2 block longs + boost shorts x2', unlock_tilt(FA(), FLAG1, 'boost_shorts'))
print('DONE')
