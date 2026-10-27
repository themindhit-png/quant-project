#!/usr/bin/env python3
"""Experiment 25 — token-unlock drift (new alpha family from the literature scan; data: DefiLlama unlock calendar,
87–128 perp tickers, 2021–2026; pct_of_circ approximate = current supply). (1) Event study: BTC-relative returns around
cliff unlocks by size bucket and year. (2) Sleeve: short names with a cliff unlock >= THR % of supply inside [-PRE, +POST]
days, hedged with equal long majors (like the listing sleeve); standalone and added to F. Caveats: calendar reconstructed
today (possible revisions/survivorship), supply denominator is current."""
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
E = U.groupby(['date', 'sym'], as_index=False).pct_of_circ.sum(); E = E[(E.date >= '2021-06-01') & (E.date <= '2026-08-20')]
print(f'cliff events with a perp symbol: {len(E)} rows, {E.sym.nunique()} symbols, pct>=0.5%: {(E.pct_of_circ>=0.5).sum()}, >=1%: {(E.pct_of_circ>=1).sum()}, >=2%: {(E.pct_of_circ>=2).sum()}', flush=True)
# ---- daily closes at 00:00 bars (bar open 00:00 -> close 01:00) for the event study
day_bars = np.flatnonzero(d.hh % 24 == 0); day_ts = d.idx[day_bars].floor('D'); JB = cidx['BTCUSDT']
LC = np.log(d.cff[day_bars].astype(np.float64)); LB = LC[:, JB]
def rel_ret(row, a, b):
    """BTC-relative log return from day a to day b relative to the event date (a<b), NaN if not tradable."""
    k = day_ts.searchsorted(row.date.floor('D')); ia, ib = k + a, k + b
    if ia < 0 or ib >= len(day_bars): return np.nan
    j = cidx[row.sym]
    if not (d.valid[day_bars[ia], j] and d.valid[day_bars[ib], j]): return np.nan
    t = d.t24[day_bars[ia], j]
    if not np.isfinite(t) or t < 5e6: return np.nan
    return (LC[ib, j] - LC[ia, j]) - (LB[ib] - LB[ia])
print('\n=== event study: BTC-relative log return (%), by size bucket ===', flush=True)
for lo, hi in ((0.5, 1.0), (1.0, 2.0), (2.0, 5.0), (5.0, 1e9), (1.0, 1e9)):
    sub = E[(E.pct_of_circ >= lo) & (E.pct_of_circ < hi)]
    if len(sub) < 10: continue
    res = {w: np.array([rel_ret(r, a, b) for r in sub.itertuples()]) for w, (a, b) in {'-14..-7': (-14, -7), '-7..0': (-7, 0), '0..+3': (0, 3), '0..+7': (0, 7), '+7..+14': (7, 14), '-7..+7': (-7, 7)}.items()}
    line = f'  pct [{lo},{hi}) n={len(sub)}: ' + ' | '.join(f'{w}: mean {np.nanmean(v)*100:+.2f} med {np.nanmedian(v)*100:+.2f} hit<0 {np.mean(v[np.isfinite(v)]<0)*100:.0f}% (n {np.isfinite(v).sum()})' for w, v in res.items())
    print(line, flush=True)
sub = E[E.pct_of_circ >= 1.0]
print('  by year (pct>=1, window -7..+7): ' + ' '.join(f'{y}: {np.nanmean([rel_ret(r, -7, 7) for r in sub[sub.date.dt.year == y].itertuples()])*100:+.2f}% (n {int((sub.date.dt.year == y).sum())})' for y in range(2021, 2027)), flush=True)
# ---- sleeve
CAL = {}
for r in E.itertuples():
    CAL.setdefault(r.sym, []).append((r.date.floor('D'), r.pct_of_circ))
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(cols):
    mm = instr.get(s)
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in cidx]; jm = np.array([cidx[s] for s in MAJ5])
def unlock_sleeve(thr=1.0, pre=7, post=2, cap=0.05, size_w=True):
    def f(dd, i, mask):
        now = dd.idx[i].floor('D'); w = np.zeros(N); sc = {}
        for s, evs in CAL.items():
            j = cidx[s]
            if not dd.valid[i, j] or not np.isfinite(dd.t24[i, j]) or dd.t24[i, j] < 1e7 or dd.age[i, j] < 720 or j in jm: continue
            tot = sum(p for (dt, p) in evs if p >= thr and -post <= (dt - now).days <= pre)
            if tot > 0: sc[j] = tot
        if not sc: return w
        vals = np.array([min(sc[j], 10.0) if size_w else 1.0 for j in sc]); vals = vals / vals.sum(); vals = np.minimum(vals, cap)
        for (j, v) in zip(sc, vals): w[j] = -v
        w[jm] += vals.sum() / len(jm); return w
    return f
P8 = np.load('out_exp17_pred_8h_p0.npy', mmap_mode='r'); Q3, Q7, Q14 = np.load('out_exp21c_pred_h3_noage.npy', mmap_mode='r'), np.load('out_exp21c_pred_h7_noage.npy', mmap_mode='r'), np.load('out_exp24b_pred_h14_noage.npy', mmap_mode='r'); VOL = rolling_std(d.ret, 168, 72)
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
ML3n, ML7n, ML14n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(Q14, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24, band=0.3)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} | SEL {sh(r[:"2024-12-31"]):5.2f} CONF {sh(r["2025-01-01":]):5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
print('\n=== unlock sleeve standalone (daily) ===', flush=True)
for thr, pre, post in ((1.0, 7, 2), (0.5, 7, 2), (1.0, 14, 3), (2.0, 7, 2), (1.0, 3, 1)):
    go(f'unlock short thr {thr}% window [-{pre},+{post}]', unlock_sleeve(thr, pre, post), BASE24)
go('unlock short thr 1% [-7,+2], equal weights', unlock_sleeve(1.0, 7, 2, size_w=False), BASE24)
print('\n=== added to F / F14 ===', flush=True)
Fp = [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8]; F14p = [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML14n), ML8]
go('F', combo(Fp, [1, 1, 1, 1, 2])); go('F + unlock (w1)', combo(Fp + [Held(unlock_sleeve())], [1, 1, 1, 1, 2, 1])); go('F + unlock (w.5)', combo(Fp + [Held(unlock_sleeve())], [1, 1, 1, 1, 2, 0.5]))
go('F14 + unlock (w1)', combo(F14p + [Held(unlock_sleeve())], [1, 1, 1, 1, 2, 1]))
print('DONE')
