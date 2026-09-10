#!/usr/bin/env python3
"""Experiment 22e — final squeeze round: age-free daily ML (exp21c panels, universe >= 90d) combined with the robust
tweaks (ML8 q .20, band .4/.5, beta-neutral, with/without listing). SEL 2022-24 / CONF 2025-26 / full, plus cost x2."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_beta, rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape     # memory: no high/low/premium
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy')
P3 = P7 = None                                          # baseline (age) ML books are known from exp22c; not reloaded (memory)
VOL = rolling_std(d.ret, 168, 72)
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
ML3n, ML7n = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24)
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
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0); w = w.copy(); w[JB] -= float((w * b).sum()); return w
    return g
def book(ml3, ml7, q8=0.3, w8=2.0, listing=True, beta=False):
    ml8 = make_ml(P8, q8, 180 * 24, True); parts, ws = [Held(sl_core), Held(ml3)] + ([Held(ml7)] if ml7 is not None else []) + [ml8], [1, 1] + ([1] if ml7 is not None else []) + [w8]
    if listing: parts.insert(0, Held(sl_listing)); ws.insert(0, 1)
    fn = combo(parts, ws); return beta0(fn) if beta else fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
X2 = dict(fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6))
rows = []
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip'], npos=m['avg_npos']))
    print(f'{label:<66} full {m["sharpe"]:5.2f} | SEL {sel:5.2f} CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
print('\n=== age-free daily ML (uni>=90d) + robust tweaks ===', flush=True)
print('N0 R5 baseline (age ML): full 1.84 | SEL 1.66 CONF 2.20 (exp22c)')
go('N1 R5 with noage ML', book(ML3n, ML7n))
go('N2 N1 + ml8 q.2 + band .4', book(ML3n, ML7n, q8=0.2), band=0.4)
go('N3 N2 + beta0', book(ML3n, ML7n, q8=0.2, beta=True), band=0.4)
go('N4 N2 band .5', book(ML3n, ML7n, q8=0.2), band=0.5)
go('N5 N3 band .5', book(ML3n, ML7n, q8=0.2, beta=True), band=0.5)
go('N6 N2 NO listing', book(ML3n, ML7n, q8=0.2, listing=False), band=0.4)
go('N7 N3 NO listing (K13-noage)', book(ML3n, ML7n, q8=0.2, listing=False, beta=True), band=0.4)
go('N8 N5 NO listing (K14-noage)', book(ML3n, ML7n, q8=0.2, listing=False, beta=True), band=0.5)
go('N9 core+ml3n+ml8 q.2 w2 + beta0, band .4 (R3b-noage)', book(ML3n, None, q8=0.2, listing=False, beta=True), band=0.4)
go('N10 listing+core+ml3n+ml8 q.2 w2 + beta0, band .4', book(ML3n, None, q8=0.2, listing=True, beta=True), band=0.4)
go('N11 N3 w8 2.5', book(ML3n, ML7n, q8=0.2, w8=2.5, beta=True), band=0.4)
print('\n=== cost x2 for the finalists ===', flush=True)
go('N2 cost x2', book(ML3n, ML7n, q8=0.2), band=0.4, **X2)
go('N3 cost x2', book(ML3n, ML7n, q8=0.2, beta=True), band=0.4, **X2)
go('N5 cost x2', book(ML3n, ML7n, q8=0.2, beta=True), band=0.5, **X2)
go('N7 cost x2', book(ML3n, ML7n, q8=0.2, listing=False, beta=True), band=0.4, **X2)
go('N9 cost x2', book(ML3n, None, q8=0.2, listing=False, beta=True), band=0.4, **X2)
go('N10 cost x2', book(ML3n, None, q8=0.2, listing=True, beta=True), band=0.4, **X2)
pd.DataFrame(rows).to_csv('out_exp22e.csv', index=False); print('DONE')
