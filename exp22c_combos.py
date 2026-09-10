#!/usr/bin/env python3
"""Experiment 22c — combinations of the individually robust tweaks from exp22/22b (ML8 q .20, band .4/.5, VT 60d,
listing pump 20% / 40 names, no listing sleeve, beta neutralisation, w8 2.5), on selection (2022-24) and confirmation
(2025-26) windows. Pre-registered choice rule: maximise SEL Sharpe subject to CONF >= baseline CONF − 0.05, preferring
lower turnover and structural simplicity at equal Sharpe (differences < 0.15 are within noise)."""
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
def sl_listing_f(max_names=30, w_name=0.05, pump=0.30):
    def f(dd, i, mask):
        age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
        cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > pump); idx = np.flatnonzero(cand); w = np.zeros(N)
        if len(idx) == 0: return w
        idx = idx[np.argsort(-t[idx])][:max_names]; ws = min(1.0 / len(idx), w_name); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
    return f
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3, min_age_h=720, iv=False):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=1e7, top_n=150)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
ML3, ML7 = make_ml(P3), make_ml(P7)
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
def book(q8=0.3, w8=2.0, listing='std', beta=False):
    ml8 = make_ml(P8, q8, 180 * 24, True)
    parts, ws = [Held(sl_core), Held(ML3), Held(ML7), ml8], [1, 1, 1, w8]
    if listing == 'std': parts.insert(0, Held(sl_listing_f())); ws.insert(0, 1)
    elif listing == 'tuned': parts.insert(0, Held(sl_listing_f(40, 0.04, 0.20))); ws.insert(0, 1)
    fn = combo(parts, ws)
    return beta0(fn) if beta else fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
rows = []
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, base=BASE8, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip'], npos=m['avg_npos']))
    print(f'{label:<58} full {m["sharpe"]:5.2f} | SEL {sel:5.2f} CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
print('\n=== combinations (8h engine) ===', flush=True)
go('K0 baseline R5', book())
go('K1 ML8 q.20', book(q8=0.2))
go('K2 band .4', book(), band=0.4)
go('K3 q.20 + band .4', book(q8=0.2), band=0.4)
go('K4 q.20 + band .4 + VT60', book(q8=0.2), band=0.4, vol_win_d=60)
go('K5 q.20 + band .5', book(q8=0.2), band=0.5)
go('K6 q.20 + band .4 + listing tuned (40n, .04, pump20)', book(q8=0.2, listing='tuned'), band=0.4)
go('K7 q.20 + band .4, NO listing (R4)', book(q8=0.2, listing=None), band=0.4)
go('K8 q.20 + band .4 + beta 0', book(q8=0.2, beta=True), band=0.4)
go('K9 q.20 + band .4 + w8 2.5', book(q8=0.2, w8=2.5), band=0.4)
go('K10 q.20 + band .4 + w8 2.5, NO listing', book(q8=0.2, w8=2.5, listing=None), band=0.4)
go('K11 q.20 + band .4 + VT60 + beta 0, NO listing', book(q8=0.2, listing=None, beta=True), band=0.4, vol_win_d=60)
go('K12 NO listing + beta 0 (q.3, band .3)', book(listing=None, beta=True))
go('K13 q.20 + band .4 + beta 0, NO listing', book(q8=0.2, listing=None, beta=True), band=0.4)
go('K14 q.20 + band .5 + beta 0, NO listing', book(q8=0.2, listing=None, beta=True), band=0.5)
print('\n=== daily book combinations (daily engine) ===', flush=True)
def dbook(listing=True, beta=False):
    parts, ws = [sl_core, ML3, ML7], [1, 1, 1]
    if listing: parts.insert(0, sl_listing_f()); ws.insert(0, 1)
    fn = combo(parts, ws); return beta0(fn) if beta else fn
go('D0 daily baseline', dbook(), BASE24); go('D1 daily NO listing', dbook(False), BASE24); go('D2 daily NO listing + beta 0', dbook(False, True), BASE24)
go('D3 daily NO listing + beta 0 + band .4', dbook(False, True), BASE24, band=0.4); go('D4 daily + beta 0', dbook(True, True), BASE24)
df = pd.DataFrame(rows); df.to_csv('out_exp22c.csv', index=False); b = df.iloc[0]
ok = df[(df.conf >= b.conf - 0.05)].sort_values('sel', ascending=False)
print(f'\nrule: max SEL s.t. CONF >= baseline CONF − 0.05 ({b.conf - 0.05:.2f}):'); print(ok[['label', 'sharpe', 'sel', 'conf', 'mdd', 'turnover', 'cost', 'npos']].head(8).round(2).to_string(index=False))
print('DONE')
