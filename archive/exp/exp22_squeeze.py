#!/usr/bin/env python3
"""Experiment 22 — squeeze: candidate improvements of the 5-sleeve book with selection discipline.
Every candidate is reported on the SELECTION window 2022-01..2024-12 and the CONFIRMATION window 2025-01..2026-08
(plus full). Families: turnover (band/smooth), ML8 universe/quantile, sleeve weights (Sharpe-tilted from 2022-24),
extra sleeves (mom720 weekly, core168/504, ML3+7 blend), caps, VT window, listing/core variants, beta cap.
Finally the best-by-selection-window combination is evaluated on the confirmation window."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
VOL = pd.DataFrame(d.ret).rolling(168, min_periods=72).std().values.astype(np.float32)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
JB = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
def sl_listing_f(max_names=30, w_name=0.05, a0=7, a1=90, pump=0.30):
    def f(dd, i, mask):
        age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
        cand = (age >= 24 * a0) & (age <= 24 * a1) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > pump); idx = np.flatnonzero(cand); w = np.zeros(N)
        if len(idx) == 0: return w
        idx = idx[np.argsort(-t[idx])][:max_names]; ws = min(1.0 / len(idx), w_name); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
    return f
sl_listing = sl_listing_f()
def sl_core_f(lb=336, q=0.2):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, lb), q)
    return f
sl_core = sl_core_f()
def make_ml(P, top=0.3, min_age_h=720, iv=False, top_n=150, min_turn=1e7):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=min_age_h, min_turn=min_turn, top_n=top_n)
        return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top, inv_vol=VOL[i] if iv else None)
    return f
def ml_blend(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150)
    def rk(P):
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan); ok = np.isfinite(s); r = np.full(N, np.nan)
        if ok.sum() > 10: rr = np.empty(ok.sum()); rr[np.argsort(s[ok])] = np.arange(ok.sum()); r[ok] = rr / (ok.sum() - 1)
        return r
    return quantile_ls(0.5 * (rk(P3) + rk(P7)), 0.3)
class Weekly:
    def __init__(self, fn): self.fn = fn; self.w = np.zeros(N); self.last = None
    def __call__(self, dd, i, mask):
        wk = (dd.hh[i] // 24) // 7
        if wk != self.last: self.w = self.fn(dd, i, mask); self.last = wk
        return self.w
def sl_mom(dd, i, mask):
    m = dd.universe(i, min_age_h=744, min_turn=1e7, top_n=100)
    with np.errstate(all='ignore'): sig = dd.cff[i - 24] / dd.cff[i - 720] - 1.0
    return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), 0.2)
ML3, ML7 = make_ml(P3), make_ml(P7)
def ML8f(**kw): return make_ml(P8, kw.pop('top', 0.3), kw.pop('min_age_h', 180 * 24), True, **kw)
ML8 = ML8f()
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
def book(listing=sl_listing, core=sl_core, ml3=ML3, ml7=ML7, ml8=ML8, w=(1, 1, 1, 1, 2), extra=()):
    parts = [Held(listing), Held(core), Held(ml3), Held(ml7), ml8] + [Held(e) for e, _ in extra]
    ws = list(w) + [wx for _, wx in extra]
    return combo(parts, ws)
from betautil import rolling_beta
BETA = rolling_beta(d.ret, JB, 720)
def beta_capped(fn, cap):
    def g(dd, i, mask):
        w = fn(dd, i, mask); b = np.where(np.isfinite(BETA[i]), BETA[i], 1.0); bb = float((w * b).sum())
        if abs(bb) > cap: w = w.copy(); w[JB] -= (bb - math.copysign(cap, bb))
        return w
    return g
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
rows = []
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, fam, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna()
    sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(family=fam, label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip']))
    print(f'[{fam:<8}] {label:<52} full {m["sharpe"]:5.2f} | SEL 22-24 {sel:5.2f} | CONF 25-26 {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% ({time.time()-t1:.0f}s)', flush=True)
    return m
print('\n=== baseline ===', flush=True)
go('R5 baseline (band .3, smooth .5, w8 2)', book(), 'base')
print('\n=== C1 turnover: band x smooth ===', flush=True)
for band in (0.4, 0.5):
    for sm in (0.5, 0.6, 0.7): go(f'band {band} smooth {sm}', book(), 'turnover', band=band, smooth=sm)
go('band .3 smooth .6', book(), 'turnover', smooth=0.6)
print('\n=== C2 ML8 universe / quantile ===', flush=True)
go('ML8 top-200', book(ml8=ML8f(top_n=200)), 'ml8uni'); go('ML8 min_turn 5e6', book(ml8=ML8f(min_turn=5e6)), 'ml8uni'); go('ML8 min_turn 5e6 top-200', book(ml8=ML8f(min_turn=5e6, top_n=200)), 'ml8uni')
go('ML8 q .25', book(ml8=ML8f(top=0.25)), 'ml8uni'); go('ML8 q .20', book(ml8=ML8f(top=0.20)), 'ml8uni'); go('ML8 age 120d', book(ml8=ML8f(min_age_h=120 * 24)), 'ml8uni')
print('\n=== C3 sleeve weights ===', flush=True)
for w in ((1, 1, 1, 1, 2.5), (1, 1, 1, 1, 3), (0.5, 1, 1, 1, 2), (1, 0.5, 1, 1, 2), (1, 1, 0.5, 0.5, 2), (1, 1, 1, 1, 1.5), (1.5, 1, 1, 1, 2), (1, 1.5, 1, 1, 2)):
    go(f'weights {w}', book(w=w), 'weights')
print('\n=== C4 extra sleeves ===', flush=True)
go('+ mom720 weekly (w1)', book(extra=[(Weekly(sl_mom), 1)]), 'extra'); go('+ core168 (w1)', book(extra=[(sl_core_f(168), 1)]), 'extra'); go('+ core504 (w1)', book(extra=[(sl_core_f(504), 1)]), 'extra')
go('ML3+ML7 blended into one sleeve (w2)', combo([Held(sl_listing), Held(sl_core), Held(ml_blend), ML8], [1, 1, 2, 2]), 'extra')
go('+ mom720 + core168', book(extra=[(Weekly(sl_mom), 1), (sl_core_f(168), 1)]), 'extra')
print('\n=== C5 caps / VT / beta ===', flush=True)
go('caps 3%/2%', book(), 'caps', pos_cap=0.03, cap_short=0.02); go('caps 1.5%/1%', book(), 'caps', pos_cap=0.015, cap_short=0.01); go('major cap 20%', book(), 'caps', cap_exempt_val=0.20)
go('VT win 60d', book(), 'vt', vol_win_d=60); go('VT win 45d', book(), 'vt', vol_win_d=45); go('VT win 60d + smooth .5', book(), 'vt', vol_win_d=60, vt_smooth=0.5)
go('beta cap ±0.10', beta_capped(book(), 0.10), 'beta'); go('beta cap ±0.20', beta_capped(book(), 0.20), 'beta')
print('\n=== C6 listing / core variants ===', flush=True)
go('listing 40 names w .04', book(listing=sl_listing_f(40, 0.04)), 'listing'); go('listing 5-120d', book(listing=sl_listing_f(a0=5, a1=120)), 'listing'); go('listing pump 20%', book(listing=sl_listing_f(pump=0.20)), 'listing')
go('core q .25', book(core=sl_core_f(q=0.25)), 'core'); go('core q .15', book(core=sl_core_f(q=0.15)), 'core'); go('core 504h', book(core=sl_core_f(504)), 'core')
df = pd.DataFrame(rows); df.to_csv('out_exp22_squeeze.csv', index=False)
base = df[df.family == 'base'].iloc[0]
print(f'\n=== candidates that beat the baseline on the SELECTION window (2022-24: {base.sel:.2f}) and their CONFIRMATION (2025-26: {base.conf:.2f}) ===')
win = df[(df.family != 'base') & (df.sel > base.sel + 0.05)].sort_values('sel', ascending=False)
print(win[['family', 'label', 'sel', 'conf', 'sharpe', 'turnover', 'cost']].round(2).to_string(index=False))
print(f'\nconfirmed (also better on 2025-26 by >0.05): {int((win.conf > base.conf + 0.05).sum())} of {len(win)}; worse on 2025-26: {int((win.conf < base.conf - 0.05).sum())}')
print('DONE')
