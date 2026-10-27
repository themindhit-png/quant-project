#!/usr/bin/env python3
"""Experiment 22d — cost robustness of the finalists (fees+slip ×1.5 and ×2, maker 50%): the biggest live risk is
execution cost, so the production choice must weigh Sharpe at modelled costs against degradation under higher costs."""
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
def book(sleeves, q8=0.3, w8=2.0, beta=False):
    ml8 = make_ml(P8, q8, 180 * 24, True); parts, ws = [], []
    for s in sleeves:
        if s == 'listing': parts.append(Held(sl_listing)); ws.append(1)
        if s == 'core': parts.append(Held(sl_core)); ws.append(1)
        if s == 'ml3': parts.append(Held(ML3)); ws.append(1)
        if s == 'ml7': parts.append(Held(ML7)); ws.append(1)
    parts.append(ml8); ws.append(w8); fn = combo(parts, ws)
    return beta0(fn) if beta else fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
rows = []
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip']))
    print(f'{label:<64} full {m["sharpe"]:5.2f} | SEL {sel:5.2f} CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% ({time.time()-t1:.0f}s)', flush=True)
    return m['sharpe']
CANDS = {
    'R5 base (listing,core,ml3,ml7 + ml8 q.3) band .3': (book(['listing', 'core', 'ml3', 'ml7']), dict()),
    'K3  R5 + ml8 q.2, band .4': (book(['listing', 'core', 'ml3', 'ml7'], q8=0.2), dict(band=0.4)),
    'K7  R4 no listing, ml8 q.2, band .4': (book(['core', 'ml3', 'ml7'], q8=0.2), dict(band=0.4)),
    'K13 R4 no listing + beta0, ml8 q.2, band .4': (book(['core', 'ml3', 'ml7'], q8=0.2, beta=True), dict(band=0.4)),
    'R3  core+ml3 + ml8 q.3 w2, band .3': (book(['core', 'ml3']), dict()),
    'R3\' core+ml3 + ml8 q.2 w2, band .4': (book(['core', 'ml3'], q8=0.2), dict(band=0.4)),
    'R3b core+ml3 + ml8 q.2 w2 + beta0, band .4': (book(['core', 'ml3'], q8=0.2, beta=True), dict(band=0.4)),
    'R4w3 no listing, ml8 q.3 w3, band .3': (book(['core', 'ml3', 'ml7'], w8=3.0), dict()),
    'R4w3\' no listing, ml8 q.2 w3, band .4': (book(['core', 'ml3', 'ml7'], q8=0.2, w8=3.0), dict(band=0.4)),
}
COSTS = {'x1.0': dict(), 'x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45)), 'x2.0': dict(fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6)),
         'maker50': dict(maker_share=0.5, slip_fn=slip_model(scale=0.5))}
res = {}
for name, (fn, kw) in CANDS.items():
    print(f'\n=== {name} ===', flush=True)
    for cn, ckw in COSTS.items():
        a = dict(kw); a.update(ckw); res[(name, cn)] = go(f'{name} | cost {cn}', fn, **a)
print('\n=== summary: Sharpe by cost scenario ===')
print(f'{"candidate":<64} {"x1.0":>6} {"x1.5":>6} {"x2.0":>6} {"mk50":>6}')
for name in CANDS: print(f'{name:<64} ' + ' '.join(f'{res[(name, c)]:6.2f}' for c in COSTS))
pd.DataFrame(rows).to_csv('out_exp22d.csv', index=False); print('DONE')
