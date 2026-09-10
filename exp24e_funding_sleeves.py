#!/usr/bin/env python3
"""Experiment 24e — the funding sleeves that improved F in exp24a (A1 funding-change contrarian, A2 funding-level carry):
combinations, weights, grids and cost stress, with SEL/CONF windows; plus the pessimistic 'ml8 weight 1' variant."""
import time, json, math, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
from betautil import rolling_std
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=False, load_premium=False); T, N = d.ret.shape
P8 = np.load('out_exp17_pred_8h_p0.npy'); Q3, Q7 = np.load('out_exp21c_pred_h3_noage.npy'), np.load('out_exp21c_pred_h7_noage.npy'); VOL = rolling_std(d.ret, 168, 72)
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
FUND0 = np.where(np.isfinite(d.fund), d.fund, 0.0).astype(np.float32); CF = np.cumsum(FUND0, axis=0, dtype=np.float64)
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
ML3n, ML7n, ML8 = make_ml(Q3, min_age_h=90 * 24), make_ml(Q7, min_age_h=90 * 24), make_ml(P8, 0.2, 180 * 24, True)
def uni(dd, i, age_h=720): return dd.universe(i, min_age_h=age_h, min_turn=1e7, top_n=150)
def q(sig, m, top=0.2): return quantile_ls(np.where(m & np.isfinite(sig), sig, np.nan), top)
def fund_change(win=24, top=0.2, age_h=720):
    def f(dd, i, mask): return q(-(fsum(i, win, 0) - fsum(i, 2 * win, win)), uni(dd, i, age_h), top)
    return f
def fund_level(win=72, top=0.2, age_h=720, iv=False):
    def f(dd, i, mask):
        m = uni(dd, i, age_h); s = -fsum(i, win, 0); return quantile_ls(np.where(m & np.isfinite(s), s, np.nan), top, inv_vol=VOL[i] if iv else None)
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
def Fp(w8=2.0): return [Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8], [1, 1, 1, 1, w8]
BASE8 = dict(reb_h=8, reb_offset=0, band=0.5, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
def sh(x): x = np.asarray(x, float); return float(x.mean() / x.std(ddof=1) * math.sqrt(365)) if len(x) > 30 and x.std(ddof=1) > 0 else float('nan')
rows = []
def go(label, fn, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(BASE8); args.update(kw); args['label'] = label
    m, eq, hp = run(d, fn, ret_diag=True, **args); de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); sel, conf = sh(r[:'2024-12-31']), sh(r['2025-01-01':])
    rows.append(dict(label=label, sharpe=m['sharpe'], sel=sel, conf=conf, cagr=m['cagr'], mdd=m['mdd'], wd=m['worst_day'], turnover=m['turnover_x'], cost=m['fees'] + m['slip']))
    print(f'{label:<64} Sh {m["sharpe"]:5.2f} | SEL {sel:5.2f} CONF {conf:5.2f} | CAGR {m["cagr"]*100:5.1f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:4.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in m['by_year'].items()) + f' ({time.time()-t1:.0f}s)', flush=True)
def F_plus(extra, ws_extra, w8=2.0):
    p, w = Fp(w8); return combo(p + extra, w + ws_extra)
print('\n=== F + funding sleeves ===', flush=True)
go('F', combo(*Fp()))
go('F + A1 (held daily, w1)', F_plus([Held(fund_change())], [1]))
go('F + A2 (8h, w1)', F_plus([fund_level()], [1]))
go('F + A2 (held daily, w1)', F_plus([Held(fund_level())], [1]))
go('F + A1 + A2 (w1, w1)', F_plus([Held(fund_change()), fund_level()], [1, 1]))
go('F + A1 + A2 (w.5, w.5)', F_plus([Held(fund_change()), fund_level()], [0.5, 0.5]))
go('F + A1 + A2 (w1.5, w1)', F_plus([Held(fund_change()), fund_level()], [1.5, 1]))
go('F + A1 + A2 (w1, w1.5)', F_plus([Held(fund_change()), fund_level()], [1, 1.5]))
print('\n=== A1/A2 parameter variants inside F + A1 + A2 (w1, w1) ===', flush=True)
go('A1 win 48h', F_plus([Held(fund_change(48)), fund_level()], [1, 1]))
go('A1 top .3', F_plus([Held(fund_change(top=0.3)), fund_level()], [1, 1]))
go('A2 win 24h', F_plus([Held(fund_change()), fund_level(24)], [1, 1]))
go('A2 win 168h', F_plus([Held(fund_change()), fund_level(168)], [1, 1]))
go('A2 inv-vol weights', F_plus([Held(fund_change()), fund_level(iv=True)], [1, 1]))
go('A1+A2 universe age>=90d', F_plus([Held(fund_change(age_h=90 * 24)), fund_level(age_h=90 * 24)], [1, 1]))
print('\n=== robustness of F + A1 + A2 (w1, w1) ===', flush=True)
FA = F_plus([Held(fund_change()), fund_level()], [1, 1])
go('band .4', FA, band=0.4); go('smooth .6', FA, smooth=0.6); go('VT win 60', FA, vol_win_d=60)
go('cost x1.5', FA, fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45)); go('cost x2', FA, fee_bps=11.0, maker_fee_bps=4.0, slip_fn=slip_model(scale=0.6))
go('maker 50%', FA, maker_share=0.5, slip_fn=slip_model(scale=0.5)); go('taker only', FA, maker_share=0.0, slip_fn=slip_model(scale=1.0))
go('ml8 weight 1 (pessimistic)', F_plus([Held(fund_change()), fund_level()], [1, 1], w8=1.0))
go('ml8 off', combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), Held(fund_change()), fund_level()], [1, 1, 1, 1, 1, 1]))
pd.DataFrame(rows).to_csv('out_exp24e.csv', index=False); print('DONE')
