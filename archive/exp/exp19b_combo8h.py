#!/usr/bin/env python3
"""Experiment 19b: fair comparison on the 8h engine — daily sleeves held between daily updates; per-8h EMA smoothing
0.794 (=0.5^(1/3), daily-equivalent to the daily engine's 0.5); band 0.3. Combos with the grid-aligned 8h ML sleeve."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7, P8 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy'), np.load('out_exp17_pred_8h_p0.npy')
instr = json.load(open('instruments.json')); launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    mm = instr.get(str(s))
    if mm and mm.get('launch'): launch_h[j] = mm['launch'] / 1000 / 3600
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]; jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
def sl_listing(dd, i, mask):
    age = np.maximum(dd.age[i].astype(float), np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, -np.inf)); t = dd.t24[i]
    cand = (age >= 24 * 7) & (age <= 24 * 90) & dd.valid[i] & np.isfinite(t) & (t >= 2e7) & (dd.age[i] >= 72); cand[jm] = False
    r24 = dd.cff[i] / dd.cff[i - 24] - 1; cand &= ~(r24 > 0.30); idx = np.flatnonzero(cand); w = np.zeros(N)
    if len(idx) == 0: return w
    idx = idx[np.argsort(-t[idx])][:30]; ws = min(1.0 / len(idx), 0.05); w[idx] = -ws; w[jm] += ws * len(idx) / len(jm); return w
def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100); return quantile_ls(core_signal(dd, i, m, 336), 0.2)
def make_ml(P, top=0.3):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=150); return quantile_ls(np.where(m & np.isfinite(P[i]), P[i], np.nan), top)
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
def daily_parts(): return [Held(sl_listing), Held(sl_core), Held(make_ml(P3)), Held(make_ml(P7))]
ML8 = make_ml(P8, 0.3); S8 = 0.5 ** (1 / 3); series = {}
def go(label, fn, keep=None, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=8, reb_offset=0, band=0.3, smooth=S8, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02,
                cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label,
                vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
    args.update(kw)
    if keep:
        m, eq, hp = run(d, fn, ret_diag=True, **args); series[keep] = (hp / eq.shift(1)).dropna(); eq.to_csv(f'out_{keep}_eq.csv')
    else: m, eq = run(d, fn, **args)
    by = m['by_year']
    print(f'{label:<50} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
go('daily held, smooth0.794 (ref)', combo(daily_parts(), [1, 1, 1, 1]), keep='d8')
go('8h ML alone smooth0.794', ML8, keep='m8')
go('daily + 8h w1', combo(daily_parts() + [ML8], [1, 1, 1, 1, 1]))
go('daily + 8h w2', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2]), keep='final8_w2')
go('daily + 8h w3', combo(daily_parts() + [ML8], [1, 1, 1, 1, 3]))
go('daily + 8h w2 band0.4', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2]), band=0.4)
go('daily + 8h w2 maker50', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2]), maker_share=0.5, slip_fn=slip_model(scale=0.5))
go('daily + 8h w2 taker', combo(daily_parts() + [ML8], [1, 1, 1, 1, 2]), maker_share=0.0, slip_fn=slip_model(scale=1.0))
a = (1 + series['d8']).groupby(series['d8'].index.floor('D')).prod() - 1; b = (1 + series['m8']).groupby(series['m8'].index.floor('D')).prod() - 1
print(f'correlation(daily book, 8h ML) daily returns: {a.corr(b):.2f}')
print('DONE')
