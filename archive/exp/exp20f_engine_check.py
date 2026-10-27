#!/usr/bin/env python3
"""Experiment 20f: engine consistency at 8h cadence. T1 daily engine (ref 1.54). T2 8h engine where the strategy at
08/16 returns EXACTLY the current position weights (pos/(gross*lev*eq)) -> should equal T1 if the engine is cadence-
consistent. T3 my drift emulation (V8 w8=0) with diagnostics of drifted weights vs actual position weights."""
import time, json, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal
START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True); T, N = d.ret.shape
P3, P7 = np.load('out_exp6_pred_h3.npy'), np.load('out_exp6_pred_h7.npy')
RET0 = np.where(np.isfinite(d.ret), d.ret, 0.0).astype(np.float32)
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
def combo(parts, weights):
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, weights): w += s * f(dd, i, mask)
        return w / sum(weights)
    return fn
BASE8 = dict(reb_h=8, reb_offset=0, band=0.3, smooth=0.0, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3), gross=1.0, pos_cap=0.02, cap_short=0.015,
             cap_exempt=MAJ5, cap_exempt_val=0.30, uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, vol_target=0.006, max_lev=3.0, max_gross_x=2.0)
BASE24 = dict(BASE8); BASE24.update(reb_h=24)
def go(label, fn, base, **kw):
    t1 = time.time(); d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC'))); args = dict(base); args.update(kw); args['label'] = label
    m, eq = run(d, fn, **args); by = m['by_year']
    print(f'{label:<58} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% MDD {m["mdd"]*100:6.1f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% npos {m["avg_npos"]:.0f} | '
          + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t1:.0f}s)', flush=True)
    return m
daily = combo([sl_listing, sl_core, make_ml(P3), make_ml(P7)], [1, 1, 1, 1])
go('T1 daily engine smooth .5', daily, BASE24, smooth=0.5)
go('T1b daily engine smooth 0', daily, BASE24)
# T2: at 08/16 return exact current position weights (engine-consistent hold); daily EMA .5 applied inside at 00
def hold_exact(diag=False):
    st = {'w': None, 'n': 0, 'err': []}
    def fn(dd, i, mask):
        if dd.hh[i] % 24 == 0:
            w = daily(dd, i, mask); st['w'] = w if st['w'] is None else 0.5 * st['w'] + 0.5 * w; return st['w']
        den = dd.cur_gross * dd.cur_lev * dd.cur_eq
        return dd.cur_pos / den if den > 0 else np.zeros(N)
    return fn
go('T2 8h engine, hold exact positions at 08/16 (band .3)', hold_exact(), BASE8)
go('T2b same, band .05', hold_exact(), BASE8, band=0.05)
# T3: drift emulation with diagnostics vs actual positions
def drift_emul():
    st = {'d': np.zeros(N), 'last': None, 'rows': []}
    def fn(dd, i, mask):
        if dd.hh[i] % 24 == 0:
            w = daily(dd, i, mask); st['d'] = 0.5 * st['d'] + 0.5 * w
        elif st['last'] is not None:
            g = np.prod(1.0 + RET0[st['last'] + 1:i + 1], axis=0); st['d'] = st['d'] * g
            den = dd.cur_gross * dd.cur_lev * dd.cur_eq; wp = dd.cur_pos / den
            nz = (wp != 0) | (st['d'] != 0)
            if nz.any() and len(st['rows']) < 4000:
                st['rows'].append((i, float(np.abs(st['d'] - wp)[nz].mean()), float(np.abs(wp)[nz].mean()), int(nz.sum()), int(((wp == 0) & (st['d'] != 0)).sum()), int(((wp != 0) & (st['d'] == 0)).sum())))
        st['last'] = i
        return st['d']
    fn.st = st
    return fn
f3 = drift_emul(); go('T3 drift emulation (V8 w8=0)', f3, BASE8)
r = pd.DataFrame(f3.st['rows'], columns=['i', 'mean_abs_diff', 'mean_abs_w', 'n', 'emul_only', 'pos_only'])
print('drifted-weights vs actual position weights at 08/16 slots (first 12 rows, then yearly means):'); print(r.head(12).to_string())
r['year'] = d.idx[r.i.values].year; print(r.groupby('year')[['mean_abs_diff', 'mean_abs_w', 'n', 'emul_only', 'pos_only']].mean().round(4).to_string())
print('DONE')
