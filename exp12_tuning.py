#!/usr/bin/env python3
"""Experiment 12: portfolio tuning (OOS 2022-01 -> 2026-08). Same sleeves as exp9c's best (basket-hedged
listing, core336, ML h3/h7 q30). Tests: (a) risk-parity sleeve scalars from standalone dvol,
(b) turnover controls (smooth/band/trade_frac), (c) listing window 7-90d with two-clock age,
(d) ML universe top-100 vs top-150, (e) VT 0.5/0.6 with lev 2.5/3, (f) rebalance offset (12:00 UTC)."""
import os, json, time, numpy as np, pandas as pd
from bt import Data, run, quantile_ls, slip_model
from strategies import core_signal

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2022-01-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True, load_premium=True)
T, N = d.ret.shape
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])
P3 = np.load('out_exp6_pred_h3.npy'); P7 = np.load('out_exp6_pred_h7.npy')
instr = json.load(open('instruments.json'))
launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    m = instr.get(str(s))
    if m and m.get('launch'):
        launch_h[j] = m['launch'] / 1000 / 3600
rows = []


def report(label, m, t0):
    by = m['by_year']
    rows.append(dict(label=label, sharpe=m['sharpe'], cagr=m['cagr'], dvol=m['dvol'], mdd=m['mdd'], calmar=m['calmar'],
                     worst=m['worst_day'], p1=m['p1'], turnover=m['turnover_x'], fees=m['fees'], slip=m['slip'], funding=m['funding'],
                     npos=m['avg_npos'], gross=m['avg_gross'], **{f'sh{y}': v[1] for y, v in by.items()}, **{f'r{y}': v[0] for y, v in by.items()}))
    print(f'{label:<52} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% '
          f'wd {m["worst_day"]*100:5.1f}% p1 {m["p1"]*100:5.2f}% turn {m["turnover_x"]:4.0f}x cost {(m["fees"]+m["slip"])*100:.1f}% fund {-m["funding"]*100:+.1f}% '
          f'npos {m["avg_npos"]:.0f} gross {m["avg_gross"]:.2f} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f'  ({time.time()-t0:.0f}s)', flush=True)


def liquid_basket(dd, i, n=50):
    m = dd.universe(i, min_age_h=720 * 3, min_turn=3e7, top_n=n); m[jm] = False
    return np.flatnonzero(m)


def sl_listing(lo=72, hi=24 * 60, thr=2e7, n_max=30, w_name=0.05, two_clock=False):
    def fn(dd, i, mask):
        age = dd.age[i].astype(float)
        if two_clock:
            ab = np.where(np.isfinite(launch_h), dd.hh[i] - launch_h, np.inf)
            age = np.maximum(age, ab)          # oldest known listing anywhere = true token age proxy
        t = dd.t24[i]
        cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= thr) & (dd.age[i] >= 72)
        cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1
        cand &= ~(r24 > 0.30)
        idx = np.flatnonzero(cand)
        w = np.zeros(N)
        if len(idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:n_max]
        ws = min(1.0 / len(idx), w_name)
        w[idx] = -ws
        b = liquid_basket(dd, i); b = b[~np.isin(b, idx)]
        if len(b):
            w[b] += ws * len(idx) / len(b)
        return w
    return fn


def sl_core(dd, i, mask):
    m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=100)
    return quantile_ls(core_signal(dd, i, m, 336), 0.2)


def make_ml(P, top=0.3, top_n=150):
    def f(dd, i, mask):
        m = dd.universe(i, min_age_h=720, min_turn=1e7, top_n=top_n)
        s = np.where(m & np.isfinite(P[i]), P[i], np.nan)
        return quantile_ls(s, top)
    return f


def combo(parts, scales=None):
    scales = scales or [1.0] * len(parts)
    def fn(dd, i, mask):
        w = np.zeros(N)
        for f, s in zip(parts, scales):
            w += s * f(dd, i, mask)
        return w / sum(scales)
    return fn


def go(label, fn, **kw):
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    args = dict(reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    args.update(kw)
    m, eq = run(d, fn, **args)
    report(label, m, t0)
    return m, eq


A = sl_listing(); A2 = sl_listing(lo=24 * 7, hi=24 * 90, two_clock=True); C = sl_core
D3 = make_ml(P3); D7 = make_ml(P7); D3s = make_ml(P3, top_n=100); D7s = make_ml(P7, top_n=100)
base = combo([A, C, D3, D7])
go('ref A+C+D3+D7 (band30 smooth0.5)', base)
go('A2(two-clock 7-90d)+C+D3+D7', combo([A2, C, D3, D7]))
# risk parity scalars ~ 1/dvol: A ~1.1, C ~1.27, D ~1.28 -> near equal; test tilts
go('tilt ML x1.5', combo([A, C, D3, D7], [1.0, 1.0, 1.5, 1.5]))
go('tilt A x1.5', combo([A, C, D3, D7], [1.5, 1.0, 1.0, 1.0]))
go('tilt C x0.5', combo([A, C, D3, D7], [1.0, 0.5, 1.0, 1.0]))
# turnover controls
go('smooth0.7 band0.4', base, smooth=0.7, band=0.4)
go('smooth0.3 band0.2', base, smooth=0.3, band=0.2)
go('trade_frac 0.5 band0.2 smooth0', base, trade_frac=0.5, band=0.2, smooth=0.0)
# ML universe
go('ML top-100 universe', combo([A, C, D3s, D7s]))
# rebalance at 12:00 UTC instead of 00:00 (ML preds are stamped at 00:00 -> uses 12h-old scores)
go('reb offset 12h (stale ML)', base, reb_offset=12)
# VT variants
go('VT0.6 lev3', base, vol_target=0.006, max_lev=3.0)
go('VT0.5 lev2.5', base, vol_target=0.005, max_lev=2.5)
go('VT0.6 lev3 smooth0.7 band0.4', base, vol_target=0.006, max_lev=3.0, smooth=0.7, band=0.4)
go('A2+C+D3+D7 VT0.6 lev3 gross<=2 stop30', combo([A2, C, D3, D7]), vol_target=0.006, max_lev=3.0, max_gross_x=2.0, stop_pct=0.30)
pd.DataFrame(rows).to_csv('out_exp12_tuning.csv', index=False)
print('DONE')
