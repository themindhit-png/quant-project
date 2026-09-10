#!/usr/bin/env python3
"""Experiment 11: transferability of the listing-drift sleeve to Bybit.
Use Bybit launchTime (instruments.json) as the age clock instead of Binance first bar, on Binance prices,
for symbols that exist on both venues. Also event study by Bybit-age. If the effect survives when the
clock is Bybit's, the live bot can use Bybit launchTime directly."""
import json, time, numpy as np, pandas as pd
from bt import Data, run, slip_model

START, END, OOS = '2021-01-01', '2026-08-31 23:00', '2021-06-01'
d = Data(start=START, end=END, min_turn_ever=5e6, load_hl=True)
T, N = d.ret.shape
instr = json.load(open('instruments.json'))
launch_h = np.full(N, np.nan)
for j, s in enumerate(d.cols):
    m = instr.get(str(s))
    if m and m.get('launch'):
        launch_h[j] = m['launch'] / 1000 / 3600
hh = d.hh
on_bybit = np.isfinite(launch_h)
print(f'symbols on both venues: {on_bybit.sum()} of {N}')
MAJ5 = [s for s in ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT'] if s in d.cols]
jm = np.array([int(np.flatnonzero(d.cols == s)[0]) for s in MAJ5])


def bybit_age(i):
    return np.where(on_bybit, hh[i] - launch_h, -1)     # hours since Bybit launch (can be negative)


# event study by Bybit age (from Bybit-age day 3), BTC-relative, requiring Binance price to exist
jb = int(np.flatnonzero(d.cols == 'BTCUSDT')[0])
curves = []
for j in np.flatnonzero(on_bybit):
    i3 = int(np.searchsorted(hh, launch_h[j] + 72))
    if i3 <= 0 or i3 + 24 * 60 >= T or not d.valid[i3, j] or d.t24[i3, j] < 2e7:
        continue
    if d.age[i3, j] < 0 or d.idx[i3] < pd.Timestamp('2021-01-01', tz='UTC'):
        continue
    p = d.cff[i3:i3 + 24 * 60 + 1:24, j]; b = d.cff[i3:i3 + 24 * 60 + 1:24, jb]
    if not (np.all(np.isfinite(p)) and p[0] > 0):
        continue
    curves.append(np.log(p / p[0]) - np.log(b / b[0]))
C = np.array(curves)
print(f'Bybit-age event study: n {len(C)} | ' + '  '.join(f'd{k}: mean {C[:, k].mean()*100:+.1f}% med {np.median(C[:, k])*100:+.1f}%' for k in (7, 14, 30, 45, 60)))


def sleeve(clock='bybit', lo=72, hi=24 * 60, thr=2e7, n_max=30, w_name=0.05):
    def fn(dd, i, mask):
        age = bybit_age(i) if clock == 'bybit' else dd.age[i]
        t = dd.t24[i]
        cand = (age >= lo) & (age <= hi) & dd.valid[i] & np.isfinite(t) & (t >= thr) & (dd.age[i] >= 24)
        if clock == 'bybit':
            cand &= on_bybit
        cand[jm] = False
        r24 = dd.cff[i] / dd.cff[i - 24] - 1
        cand &= ~(r24 > 0.30)
        idx = np.flatnonzero(cand)
        w = np.zeros(N)
        if len(idx) == 0:
            return w
        idx = idx[np.argsort(-t[idx])][:n_max]
        ws = min(1.0 / len(idx), w_name)
        w[idx] = -ws; w[jm] += ws * len(idx) / len(jm)
        return w
    return fn


for label, fn in [('clock=Binance age (reference)', sleeve('binance')), ('clock=Bybit launchTime', sleeve('bybit')),
                  ('clock=Bybit, age 7-90d', sleeve('bybit', lo=24 * 7, hi=24 * 90))]:
    t0 = time.time()
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(OOS, tz='UTC')))
    m, eq = run(d, fn, reb_h=24, band=0.3, smooth=0.5, fee_bps=5.5, maker_share=0.7, slip_fn=slip_model(scale=0.3),
                gross=1.0, pos_cap=0.02, cap_short=0.015, cap_exempt=MAJ5, cap_exempt_val=0.30, stop_pct=0.30,
                uni_kwargs=dict(min_age_h=0, min_turn=0, top_n=0), verbose=False, label=label)
    by = m['by_year']
    print(f'{label:<34} Sh {m["sharpe"]:5.2f} CAGR {m["cagr"]*100:6.1f}% dvol {m["dvol"]*100:.2f}% MDD {m["mdd"]*100:6.1f}% wd {m["worst_day"]*100:5.1f}% '
          f'turn {m["turnover_x"]:3.0f}x npos {m["avg_npos"]:.0f} | ' + ' '.join(f'{y}:{v[1]:.1f}' for y, v in by.items()) + f' ({time.time()-t0:.0f}s)', flush=True)
print('DONE')
