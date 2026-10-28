#!/usr/bin/env python3
"""Phase T: own (personal) Bybit account — growth vs drawdown for the vol target, with the bot's own floor (25 % from the equity peak,
risk 'trailing_pct', floor guard -> latched stop), no firm limits, no payouts (everything reinvested). Fast simulator on the V2 book.
Note: above exposure 4 the per-unit return is extrapolated from the level-4 base run (position caps bind harder in reality -> optimistic).
Run: python3 research/personal.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fastsim as fs

ts, R, G, V, hour, day, RH = fs.load_books()
F = fs.fvec(T1=1e9, T2=1e9, TOTAL_DD=25, DAILY_DD=100, TRAIL=2, NOTIONAL_X=100)
rows = []
for vt in (0.006, 0.008, 0.010, 0.012, 0.015):
    for cppi in (0.0, 0.5):
        p = fs.pvec(VT=vt, BASE=1.0, CPPI=cppi, DD_BUF=0.3, THROTTLE=100, HALT_FRAC=10, FG=0.05, BOOK=0, VT_MAX=4.0, MAXG=3.5)
        res = {}
        for lab, a, b in (('2022-26', '2022-01-03', None), ('2025-26', '2025-01-01', None), ('2026', '2026-01-01', None)):
            i0 = int(ts.searchsorted(pd.Timestamp(a, tz=ts.tz))); w = np.zeros(R.shape[2])
            o, ie, mn, ee, *_ = fs.sim(R, G, V, RH, hour, day, i0, p, F, 1, 1, 0, 10000.0, np.zeros(1), w)
            eq = pd.Series(w[i0 + 1:ie + 1], index=ts[i0 + 1:ie + 1]).resample('D').last().dropna()
            yrs = len(eq) / 365.0
            res[lab] = dict(cagr=(eq.iloc[-1] / 10000) ** (1 / yrs) - 1, mdd=(eq / eq.cummax() - 1).min(), stop=o in (2, 3))
        rows.append(dict(VT=vt, CPPI=cppi, **{f'{k}_{m}': (round(v[m] * 100, 1) if m != 'stop' else v[m]) for k, v in res.items() for m in v}))
df = pd.DataFrame(rows); df.to_csv(os.path.join(fs.ROOT, 'research', 'phase4_out', 'T_personal.csv'), index=False)
pd.set_option('display.width', 220); print(df.to_string(index=False))
