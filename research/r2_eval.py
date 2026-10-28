#!/usr/bin/env python3
"""R2 gate table: DEV/HOLD Sharpe, stress x1.5, worst 6 months, MDD vs base V0. Run after research/exp_r2.py."""
import glob, math, pandas as pd, numpy as np
DEV = ('2022-01-01', '2026-02-28'); HOLD = ('2026-03-01', '2026-10-26')
def sh(x): x = x.dropna(); return x.mean() / x.std(ddof=1) * math.sqrt(365) if len(x) > 30 else float('nan')
def mdd(x): e = (1 + x.dropna()).cumprod(); return (e / e.cummax() - 1).min()
def w6(x): e = (1 + x.dropna()).cumprod(); return (e / e.shift(182) - 1).min()
rows = {}
for f in sorted(glob.glob('research/phase4_out/R2_V*.csv')):
    t = f.split('R2_')[1][:-4]; df = pd.read_csv(f, index_col=0, parse_dates=True); a, b = df['lag1_x1'], df['lag1_x1.5']
    rows[t] = dict(shDEV=sh(a[DEV[0]:DEV[1]]), shSTR=sh(b[DEV[0]:DEV[1]]), w6DEV=w6(a[DEV[0]:DEV[1]]) * 100, mddDEV=mdd(a[DEV[0]:DEV[1]]) * 100,
                   shHOLD=sh(a[HOLD[0]:HOLD[1]]), mddHOLD=mdd(a[HOLD[0]:HOLD[1]]) * 100)
T = pd.DataFrame(rows).T
b = T.loc['V0']
T['gate'] = [('base' if t == 'V0' else 'PASS' if (r.shDEV - b.shDEV >= -0.10 and r.shSTR - b.shSTR >= -0.10 and r.w6DEV >= b.w6DEV - 1 and r.mddDEV >= b.mddDEV - 1) else 'fail') for t, r in T.iterrows()]
print(T.round(2).to_string())
