#!/usr/bin/env python3
"""Audit E: share of fcarry weight placed in names with NO funding print in the last 72h (f72 == 0 exactly: data gap or true zero)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from E_indep import Panel, sig_fcarry, OOS
p = Panel(); i0 = int(p.time.searchsorted(OOS)); rows = []
nf = np.vstack([np.zeros((1, p.N)), np.cumsum(p.fund != 0, 0)])
for i in range(i0, p.T - 1):
    if p.hour[i] % 8: continue
    w = sig_fcarry(p, i); none = (nf[i + 1] - nf[i + 1 - 72]) == 0
    rows.append((p.time[i].year, w[(w > 0) & none].sum(), -w[(w < 0) & none].sum(), ','.join(p.sym[(w != 0) & none])))
df = pd.DataFrame(rows, columns=['y', 'long', 'short', 'names'])
print('fcarry: avg share of LONG side in names without any funding print in 72h, by year (%):', (df.groupby('y').long.mean() * 100).round(1).to_dict())
print('                      SHORT side:', (df.groupby('y').short.mean() * 100).round(1).to_dict())
print('names:', pd.Series(','.join(df.names).split(',')).replace('', np.nan).dropna().value_counts().head(10).to_dict())
