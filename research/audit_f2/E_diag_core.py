#!/usr/bin/env python3
"""Audit E: where does the raw core sleeve's 2025 return come from? Paper attribution (no costs), positions set on
00:00 bars and drifting for 24h, per symbol per year. Also scans held names for extreme bars."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from E_indep import Panel, sig_core, OOS
p = Panel(); i0 = int(p.time.searchsorted(OOS)); rows = []; big = []
for i in range(i0, p.T - 25):
    if p.hour[i] != 0: continue
    w = sig_core(p, i)
    g = np.cumprod(1 + p.R0[i + 1:i + 25], 0)                     # drift of each name over the next 24 bars
    contrib = w * (g[-1] - 1)
    nz = np.flatnonzero(w)
    for j in nz: rows.append((p.time[i].year, p.sym[j], contrib[j], w[j]))
    hr = p.R0[i + 1:i + 25, nz]; k = np.unravel_index(np.argmax(np.abs(hr)), hr.shape)
    if abs(hr[k]) > 0.5: big.append((str(p.time[i + 1 + k[0]]), p.sym[nz[k[1]]], round(float(hr[k]), 3), round(float(w[nz[k[1]]]), 4)))
df = pd.DataFrame(rows, columns=['year', 'sym', 'c', 'w'])
for y, g in df.groupby('year'):
    s = g.groupby('sym').c.sum().sort_values()
    print(f'{y}: paper sum {g.c.sum():+.3f} | long {g[g.w>0].c.sum():+.3f} short {g[g.w<0].c.sum():+.3f} | top+ ' +
          ', '.join(f'{a} {b:+.3f}' for a, b in s[::-1][:6].items()) + ' | top- ' + ', '.join(f'{a} {b:+.3f}' for a, b in s[:4].items()))
print('held bars with |1h ret| > 50%:', len(big))
for b in big[:30]: print('  ', b)
