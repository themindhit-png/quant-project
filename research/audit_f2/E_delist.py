#!/usr/bin/env python3
"""Audit E, task 4: delisted symbols and frozen prices.
(a) panel scan: trailing frozen runs (close unchanged and qvol==0) before the last bar of delisted names, and frozen runs
    inside listed periods; (b) exposure the engines keep in names without a price (zombie) or with zero 24h volume."""
import os, sys, pickle, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); PAN = os.path.join(ROOT, 'data', 'panels')
END = pd.Timestamp('2026-08-31 23:00', tz='UTC')
c = pd.read_parquet(os.path.join(PAN, 'close.parquet')).loc[pd.Timestamp('2021-11-01', tz='UTC'):END]
qv = pd.read_parquet(os.path.join(PAN, 'qvol.parquet')).loc[pd.Timestamp('2021-11-01', tz='UTC'):END][c.columns]
C, Q = c.values, qv.values; T, N = C.shape
same = np.zeros_like(C, bool); same[1:] = (C[1:] == C[:-1]) & np.isfinite(C[1:])
frozen = same & (np.nan_to_num(Q) == 0)
last = np.array([np.flatnonzero(np.isfinite(C[:, j]))[-1] if np.isfinite(C[:, j]).any() else -1 for j in range(N)])
delisted = (last >= 0) & (last < T - 25)
trail = []
for j in np.flatnonzero(delisted):
    k = last[j]; n = 0
    while k - n > 0 and frozen[k - n, j]: n += 1
    trail.append(n)
trail = np.array(trail)
print(f'symbols {N}, delisted before {END.date()}: {delisted.sum()}; trailing frozen run (h) before last bar: '
      f'>=1h {np.sum(trail>=1)}, >=6h {np.sum(trail>=6)}, >=24h {np.sum(trail>=24)}, max {trail.max() if len(trail) else 0}')
top = np.argsort(-trail)[:8]; dj = np.flatnonzero(delisted)
print('  longest:', ', '.join(f'{c.columns[dj[t]]} {trail[t]}h' for t in top if trail[t] > 0))
# frozen runs inside listed history (>= 6h)
runs = []
for j in range(N):
    f = frozen[:, j]
    if not f.any(): continue
    d = np.diff(np.r_[0, f.astype(int), 0]); st = np.flatnonzero(d == 1); en = np.flatnonzero(d == -1)
    for s, e in zip(st, en):
        if e - s >= 6: runs.append((c.columns[j], str(c.index[s]), e - s))
print(f'frozen runs >=6h anywhere: {len(runs)}; >=24h: {sum(r[2]>=24 for r in runs)}')
for r in sorted(runs, key=lambda x: -x[2])[:10]: print('  ', r)
# (b) engine exposure in names without price / without volume (from E_indep runs, positions identical to bt for core/fcarry)
for n in ['core_full', 'fcarry_full', 'combo_full', 'combo_full_remask']:
    pth = os.path.join(ROOT, 'logs', f'E_indep_{n}.pkl')
    if not os.path.exists(pth): continue
    hf = pickle.load(open(pth, 'rb'))['held_frozen']
    zh = sum(1 for x in hf if x[1] > 0); zg = sum(x[1] for x in hf); fh = sum(1 for x in hf if x[2] > 0); fg = sum(x[2] for x in hf)
    print(f'{n}: hours with zombie exposure (no price) {zh}, sum of gross/eq over those hours {zg:.2f} (mean {zg/max(zh,1)*100:.2f}% eq); '
          f'hours with zero-volume names {fh}, sum gross/eq {fg:.2f}; names: {sorted(set(s for x in hf for s in x[3]))[:15]}')
