#!/usr/bin/env python3
"""Audit E: (1) how often quantile selection of the funding sleeves (fcarry = -72h funding sum, fchg = -(24h - previous 24h))
falls inside a block of exactly tied scores (selection then decided by sort order = arbitrary);
(2) funding-data coverage of liquid names (24h windows without any funding print while the name trades >= $10M/day)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from E_indep import Panel, OOS
p = Panel(); i0 = int(p.time.searchsorted(OOS)); out = {'fcarry': [], 'fchg': []}
for i in range(i0, p.T - 1, 8):
    if p.hour[i] % 8: continue
    m = p.eligible(i, 720, 1e7, 150)
    f24 = p.cF[i + 1] - p.cF[i + 1 - 24]; f48 = p.cF[i + 1 - 24] - p.cF[i + 1 - 48]; f72 = p.cF[i + 1] - p.cF[i + 1 - 72]
    for name, sc in (('fcarry', -f72), ('fchg', -(f24 - f48))):
        s = np.round(sc[m], 12); n = len(s)
        if n < 10: continue
        k = max(2, int(round(n * 0.2))); o = np.sort(s); res = []
        for side, thr, sel in (('short', o[k - 1], s <= o[k - 1]), ('long', o[-k], s >= o[-k])):
            tied_block = int((s == thr).sum()); chosen_in_block = k - int((s < thr).sum()) if side == 'short' else k - int((s > thr).sum())
            res.append((tied_block > 1, chosen_in_block / k if tied_block > 1 else 0.0, tied_block))
        out[name].append((p.time[i].year, *res[0], *res[1]))
for name, rows in out.items():
    df = pd.DataFrame(rows, columns=['y', 'S_tie', 'S_share', 'S_block', 'L_tie', 'L_share', 'L_block'])
    print(f'{name}: rebalances {len(df)}; boundary inside a tie block: short {df.S_tie.mean()*100:.0f}% long {df.L_tie.mean()*100:.0f}%; '
          f'avg share of a side picked arbitrarily from ties: short {df.S_share.mean()*100:.1f}% long {df.L_share.mean()*100:.1f}%; median block when tied: '
          f'{df.S_block[df.S_tie].median() if df.S_tie.any() else 0:.0f}/{df.L_block[df.L_tie].median() if df.L_tie.any() else 0:.0f}')
    print('   by year short-share%:', (df.groupby('y').S_share.mean() * 100).round(1).to_dict(), 'long-share%:', (df.groupby('y').L_share.mean() * 100).round(1).to_dict())
# funding coverage
liq = p.ok & np.isfinite(p.t24) & (p.t24 >= 1e7) & (p.age >= 720)
nf24 = p.cF.shape and (np.vstack([np.zeros((1, p.N)), np.cumsum(p.fund != 0, 0)]))
cnt = nf24[24:] - nf24[:-24]                                       # prints in the 24 bars ending at each bar
L = liq[23:][i0:]; C = cnt[i0:]
print(f'funding coverage: liquid symbol-hours {L.sum()}, with no funding print in the last 24h: {(L & (C == 0)).sum()} ({(L & (C == 0)).sum()/L.sum()*100:.3f}%)')
bad = pd.Series((L & (C == 0)).sum(0), index=p.sym); print('  worst:', bad[bad > 0].sort_values()[::-1][:8].to_dict())
