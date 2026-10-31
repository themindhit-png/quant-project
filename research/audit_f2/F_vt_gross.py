#!/usr/bin/env python3
"""Audit F2 (agent F): the vol target de-levers realised returns by the exposure multiplier. Research bt.run: lev_eff = gross(target) /
(eq x |w_smoothed|) (w after EMA). Bot main.rebalance: mult_real = gross(target) / (eq x |w_raw|) with info['gross_w'] taken in
Sleeves.combined BEFORE the EMA in RiskEngine.build_targets. Ratio r = |w_smoothed| / |w_raw| -> bot de-levered vol is 1/r higher,
bot lev = r x research lev (when not clipped). This script measures r on the F2+V2 book (research signal functions) at every 8h bar.
Read-only. Run (repo root): python3 research/audit_f2/F_vt_gross.py > logs/agent_F2_F_vt_gross.log"""
import os, sys, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT); os.chdir(ROOT)
import exp_common as X
from exp_common import d, F2, VOL, N

def v2(fn, mult=0.5):                      # research/exp_r2.py v2 (copied to avoid its module-level arrays)
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]))
        if len(nz) < 10: return w
        L, S = w[w > 0].sum(), -w[w < 0].sum(); w[nz[np.argsort(VOL[i][nz])[int(len(nz) * 2 / 3):]]] *= mult
        L2, S2 = w[w > 0].sum(), -w[w < 0].sum()
        if L2 > 0: w[w > 0] *= L / L2
        if S2 > 0: w[w < 0] *= S / S2
        return w
    return g

fn = v2(F2())
i0 = int(d.idx.searchsorted(pd.Timestamp(os.environ.get('FROM', '2023-01-01'), tz='UTC')))
bars = [i for i in range(i0, len(d.idx) - 1) if d.hh[i] % 8 == 0]
prev = None; rows = []
for i in bars:
    mask = d.universe(i, min_age_h=0, min_turn=0, top_n=0)
    w = fn(d, i, mask); w = np.where(np.isfinite(w) & mask, w, 0.0)
    ws = w if prev is None else 0.5 * prev + 0.5 * w
    prev = ws.copy()
    gr, gs = float(np.abs(w).sum()), float(np.abs(ws).sum())
    if gr > 1e-9:
        rows.append((d.idx[i], gr, gs, gs / gr))
df = pd.DataFrame(rows, columns=['t', 'gross_raw', 'gross_smooth', 'r']).set_index('t')
print(df['r'].describe().round(4).to_string())
print('by year mean r:', df['r'].groupby(df.index.year).mean().round(3).to_dict())
print(f'mean gross raw {df.gross_raw.mean():.3f}  smoothed {df.gross_smooth.mean():.3f}')
print('DONE')
