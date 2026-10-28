#!/usr/bin/env python3
"""F2 with the most volatile third of held names removed (V1) or at half weight (V2), research engine, lag 1h, costs x1.
Idea from research/diag_legs.py (2026-10-28): the 2026 drawdown comes from the top-vol tercile. Found on HOLD -> decide on DEV.
Run: END="2026-10-26 23:00" PRED_SUF8=_oct_noconst PRED_SUF37=_noage_oct python3 research/exp_hivol.py"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_common as X
from exp_common import d, F2, BASE8, VOL
from bt import run

def damp(fn, mult):
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]))
        if len(nz) < 10: return w
        hv = nz[np.argsort(VOL[i][nz])[int(len(nz) * 2 / 3):]]
        L, S = w[w > 0].sum(), -w[w < 0].sum(); w[hv] *= mult
        L2, S2 = w[w > 0].sum(), -w[w < 0].sum()
        if L2 > 0: w[w > 0] *= L / L2
        if S2 > 0: w[w < 0] *= S / S2
        return w
    return g

LAG = 1; out = {}
for tag, mult in (('V1_drop', 0.0), ('V2_half', 0.5)):
    f0 = damp(F2(), mult)
    def fn(dd, i, mask, f0=f0): return f0(dd, i - LAG, mask)
    args = dict(BASE8); args['reb_offset'] = LAG; args['label'] = tag
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
    m, eq, hp = run(d, fn, ret_diag=True, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None)
    out[tag] = r; print(f'{tag}: Sh {m["sharpe"]:.2f} CAGR {m["cagr"]*100:.1f}% MDD {m["mdd"]*100:.1f}% cost {(m["fees"]+m["slip"])*100:.1f}%', flush=True)
pd.DataFrame(out).to_csv('research/phase4_out/F2_hivol.csv')
