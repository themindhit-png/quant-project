#!/usr/bin/env python3
"""Where does F2 make / lose money? Paper decomposition of the F2 book (no costs, no vol target, no smoothing):
each 8h rebalance records every sleeve's weights; a leg earns weight x (price return over the next 8 bars - funding).
Legs are split by sleeve, side and coin type at decision time (vol rank, pump score, age, turnover rank).
Output: daily component returns -> research/phase4_out/legs_daily.parquet + a summary table by period.
Run: END="2026-10-26 23:00" PRED_SUF8=_oct_noconst PRED_SUF37=_noage_oct python3 research/diag_legs.py"""
import os, sys, time, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_common as X
from exp_common import d, N, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level, VOL

SLEEVES = dict(listing=(sl_listing, 1, True), core=(sl_core, 1, True), ml3=(ML3n, 1, True), ml7=(ML7n, 1, True),
               ml8=(ML8, 2, False), fchg=(fund_change, 1, True), fcarry=(fund_level, 1, False))
TOT = sum(v[1] for v in SLEEVES.values())
R = np.where(np.isfinite(d.ret), d.ret, 0.0); FU = np.where(np.isfinite(d.fund), d.fund, 0.0)
CR = np.vstack([np.zeros((1, N)), np.cumsum(R, 0)]); CFu = np.vstack([np.zeros((1, N)), np.cumsum(FU, 0)])
# pump score: largest 1h return over the previous 7 days
MAXR = pd.DataFrame(R).rolling(168, min_periods=24).max().values

def rank01(x, m):
    out = np.full(N, np.nan); idx = np.flatnonzero(m & np.isfinite(x))
    if len(idx) > 1: out[idx] = np.argsort(np.argsort(x[idx])) / (len(idx) - 1)
    return out

t0 = time.time(); i0 = max(int(d.idx.searchsorted(pd.Timestamp('2022-01-01', tz='UTC'))), 24 * 35); T = len(d.idx)
held = {k: np.zeros(N) for k in SLEEVES}; rows = []
for i in range(i0, T - 9):
    if d.hh[i] % 8: continue
    mask = d.valid[i]; daily = d.hh[i] % 24 == 0
    W = {}
    for k, (fn, s, is_daily) in SLEEVES.items():
        if is_daily:
            if daily: held[k] = np.nan_to_num(fn(d, i, mask))
            W[k] = held[k] * s / TOT
        else:
            W[k] = np.nan_to_num(fn(d, i, mask)) * s / TOT
    fwd = (CR[i + 9] - CR[i + 1]) - (CFu[i + 9] - CFu[i + 1])     # bars i+1..i+8 earn on positions set at i; shorts receive +funding
    anyw = np.zeros(N, bool)
    for w in W.values(): anyw |= w != 0
    vr = rank01(VOL[i], anyw); pr = rank01(MAXR[i], anyw); tr = rank01(d.t24[i].astype(float), anyw)
    young = d.age[i] < 180 * 24
    day = d.idx[i].floor('D').tz_localize(None)
    for k, w in W.items():
        for side, sm in (('L', w > 0), ('S', w < 0)):
            if not sm.any(): continue
            c = w * fwd
            rec = dict(day=day, sleeve=k, side=side, pnl=float(c[sm].sum()),
                       pnl_hivol=float(c[sm & (vr >= 2 / 3)].sum()), pnl_pump=float(c[sm & (pr >= 2 / 3)].sum()),
                       pnl_young=float(c[sm & young].sum()), pnl_lowturn=float(c[sm & (tr < 1 / 3)].sum()),
                       gross=float(np.abs(w[sm]).sum()), gross_hivol=float(np.abs(w[sm & (vr >= 2 / 3)]).sum()))
            rows.append(rec)
df = pd.DataFrame(rows); daily_df = df.groupby(['day', 'sleeve', 'side']).sum(numeric_only=True)
os.makedirs('research/phase4_out', exist_ok=True); daily_df.to_parquet('research/phase4_out/legs_daily.parquet')
print(f'legs: {len(df)} rows, {time.time() - t0:.0f}s')

P = {'DEV 2022-01..2026-02': ('2022-01-01', '2026-02-28'), 'HOLD 2026-03..10': ('2026-03-01', '2026-10-26'), 'live 11.09-26.10': ('2026-09-11', '2026-10-26')}
def ann(x, a, b): x = x[a:b]; return x.sum() / max(len(x), 1) * 365 * 100
def shp(x, a, b): x = x[a:b]; return x.mean() / x.std(ddof=1) * np.sqrt(365) if len(x) > 30 and x.std(ddof=1) > 0 else np.nan
dd = daily_df.reset_index()
book = dd.groupby('day').pnl.sum()
print('\nBOOK (paper, no costs): ' + ' | '.join(f'{p}: {ann(book, a, b):+.1f}%/yr Sh {shp(book, a, b):.2f}' for p, (a, b) in P.items()))
for col in ['pnl', 'pnl_hivol', 'pnl_pump', 'pnl_young', 'pnl_lowturn']:
    print(f'\n== {col} (annualised % of equity, paper) ==')
    t = dd.pivot_table(index='day', columns=['sleeve', 'side'], values=col, aggfunc='sum').fillna(0.0)
    out = pd.DataFrame({p: t[a:b].sum() / max(len(t[a:b]), 1) * 365 * 100 for p, (a, b) in P.items()})
    side = dd.pivot_table(index='day', columns='side', values=col, aggfunc='sum').fillna(0.0)
    alls = pd.DataFrame({p: side[a:b].sum() / max(len(side[a:b]), 1) * 365 * 100 for p, (a, b) in P.items()})
    alls.index = pd.MultiIndex.from_tuples([('ALL', x) for x in alls.index]); out = pd.concat([out, alls])
    print(out.round(1).to_string())
