#!/usr/bin/env python3
"""Phase R2 (docs/PHASE_R_PLAN.md): variants V2-V6 of F2, research engine, lag 1h, costs x1 and x1.5. Variant list fixed before the runs (2026-10-28).
V0 base | V2 half weight of top-vol tercile | V3 every sleeve weights ~ 1/vol | V4 short cap by vol | V5 no shorts in top-decile max30 | V6 +skew30 sleeve (1 of 9) | V26 = V2 + V6 (the one combination attempt, run after the gate)
V2/V4/V5 act on the combined book, V3 on each sleeve (side sums kept). V4 cap = 2 x median |short weight| x (median vol / vol).
Run: END="2026-10-26 23:00" PRED_SUF8=_oct_noconst PRED_SUF37=_noage_oct python3 research/exp_r2.py V2 [V3 ...]"""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_common as X
from exp_common import d, N, F2, BASE8, VOL, combo, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level, uni, q
from bt import run, slip_model

def renorm(w, L, S):
    L2, S2 = w[w > 0].sum(), -w[w < 0].sum()
    if L2 > 0: w[w > 0] *= L / L2
    if S2 > 0: w[w < 0] *= S / S2
    return w
def sides(w): return w[w > 0].sum(), -w[w < 0].sum()

def v2(fn, mult=0.5):
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]))
        if len(nz) < 10: return w
        L, S = sides(w); w[nz[np.argsort(VOL[i][nz])[int(len(nz) * 2 / 3):]]] *= mult
        return renorm(w, L, S)
    return g
def invvol(w, i):
    nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]) & (VOL[i] > 0))
    if len(nz) < 10: return w
    w = w.copy(); L, S = sides(w); w[nz] *= np.median(VOL[i][nz]) / VOL[i][nz]
    return renorm(w, L, S)
def v4(fn, k=2.0):
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); sh = np.flatnonzero((w < 0) & np.isfinite(VOL[i]) & (VOL[i] > 0))
        if len(sh) < 5: return w
        L, S = sides(w); cap = k * np.median(-w[sh]) * np.median(VOL[i][sh]) / VOL[i][sh]
        w[sh] = -np.minimum(-w[sh], cap)
        return renorm(w, L, S)
    return g
# max30 = largest daily (24h-summed) return over 30 days, as research/screen_ideas.py
R = np.where(np.isfinite(d.ret), d.ret, 0.0)
def max30(i): return R[i + 1 - 720:i + 1].reshape(30, 24, N).sum(1).max(0)
def v5(fn, dec=0.9):
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); m = uni(dd, i); idx = np.flatnonzero(m)
        if len(idx) < 20: return w
        x = max30(i); thr = np.nanquantile(x[idx], dec)
        kill = (w < 0) & (x >= thr); L, S = sides(w); w[kill] = 0
        return renorm(w, L, S)
    return g
CR2 = np.vstack([np.zeros((1, N)), np.cumsum(R * R, 0)]); CR3 = np.vstack([np.zeros((1, N)), np.cumsum(R ** 3, 0)])
def skew30_sig(i): return -((CR3[i + 1] - CR3[i + 1 - 720]) / 720) / np.maximum((CR2[i + 1] - CR2[i + 1 - 720]) / 720, 1e-12) ** 1.5
def sl_skew(dd, i, mask): return q(skew30_sig(i), uni(dd, i), 0.2) if False else __import__('bt').quantile_ls(np.where(uni(dd, i) & np.isfinite(skew30_sig(i)), skew30_sig(i), np.nan), 0.2, inv_vol=VOL[i])

def book(tag):
    if tag == 'V3':
        t = lambda f: (lambda dd, i, mask: invvol(f(dd, i, mask), i))
        return combo([Held(t(sl_listing)), Held(t(sl_core)), Held(t(lambda a, b, c: ML3n(a, b, c))), Held(t(lambda a, b, c: ML7n(a, b, c))), t(ML8),
                      Held(t(fund_change)), t(fund_level)], [1, 1, 1, 1, 2, 1, 1])
    if tag == 'V6':
        return combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level, Held(sl_skew)], [1, 1, 1, 1, 2, 1, 1, 1])
    if tag == 'V26':
        return v2(combo([Held(sl_listing), Held(sl_core), Held(ML3n), Held(ML7n), ML8, Held(fund_change), fund_level, Held(sl_skew)], [1, 1, 1, 1, 2, 1, 1, 1]))
    base = F2()
    return {'V0': base, 'V2': v2(base), 'V4': v4(base), 'V5': v5(base)}[tag]

COST = {'lag1_x1': {}, 'lag1_x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45))}
for tag in sys.argv[1:]:
    res = {}
    for sc, kw in COST.items():
        f0 = book(tag)
        def fn(dd, i, mask, f0=f0): return f0(dd, i - 1, mask)
        args = dict(BASE8); args.update(kw); args['reb_offset'] = 1; args['label'] = f'{tag}/{sc}'
        d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
        m, eq, hp = run(d, fn, ret_diag=True, **args)
        de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None) if r.index.tz is not None else r.index
        res[sc] = r; print(f'{tag} {sc}: Sh {m["sharpe"]:.2f} CAGR {m["cagr"]*100:.1f}% MDD {m["mdd"]*100:.1f}%', flush=True)
    pd.DataFrame(res).to_csv(f'research/phase4_out/R2_{tag}.csv')
