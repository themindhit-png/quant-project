#!/usr/bin/env python3
"""Reverse test 2020-21 (docs/REVERSE_2020_21.md): F2+V2 book, each sleeve alone and DIAGNOSTIC regime guards through the research
engine in the R2 lag-1 configuration (signal on bar i-1, execution at bar i, BASE8 risk layer; costs x1 and x1.5) — same as
research/exp_r2.py / r4_sleeves.py. Nothing here is a parameter change of the live book (guards = diagnostics, data-snooping caveat).
  reverse : START=2020-01-01 OOS=2020-05-01 EVAL_END=2022-01-01 PRED_SUF8=_rev PRED_SUF37=_rev END="2026-10-26 23:00" OUT_TAG=REV python3 research/rev_book.py
  forward : END="2026-10-26 23:00" PRED_SUF8=_oct_noconst_r4 PRED_SUF37=_noage_oct_r4 OUT_TAG=FWD python3 research/rev_book.py   (F2V2 x1 must give 1.99 = R4_E_V2)
Guards (decided before running; decision-time data only, evaluated on bar i-1 like the book): the book is dollar-neutral after caps
(engine restores neutrality), so "scale shorts" == "scale the whole book". mult m in {0.5, 0}:
  btc15 : BTC 30-day return > +15 %            -> book x m
  alt10 : median 30-day return of the top-50 alts by 24h turnover (age >= 30 d, BTC excluded) minus BTC 30-day return > +10 pp -> book x m
Parts: env PARTS=book,sleeves,guards (default all)."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import exp_common as X
from exp_common import d, N, F2, BASE8, VOL, Held, sl_listing, sl_core, ML3n, ML7n, ML8, fund_change, fund_level, JB
from bt import run, slip_model
TAG = os.environ.get('OUT_TAG', 'REV'); PARTS = os.environ.get('PARTS', 'book,sleeves,guards').split(',')
EEND = os.environ.get('EVAL_END'); END_I = int(d.idx.searchsorted(pd.Timestamp(EEND, tz='UTC'))) if EEND else None
OUT = 'research/phase4_out/'

def renorm(w, L, S):   # verbatim research/exp_r2.py
    L2, S2 = w[w > 0].sum(), -w[w < 0].sum()
    if L2 > 0: w[w > 0] *= L / L2
    if S2 > 0: w[w < 0] *= S / S2
    return w
def sides(w): return w[w > 0].sum(), -w[w < 0].sum()
def v2(fn, mult=0.5):   # verbatim research/exp_r2.py
    def g(dd, i, mask):
        w = fn(dd, i, mask).copy(); nz = np.flatnonzero((w != 0) & np.isfinite(VOL[i]))
        if len(nz) < 10: return w
        L, S = sides(w); w[nz[np.argsort(VOL[i][nz])[int(len(nz) * 2 / 3):]]] *= mult
        return renorm(w, L, S)
    return g

# ---- regime series (hourly, known at bar i close)
cff = d.cff; T = cff.shape[0]
btc30 = np.full(T, np.nan); btc30[720:] = cff[720:, JB] / cff[:-720, JB] - 1
alt30 = np.full(T, np.nan)
for i in range(720, T):
    if d.hh[i] % 8: continue
    t = d.t24[i]; ok = d.valid[i] & (d.age[i] >= 720) & np.isfinite(t); ok[JB] = False
    idx = np.flatnonzero(ok)
    if len(idx) < 10: continue
    idx = idx[np.argsort(-t[idx])][:50]; r = cff[i, idx] / cff[i - 720, idx] - 1; alt30[i] = np.nanmedian(r)
alt30 = pd.Series(alt30).ffill().values
SPREAD = alt30 - btc30
FLAG = {'btc15': btc30 > 0.15, 'alt10': SPREAD > 0.10}
def guard(fn, flag, m):
    def g(dd, i, mask):
        w = fn(dd, i, mask)
        return w * m if flag[i] else w
    return g
def book(): return v2(F2())

COST = {'x1': {}, 'x1.5': dict(fee_bps=8.25, maker_fee_bps=3.0, slip_fn=slip_model(scale=0.45))}
def go(label, f0, cost='x1'):
    def fn(dd, i, mask, f0=f0): return f0(dd, i - 1, mask)
    args = dict(BASE8); args.update(COST[cost]); args['reb_offset'] = 1; args['label'] = label
    d.start_i = int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC')))
    m, eq, hp = run(d, fn, ret_diag=True, end_i=END_I, **args)
    de = eq.groupby(eq.index.floor('D')).last(); r = de.pct_change().dropna(); r.index = r.index.tz_localize(None)
    print(f'{TAG} {label} {cost}: Sh {m["sharpe"]:.2f} CAGR {m["cagr"]*100:.1f}% MDD {m["mdd"]*100:.1f}% gross {m["avg_gross"]:.2f}', flush=True)
    return r

# daily regime/factor series for the report: alts-vs-BTC factor = equal-weight daily return of top-50 alts (by turnover at previous day end) minus BTC
if 'book' in PARTS:
    di = np.flatnonzero(d.hh % 24 == 0); rows = []
    for a, b in zip(di[:-1], di[1:]):
        t = d.t24[a]; ok = d.valid[a] & d.valid[b] & (d.age[a] >= 720) & np.isfinite(t); ok[JB] = False; idx = np.flatnonzero(ok)
        if len(idx) < 10: continue
        idx = idx[np.argsort(-t[idx])][:50]; ra = np.nanmean(cff[b, idx] / cff[a, idx] - 1); rb = cff[b, JB] / cff[a, JB] - 1
        rows.append((d.idx[a].tz_localize(None), ra - rb, rb, btc30[a], SPREAD[a]))
    pd.DataFrame(rows, columns=['date', 'alt_minus_btc', 'btc', 'btc30', 'alt30_minus_btc30']).set_index('date').to_csv(OUT + f'{TAG}_factor.csv')
    res = {f'F2V2_{c}': go('F2V2', book(), c) for c in COST}
    pd.DataFrame(res).to_csv(OUT + f'{TAG}_book.csv')
if 'sleeves' in PARTS:
    SL = dict(listing=lambda: Held(sl_listing), core=lambda: Held(sl_core), ml3=lambda: Held(ML3n), ml7=lambda: Held(ML7n), ml8=lambda: ML8,
              fchg=lambda: Held(fund_change), fcarry=lambda: fund_level)
    pd.DataFrame({k: go(k, f()) for k, f in SL.items()}).to_csv(OUT + f'{TAG}_sleeves.csv')
if 'guards' in PARTS:
    res = {}
    for gname, flag in FLAG.items():
        print(f'{TAG} flag {gname}: share of hours on (eval window) {np.nanmean(flag[int(d.idx.searchsorted(pd.Timestamp(X.OOS, tz='UTC'))):END_I])*100:.1f}%', flush=True)
        for m in (0.5, 0.0):
            for c in COST:
                res[f'{gname}_m{m}_{c}'] = go(f'{gname} m{m}', guard(book(), flag, m), c)
    pd.DataFrame(res).to_csv(OUT + f'{TAG}_guards.csv')
print('DONE')
