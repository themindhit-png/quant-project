#!/usr/bin/env python3
"""Reg. 5 realism step R-3 (docs/REG5_PLAN.md): what does the rebalance minute cost? Price of executing the 8h-ML legs (and the whole
book) at hh:00 / hh:05 instead of hh:10 in the rebalance hours 01/09/17 UTC, from 5-minute price snapshots (2022-06 .. 2026-10).
Gain (bps per unit traded) = sum_j dw_j * (P10_j / Pa_j - 1) / sum_j |dw_j|  (positive = the earlier minute buys cheaper / sells dearer).
dw: 8h-ML sleeve = change of its raw weights between consecutive decisions; book = change of held weights between rebalances
(logs/f2_weights_BR.parquet from `FUND_SRC=bybit p5.py run base --tag BR --rec`). Snapshots: env K5_DIR (o0/o5/o10 = price at minute 0/5/10 of the hour).
Run: python3 research/p5_rebal_minute.py"""
import os, sys, glob, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT); sys.path.insert(0, ROOT)
os.environ.update(END='2026-10-26 23:00', PRED_SUF8='_oct_noconst_r4', PRED_SUF37='_noage_oct_r4')
import exp_common as X
from exp_common import d, N, ML8
K5 = os.environ.get('K5_DIR', '/home/user/edge-lab/data/k5b')
T = d.ret.shape[0]; reb = np.flatnonzero((d.hh % 8 == 1) & (d.idx >= pd.Timestamp('2022-06-01', tz='UTC')))
PX = {k: np.full((len(reb), N), np.nan) for k in ('o0', 'o5', 'o10')}; pos_of = {t: n for n, t in enumerate(d.idx[reb])}; nsym = 0
for f in sorted(glob.glob(os.path.join(K5, '*.parquet'))):
    s = os.path.basename(f)[:-8]
    if s not in set(d.cols): continue
    j = int(np.flatnonzero(d.cols == s)[0]); x = pd.read_parquet(f, columns=['o0', 'o5', 'o10']); x = x[x.index.isin(d.idx[reb])]
    rr = np.array([pos_of[t] for t in x.index])
    if len(rr) == 0: continue
    for k in PX: PX[k][rr, j] = x[k].values
    nsym += 1
print(f'5-min snapshots: {nsym} symbols, {len(reb)} rebalance hours', flush=True)
def gains(DW):
    out = []
    for n in range(len(reb)):
        dw = DW[n]; ok = np.isfinite(PX['o10'][n]) & np.isfinite(PX['o0'][n]) & np.isfinite(PX['o5'][n]) & (dw != 0)
        a = np.abs(dw[ok]).sum()
        if a <= 0: out.append((np.nan, np.nan, 0.0, 0.0)); continue
        cov = a / max(np.abs(dw).sum(), 1e-12)
        out.append(tuple((dw[ok] * (PX['o10'][n][ok] / PX[k][n][ok] - 1)).sum() / a * 1e4 for k in ('o0', 'o5')) + (a, cov))
    return pd.DataFrame(out, columns=['g00', 'g05', 'traded', 'cover'], index=d.idx[reb].tz_localize(None))
# 8h ML sleeve: decision at bar reb-1 (hh % 8 == 0), executed in hour reb
DW = np.zeros((len(reb), N)); prev = None
for n, i in enumerate(reb):
    d.prev_mask = None; w = ML8(d, i - 1, None); DW[n] = w - (prev if prev is not None else 0); prev = w
G8 = gains(DW)
def show(G, lab):
    G = G[G.traded > 0]; y = G.groupby(G.index.year)
    tab = pd.DataFrame({'g00_bps': y.apply(lambda g: np.average(g.g00, weights=g.traded)), 'g05_bps': y.apply(lambda g: np.average(g.g05, weights=g.traded)),
                        'n': y.size(), 'cover': y.cover.mean()})
    print(f'\n{lab}: gain of executing at hh:00 / hh:05 instead of hh:10, bps per unit traded (traded-weighted), by year'); print(tab.round(2).to_string())
    print(f'all: hh:00 {np.average(G.g00, weights=G.traded):+.2f} bps, hh:05 {np.average(G.g05, weights=G.traded):+.2f} bps; years > 0 (hh:00): {(tab.g00_bps > 0).sum()}/{len(tab)}')
    return tab
t8 = show(G8, '8h-ML legs')
wf = os.environ.get('WEIGHTS', 'logs/f2_weights_BR.parquet')
if os.path.exists(wf):
    W = pd.read_parquet(wf); W['time'] = pd.to_datetime(W.time); piv = W.pivot_table(index='time', columns='symbol', values='w', aggfunc='sum').fillna(0.0)
    piv = piv.reindex(columns=d.cols, fill_value=0.0); piv = piv.reindex(d.idx[reb]).fillna(0.0)   # held weights after each rebalance
    H = piv.values; DB = np.vstack([H[:1], np.diff(H, axis=0)])
    tb = show(gains(DB), 'whole book (held-weight changes)')
G8.to_csv('research/phase4_out/P5_rebal_minute_ml8.csv'); print('DONE')
